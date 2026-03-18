
import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeEmbedding(nn.Module):
    def __init__(self, input_dim=3, embed_dim=64):
        super(NodeEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, embed_dim)

    def forward(self, node_feats):
        return self.fc(node_feats)

class AgentEmbedding(nn.Module):
    def __init__(self, embed_dim=64):
        super(AgentEmbedding, self).__init__()
        self.agent_fc = nn.Linear(2, embed_dim)

    def forward(self, agent_pos, budget_left):
        x = torch.cat([agent_pos.unsqueeze(-1), budget_left.unsqueeze(-1)], dim=-1)
        return self.agent_fc(x)

class DDTM(nn.Module):
    def __init__(self, num_nodes, embed_dim=64, nhead=4, num_layers=2):
        super(DDTM, self).__init__()
        self.node_embedding = NodeEmbedding(input_dim=3, embed_dim=embed_dim)
        self.agent_embedding = AgentEmbedding(embed_dim=embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_fc = nn.Linear(embed_dim, num_nodes)

    def forward(self, node_feats, agent_pos, budget_left, visited_mask):
        B, N, _ = node_feats.size()

        node_embed = self.node_embedding(node_feats)       
        memory = self.encoder(node_embed)                  

        logits_list = []
        for i in range(agent_pos.size(1)):
            agent_embed = self.agent_embedding(agent_pos[:, i], budget_left[:, i])  
            tgt = agent_embed.unsqueeze(1)                  
            out = self.decoder(tgt, memory)                 
            out = self.output_fc(out.squeeze(1))            
            out[visited_mask == 1] = -1e9                   
            logits_list.append(out)

        logits = torch.stack(logits_list, dim=1)            
        return logits

    def select_action(self, logits, temperature=1.0):
        probs = F.softmax(logits / temperature, dim=-1)

        
        if torch.isnan(probs).any() or (probs.sum(-1) == 0).any():
            print("[Fallback] Invalid logits detected — using uniform distribution")
            
            mask = (logits != -1e9).float()
            probs = mask / mask.sum(-1, keepdim=True).clamp(min=1e-8)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

