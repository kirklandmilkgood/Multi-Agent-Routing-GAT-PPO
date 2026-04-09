import osmnx as ox
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# 設定城市
county_name = "Taichung, Taiwan"
place_name = [f"Central District, {county_name}", f"East District, {county_name}", f"North District, {county_name}", 
              f"West District, {county_name}",f"South District, {county_name}", f"Xitun District, {county_name}", 
              f"Beitun District, {county_name}", f"Nantun District, {county_name}"]
print(f"正在從 OpenStreetMap 下載 {county_name} 的真實路網...")

# 下載路網 (network_type='drive' ，只抓取汽車可行駛的道路)
G = ox.graph_from_place(place_name, network_type='drive')

# 將圖形轉為無向圖 (確保雙向連通)
G = ox.convert.to_undirected(G)
print(f"下載完成！總節點數: {len(G.nodes)} | 總邊數: {len(G.edges)}")

is_connected = nx.is_connected(G)
if not is_connected:
    # 取得最大的連通分量 (LCC)
    # nx.connected_components 回傳所有孤島，取節點數最多的那一個
    largest_cc = max(nx.connected_components(G), key=len)
    
    # 建立子圖
    G = G.subgraph(largest_cc).copy()
    print(f"已過濾孤島！剩餘節點數: {len(G.nodes)} | 剩餘邊數: {len(G.edges)}")
else:
    print("路網本身已完全連通，無需處理。")


# 視覺化路網資料
print("正在進行路網視覺化...")

# bgcolor: 背景顏色 ('k' 為黑色，'w' 為白色)
# edge_color: 道路顏色 ('#ffffff' 為白色，)
# edge_linewidth: 路邊粗細
# node_size: 節點大小
# node_color: 節點顏色
fig, ax = ox.plot_graph(G, 
                        bgcolor='k',           # 黑底
                        edge_color='#ffffff',   # 白線道路
                        edge_linewidth=0.8,     # 道路粗細
                        node_size=5,           # 隱藏節點，突顯道路
                        node_color="#ec93d7",  # 設定節點顏色為紅色
                        node_zorder=3,         # 確保節點繪製在道路線條的上方
                        show=False, 
                        close=False)
output_img = "../dataset/Taichung.png"
fig.savefig(output_img, dpi=300, bbox_inches='tight', pad_inches=0)
print(f"路網視覺化圖片已儲存為 {output_img}")
plt.show()

# 轉換為 Excel 格式 (from, to, weight)
edges_data = []
# OSMnx 的節點 ID 為一長串經緯度雜湊碼，將其重新映射為 0 ~ N-1
node_mapping = {old_id: new_id for new_id, old_id in enumerate(G.nodes)}

for u, v, data in G.edges(data=True):
    source = node_mapping[u]
    target = node_mapping[v]
    
    # 取得道路真實長度 (公尺) 當作 travel cost
    raw_weight = data.get('length', 10.0)  
    edges_data.append({'From': source, 'To': target, 'Weight': raw_weight})

df_edges = pd.DataFrame(edges_data)

# 執行 min-max 縮放至 1~10 區間
min_w = df_edges['Weight'].min()
max_w = df_edges['Weight'].max()
# 公式： scaled = 1 + 9 * (x - min) / (max - min)
if max_w > min_w:
    scaled_weights = 1 + 9 * (df_edges['Weight'] - min_w) / (max_w - min_w)
    # 四捨五入並強制轉為整數
    df_edges['Weight'] = scaled_weights.round().astype(int)
else:
    # 若整張圖的邊都一樣長，統一設為 1
    df_edges['Weight'] = 1

# 輸出 Excel 檔案
output_filename = f"../dataset/Large_network.xlsx"
df_edges.to_excel(output_filename, index=False)
print(f"檔案已儲存為 {output_filename}")