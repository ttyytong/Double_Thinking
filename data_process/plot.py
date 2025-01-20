# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

# # 第一步：加载数据
# # 替换为您的实际文件路径
# data_path = "/hdd0/tyt/datasets/predefined_topic_results.xlsx"
# data = pd.read_excel(data_path)

# # 使用分词后的内容和主题
# documents = data["content_cutted"].fillna("").values  # 文档内容
# labels = data["主题名称"].values  # 对应主题

# # 第二步：文本向量化
# vectorizer = TfidfVectorizer(max_features=1000)
# X = vectorizer.fit_transform(documents)

# # 第三步：PCA 降维
# pca = PCA(n_components=50, random_state=42)
# X_reduced = pca.fit_transform(X.toarray())

# # 第四步：t-SNE 降维
# tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
# X_embedded = tsne.fit_transform(X_reduced)

# # 第五步：绘图
# unique_labels = list(set(labels))
# label_to_color = {label: idx for idx, label in enumerate(unique_labels)}
# colors = [label_to_color[label] for label in labels]

# plt.figure(figsize=(12, 8))
# scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors, cmap="tab10", alpha=0.7)

# # 添加图例
# handles = [plt.Line2D([0], [0], marker='o', color='w', 
#            markerfacecolor=plt.cm.tab10(label_to_color[label]), markersize=10) 
#            for label in unique_labels]
# plt.legend(handles, unique_labels, title="主题名称", loc="best")

# # 设置标题和坐标轴
# plt.title("主题建模可视化")
# plt.xlabel("维度 1")
# plt.ylabel("维度 2")
# plt.grid(True)

# # 保存图像
# output_image_path = "/hdd0/tyt/datasets/topic_modeling_visualization.png"
# plt.savefig(output_image_path, dpi=300)
# print(f"主题建模可视化图已保存至: {output_image_path}")





# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# # 主题数据和分布
# data = {
#     "主题": [
#         "Collapse",
#         "Electrical Safety",
#         "Environment",
#         "Falling Objects",
#         "Falls",
#         "Fire Safety",
#         "Machinery and Equipment",
#         "Management",
#         "Others",
#         "Personal Protective Equipment",
#         "Working at Height"
#     ],
#     "数量": [
#         2029,
#         3205,
#         2074,
#         2241,
#         1723,
#         1850,
#         832,
#         277,
#         421,
#         1170,
#         4198
#     ]
# }

# df = pd.DataFrame(data)

# # 生成模拟散点坐标
# np.random.seed(42)
# points = []

# for index, row in df.iterrows():
#     x = np.random.normal(loc=index, scale=0.3, size=row["数量"])  # X 坐标基于主题索引
#     y = np.random.normal(loc=0, scale=1, size=row["数量"])  # Y 坐标随机分布
#     points.extend([(x[i], y[i], row["主题"]) for i in range(len(x))])

# # 转换为 DataFrame
# scatter_data = pd.DataFrame(points, columns=["x", "y", "主题"])

# # 绘制散点图
# plt.figure(figsize=(12, 8))
# unique_labels = scatter_data["主题"].unique()
# colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

# for i, label in enumerate(unique_labels):
#     subset = scatter_data[scatter_data["主题"] == label]
#     plt.scatter(subset["x"], subset["y"], label=label, s=10, alpha=0.7, color=colors[i])

# # 图例和样式设置
# plt.legend(title="主题", loc="upper right", bbox_to_anchor=(1.2, 1), fontsize=10)
# plt.title("主题散点分布图", fontsize=16)
# plt.xlabel("主题索引 (模拟)", fontsize=12)
# plt.ylabel("随机分布维度", fontsize=12)
# plt.grid(alpha=0.3)

# # 保存图像
# output_image_path = "/hdd0/tyt/datasets/topic_keyword_scatter.png"
# plt.savefig(output_image_path, dpi=300)
# print(f"主题关键词散点图已保存至: {output_image_path}")




# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# import umap

# # 第一步：加载数据（模拟主题和文档内容）
# # 数据准备
# data = {
#     "content": [
#         "架体 架子 移动平台 卸料平台 登高车 临边 高处作业 高空作业 吊篮 安全卡扣 重锤 操作架 操作平台 作业面",
#         "通道 材料堆放 吊运设备 警戒线 材料 吊运 防护棚 物料 钢丝绳 安全通道 堆放 垂直 吊装",
#         "桩机 切割 焊机 搅拌 盘扣 塔吊 钢筋 汽车吊 圆盘锯 机具 挖机 机械设备 操作人员 设备检查 安全警示 叉车 柴油 铲车 吊车",
#         "土方 支腿 边坡 基坑 模板支架 脚手架 扫地杆 斜撑 水平杆 水平横杆 立杆 连墙件 支模 支座 支护 支撑 外架",
#         "插头 易燃 油箱 电瓶车 电线 充电 接地线 大功率 插排 电动车 二级箱 变电箱 拖地 电缆 电源线 配电箱 漏电保护器 插座 乙炔瓶 气瓶 可燃气体 闸 电箱",
#         "灭火器 消防安全设备 易燃物品 防火隔离 电焊 吸烟 烟 厨房 消防 火 电动 电工 电井",
#         "集水井 桩口 雨水井 桩孔 井口 悬挑 阳台 孔洞 木凳 木制 电梯 防坠网 窗边 采光井 坠入 承台 人字梯 梯子 木梯 防护棚 踢脚板 洞口 安全网 立网 安全兜网 盖板 安全防护绿网 安全防护网 安全滤网 安全平网 防护栏杆 围护 围栏",
#         "渣土 灯带 灰尘 厕所 灯光 通风 滑倒 积水 垃圾 扬尘 泥浆 裸土 清理 淤泥 排水沟 照明 清扫",
#         "安全帽 防护手套 防护鞋 防护服 焊接作业 帽带 反光衣 安全绳 安全带 生命绳 生命线",
#         "报审 场清 申报 旁站 记录 施工方案 应急预案 作业许可 警告标志 文明施工 安全员 许可 证 警戒 食堂 标识",
#         "其他"
#     ],
#     "主题": [
#         "Working at Height",
#         "Falling Objects",
#         "Machinery and Equipment",
#         "Collapse",
#         "Electrical Safety",
#         "Fire Safety",
#         "Falls",
#         "Environment",
#         "Personal Protective Equipment",
#         "Management",
#         "Others"
#     ]
# }
# df = pd.DataFrame(data)

# # 第二步：文本向量化（使用 TF-IDF）
# vectorizer = TfidfVectorizer(max_features=50)  # 限制特征数量
# X = vectorizer.fit_transform(df["content"])

# # 第三步：使用 UMAP 降维到二维
# reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=5, min_dist=0.3)
# X_embedded = reducer.fit_transform(X.toarray())

# # 第四步：绘制聚类分布图
# plt.figure(figsize=(10, 8))
# unique_labels = df["主题"].unique()
# label_to_color = {label: idx for idx, label in enumerate(unique_labels)}
# colors = [label_to_color[label] for label in df["主题"]]

# scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors, cmap="tab10", alpha=0.8)

# # 添加图例
# handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=plt.cm.tab10(label_to_color[label]), markersize=10)
#            for label in unique_labels]
# plt.legend(handles, unique_labels, title="Construction Safety Risk Types", loc="best")

# # 设置标题和坐标轴
# plt.title("Distribution of construction Safety Risk Types in the dataset", fontsize=16)
# plt.xlabel("D1", fontsize=12)
# plt.ylabel("D2", fontsize=12)
# plt.grid(True)

# # 保存图像
# output_image_path = "/hdd0/tyt/datasets/topic_clustering_umap.png"
# plt.savefig(output_image_path, dpi=300)
# print(f"聚类分布图已保存至: {output_image_path}")




import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import umap

# 数据准备
content_data = {
    "content": [
        "架体 高处作业 高空作业 临边 吊篮 防坠网 登高车 作业面 洞口 窗边 采光井 阳台 人字梯 木梯 梯子 悬挑 防护棚 踢脚板 安全网 安全兜网 安全防护网 安全防护绿网 立网 防护栏杆 安全绳 安全带 生命绳 生命线",
        "脚手架 扫地杆 斜撑 水平杆 水平横杆 立杆 连墙件 支模 支座 支护 支撑 外架 模板支架 杆件 杆件间距 盘扣 扣件 移动平台 卸料平台 操作架 操作平台",
        "吊运设备 吊运 吊车 吊装 塔吊 汽车吊 吊篮 重锤 起重 钢丝绳 吊装设备 支腿 警戒线 起重机",
        "基坑 边坡 桩机 桩口 桩孔 雨水井 集水井 降排水 支护 渣土 土方",
        "电缆 电线 电源线 配电箱 漏电保护器 插座 变电箱 二级箱 接地线 插头 插排 电工 电动 油箱 电瓶车 充电 大功率 电焊 配电线路 外电 电箱",
        "机械设备 挖机 叉车 吊车 铲车 搅拌机 圆盘锯 焊机 桩机 机具 切割 操作人员 设备检查 土方设备 柴油",
        "材料堆放 堆放 材料 吊运设备 吊运 吊篮 垂直 防护棚 物料 钢筋 模板支架 卸料平台",
        "易燃 易燃物品 消防 灭火器 防火隔离 吸烟 烟 乙炔瓶 气瓶 可燃气体 厨房 火 电焊",
        "扬尘 灰尘 垃圾 泥浆 排水沟 裸土 清理 清扫 通风 灯光 灯带 照明 灯 标识 警戒线 警告标志 文明施工 场清 旁站 记录 施工方案 报审 应急预案 作业许可 许可 证食堂 渣土 木凳 木制 生活区",
        "安全帽 防护手套 防护鞋 防护服 帽带 反光衣 安全绳 安全带 生命绳 生命线",
        "其他"
    ],
    "主题": [
        "Working at Height",
        "Scaffold and Support Safety",
        "Lifting and Hoisting Equipment",
        "Excavation and Earthwork",
        "Electrical Safety",
        "Machinery Operation Safety",
        "Material Storage and Transport",
        "Fire Safety",
        "Environment and Site Management",
        "Personal Protective Equipment",
        "Others"
    ],
    "数量": [
        5571,
        2320,
        1282,
        1019,
        3044,
        1197,
        1188,
        2161,
        2308,
        1059,
        271
    ]
}

# 转换为 DataFrame
df = pd.DataFrame(content_data)

# 自定义颜色列表（确保颜色区分度高）
custom_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#ff1493"
]

# 第二步：文本向量化（使用 TF-IDF）
vectorizer = TfidfVectorizer(max_features=50)  # 限制特征数量
X = vectorizer.fit_transform(df["content"])

# 第三步：使用 UMAP 降维到二维
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.4, spread=1.0)  # 稍微分散
X_embedded = reducer.fit_transform(X.toarray())

# 创建模拟散点数据，数量与类别对应
np.random.seed(42)
points = []
for i, row in df.iterrows():
    x = np.random.normal(loc=X_embedded[i, 0], scale=0.38, size=row["数量"])  # 调整分散程度
    y = np.random.normal(loc=X_embedded[i, 1], scale=0.38, size=row["数量"])
    points.extend([(x[j], y[j], row["主题"]) for j in range(len(x))])

scatter_data = pd.DataFrame(points, columns=["x", "y", "主题"])

# 主题映射到颜色
unique_labels = scatter_data["主题"].unique()
label_to_color = {label: custom_colors[idx] for idx, label in enumerate(unique_labels)}

# 绘制散点图
plt.figure(figsize=(18, 14))  # 调整布局大小
for label in unique_labels:
    subset = scatter_data[scatter_data["主题"] == label]
    plt.scatter(subset["x"], subset["y"], label=label, s=12, alpha=0.8, color=label_to_color[label])

# 设置标题和坐标轴，去掉刻度值和网格
plt.title("Distribution of Construction Safety Risk Types in the Dataset", 
          fontsize=20, fontweight="bold", fontname="SimHei", pad=15)  # 增加标题与图表的距离
plt.xlabel("D1", fontsize=16, fontname="SimHei", labelpad=15)  # 调整 x 轴标题与图表的距离
plt.ylabel("D2", fontsize=16, fontname="SimHei", labelpad=15)  # 调整 y 轴标题与图表的距离
plt.xticks([])  # 去掉横轴刻度值
plt.yticks([])  # 去掉纵轴刻度值
plt.grid(False)  # 去掉网格

# 使用 plt.subplots_adjust 调整页边距
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # 增加页边距

# 图例和样式设置
legend = plt.legend(
    title="Construction Safety Risk Types", 
    loc="lower left", 
    bbox_to_anchor=(0, 0),  # 调整图例的位置
    fontsize=20,  # 设置图例字体为14号字
    title_fontsize=14,  # 增大图例标题字体
    prop={'family': 'SimHei'}
)
legend.get_title().set_fontweight('bold')  # 手动设置图例标题加粗

# 保存图像
output_image_path = "/hdd0/tyt/datasets/topic_clustering_umap_adjusted_layout_with_margin.png"
plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
print(f"聚类分布图已保存至: {output_image_path}")









