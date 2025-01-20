# import os
# import pandas as pd
# import re
# from bertopic import BERTopic
# from sentence_transformers import SentenceTransformer
# from umap import UMAP
# from hdbscan import HDBSCAN
# from bertopic.vectorizers import ClassTfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Step 1 - 数据准备
# file_path = '/hdd0/tyt/datasets/filtered_image_filenames2.xlsx'

# # 读取Excel文件
# data = pd.read_excel(file_path, engine='openpyxl').astype(str)

# # 查看数据内容
# print(data.head())

# # 停用词路径
# stop_file = "/ssd0/tyt/CogVLM/data_process/stop_words.txt"

# # 分词及去除停用词
# def chinese_word_cut(mytext):
#     stop_list = []
#     try:
#         with open(stop_file, encoding='utf-8') as stopword_list:
#             stop_list = [line.strip() for line in stopword_list]
#     except FileNotFoundError:
#         print(f"Error: Stop file '{stop_file}' not found.")
#     word_list = []
#     words = re.findall(r'[\u4e00-\u9fa5]+', mytext)  # 匹配中文词语
#     for word in words:
#         if word in stop_list or len(word) < 2:
#             continue
#         word_list.append(word)
#     return " ".join(word_list)

# # 对文本进行分词和去除停用词
# data["content_cutted"] = data["文件名"].apply(chinese_word_cut)

# # Step 2 - BERTopic建模

# # 加载SentenceTransformer模型
# embedding_model = SentenceTransformer('/hdd0/tyt/huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# # 使用UMAP进行降维
# umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

# # 使用HDBSCAN进行聚类
# hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', prediction_data=True)

# # 使用c-TF-IDF模型进行主题表示
# vectorizer_model = CountVectorizer(stop_words="english")
# ctfidf_model = ClassTfidfTransformer()

# # 构建BERTopic模型
# topic_model = BERTopic(
#     embedding_model=embedding_model,
#     umap_model=umap_model,
#     hdbscan_model=hdbscan_model,
#     vectorizer_model=vectorizer_model,
#     ctfidf_model=ctfidf_model,
#     # diversity=0.5,
#     nr_topics='none',
#     top_n_words=10
# )

# # Step 3 - 训练模型并提取主题
# filtered_text = data["content_cutted"].tolist()
# topics, probabilities = topic_model.fit_transform(filtered_text)

# # 查看每个主题的词汇
# print(topic_model.get_topic_freq())

# # 查看第一个主题的具体词汇分布
# print(topic_model.get_topic(0))

# # 查看每个文档的主题分布
# document_info = topic_model.get_document_info(filtered_text)
# print(document_info.head())

# # Step 4 - 保存结果
# output_path = "/hdd0/tyt/datasets/topic_results.xlsx"
# document_info.to_excel(output_path, index=False)
# print(f"主题建模结果已保存至: {output_path}")




# import os
# import pandas as pd
# import re
# from bertopic import BERTopic
# from sentence_transformers import SentenceTransformer
# from umap import UMAP
# from hdbscan import HDBSCAN
# from bertopic.vectorizers import ClassTfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer

# # 禁用tokenizers并行警告
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Step 1 - 数据准备
# file_path = '/hdd0/tyt/datasets/filtered_image_filenames2.xlsx'

# # 尝试读取Excel文件
# try:
#     data = pd.read_excel(file_path, engine='openpyxl').astype(str)
# except FileNotFoundError:
#     raise FileNotFoundError(f"Error: File '{file_path}' not found. Please check the path.")

# # 停用词路径
# stop_file = "/ssd0/tyt/CogVLM/data_process/stop_words.txt"

# # 分词及去除停用词
# def chinese_word_cut(mytext):
#     stop_list = []
#     try:
#         with open(stop_file, encoding='utf-8') as stopword_list:
#             stop_list = [line.strip() for line in stopword_list]
#     except FileNotFoundError:
#         print(f"Warning: Stop file '{stop_file}' not found. Proceeding without stopword removal.")
    
#     word_list = []
#     words = re.findall(r'[\u4e00-\u9fa5]+', mytext)  # 匹配中文词语
#     for word in words:
#         if word in stop_list or len(word) < 2:  # 去掉停用词及长度小于2的词
#             continue
#         word_list.append(word)
#     return " ".join(word_list)

# # 对文本进行分词和去除停用词
# data["content_cutted"] = data["文件名"].apply(chinese_word_cut)

# # Step 2 - BERTopic建模

# # 加载SentenceTransformer模型
# embedding_model_path = '/hdd0/tyt/huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
# try:
#     embedding_model = SentenceTransformer(embedding_model_path)
# except Exception as e:
#     raise RuntimeError(f"Error loading embedding model from '{embedding_model_path}': {e}")

# # 使用HDBSCAN进行聚类
# hdbscan_model = HDBSCAN(
#     min_cluster_size=5,  # 降低最小聚类大小
#     min_samples=2,       # 增加样本密度敏感性
#     metric='euclidean', 
#     prediction_data=True, 
#     core_dist_n_jobs=1
# )

# # 使用UMAP进行降维
# umap_model = UMAP(
#     n_neighbors=25,  # 增大邻居数量
#     n_components=3,  # 减少降维维度
#     min_dist=0.1, 
#     metric='cosine', 
#     random_state=42
# )

# # 使用c-TF-IDF模型进行主题表示
# # 修改为中文停用词
# try:
#     with open(stop_file, encoding='utf-8') as stopword_list:
#         stop_list = [line.strip() for line in stopword_list]
# except FileNotFoundError:
#     print(f"Warning: Stop file '{stop_file}' not found. Proceeding without stopword removal.")
#     stop_list = []

# vectorizer_model = CountVectorizer(
#     token_pattern=r'[\u4e00-\u9fa5]{2,}',  # 匹配连续两个及以上的中文字符
#     stop_words=stop_list,                  # 使用自定义的中文停用词列表
#     max_features=5000                      # 限制特征数为 5000
# )
# ctfidf_model = ClassTfidfTransformer()

# # 构建BERTopic模型
# topic_model = BERTopic(
#     embedding_model=embedding_model,
#     umap_model=umap_model,
#     hdbscan_model=hdbscan_model,
#     vectorizer_model=vectorizer_model,
#     ctfidf_model=ctfidf_model,
#     nr_topics=15,  # 设置为15类
#     top_n_words=2  # 输出主题词减少为4个
# )

# # Step 3 - 训练模型并提取主题
# filtered_text = data["content_cutted"].tolist()
# topics, probabilities = topic_model.fit_transform(filtered_text)

# # 查看每个主题的词汇及频率
# print("主题频率:")
# print(topic_model.get_topic_freq())

# # 查看第一个主题的具体词汇分布
# print("第一个主题的具体词汇分布:")
# print(topic_model.get_topic(0))

# # 获取每个文档的主题分布
# document_info = topic_model.get_document_info(filtered_text)
# print("文档主题分布示例:")
# print(document_info.head())

# # Step 4 - 保存结果
# output_path = "/hdd0/tyt/datasets/topic_results.xlsx"
# try:
#     document_info.to_excel(output_path, index=False)
#     print(f"主题建模结果已保存至: {output_path}")
# except Exception as e:
#     raise RuntimeError(f"Error saving results to '{output_path}': {e}")






# import os
# import pandas as pd
# import re
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.decomposition import NMF
# from sentence_transformers import SentenceTransformer
# from hdbscan import HDBSCAN
# from umap import UMAP
# from bertopic import BERTopic

# # Step 1 - 数据准备
# file_path = '/hdd0/tyt/datasets/filtered_image_filenames3.xlsx'

# # 尝试读取Excel文件
# try:
#     data = pd.read_excel(file_path, engine='openpyxl').astype(str)
# except FileNotFoundError:
#     raise FileNotFoundError(f"Error: File '{file_path}' not found. Please check the path.")

# # 停用词路径
# stop_file = "/ssd0/tyt/CogVLM/data_process/stop_words.txt"

# # 分词及去除停用词
# def chinese_word_cut(mytext):
#     stop_list = []
#     try:
#         with open(stop_file, encoding='utf-8') as stopword_list:
#             stop_list = [line.strip() for line in stopword_list]
#     except FileNotFoundError:
#         print(f"Warning: Stop file '{stop_file}' not found. Proceeding without stopword removal.")
    
#     word_list = []
#     words = re.findall(r'[\u4e00-\u9fa5]+', mytext)  # 匹配中文词语
#     for word in words:
#         if word in stop_list or len(word) < 2:  # 去掉停用词及长度小于2的词
#             continue
#         word_list.append(word)
#     return " ".join(word_list)

# # 对文本进行分词和去除停用词
# data["content_cutted"] = data["文件名"].apply(chinese_word_cut)

# # Step 2 - 选用更适合短文本的主题建模方法

# # 选择是否使用Embedding-based方法或传统方法
# USE_EMBEDDING_BASED = True

# if USE_EMBEDDING_BASED:
#     # Embedding-based 方法
#     embedding_model_path = '/hdd0/tyt/huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
#     try:
#         embedding_model = SentenceTransformer(embedding_model_path)
#     except Exception as e:
#         raise RuntimeError(f"Error loading embedding model from '{embedding_model_path}': {e}")
    
#     # 使用HDBSCAN进行聚类
#     hdbscan_model = HDBSCAN(
#         min_cluster_size=5,  # 降低最小聚类大小
#         min_samples=2,       # 增加样本密度敏感性
#         metric='euclidean', 
#         prediction_data=True, 
#         core_dist_n_jobs=1
#     )
    
#     # 使用UMAP进行降维
#     umap_model = UMAP(
#         n_neighbors=15,  # 调整邻居数量
#         n_components=2,  # 降低降维维度以便HDBSCAN聚类
#         min_dist=0.1, 
#         metric='cosine', 
#         random_state=42
#     )
    
#     # 构建BERTopic模型
#     topic_model = BERTopic(
#         embedding_model=embedding_model,
#         umap_model=umap_model,
#         hdbscan_model=hdbscan_model,
#         nr_topics=10,  # 设置为10类（也可根据数据动态调整）
#         top_n_words=5  # 输出主题词数量
#     )
    
#     # 提取文本内容并进行建模
#     filtered_text = data["content_cutted"].tolist()
#     topics, probabilities = topic_model.fit_transform(filtered_text)
    
#     # 查看每个主题的词汇及频率
#     print("主题频率:")
#     print(topic_model.get_topic_freq())
    
#     # 查看第一个主题的具体词汇分布
#     print("第一个主题的具体词汇分布:")
#     print(topic_model.get_topic(0))
    
#     # 获取每个文档的主题分布
#     document_info = topic_model.get_document_info(filtered_text)
#     print("文档主题分布示例:")
#     print(document_info.head())
    
# else:
#     # 传统NMF方法（非负矩阵分解）
#     vectorizer = TfidfVectorizer(
#         token_pattern=r'[\u4e00-\u9fa5]{2,}',  # 匹配连续两个及以上的中文字符
#         stop_words=stop_file if os.path.exists(stop_file) else None,  # 使用自定义停用词
#         max_features=5000  # 限制特征数
#     )
    
#     # 构建TF-IDF矩阵
#     tfidf_matrix = vectorizer.fit_transform(data["content_cutted"])
    
#     # 使用NMF进行主题建模
#     n_topics = 10  # 设置主题数
#     nmf_model = NMF(n_components=n_topics, random_state=42)
#     nmf_topic_matrix = nmf_model.fit_transform(tfidf_matrix)
    
#     # 获取主题词
#     feature_names = vectorizer.get_feature_names_out()
#     for topic_idx, topic in enumerate(nmf_model.components_):
#         print(f"主题 {topic_idx}: {' '.join([feature_names[i] for i in topic.argsort()[:-6:-1]])}")
    
#     # 分配每个文档的主题
#     dominant_topics = nmf_topic_matrix.argmax(axis=1)
#     data["dominant_topic"] = dominant_topics
#     print(data[["文件名", "dominant_topic"]].head())

# # Step 3 - 保存结果
# output_path = "/hdd0/tyt/datasets/topic_results1.xlsx"
# try:
#     if USE_EMBEDDING_BASED:
#         document_info.to_excel(output_path, index=False)
#     else:
#         data.to_excel(output_path, index=False)
#     print(f"主题建模结果已保存至: {output_path}")
# except Exception as e:
#     raise RuntimeError(f"Error saving results to '{output_path}': {e}")





# import os
# import pandas as pd
# import re
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.decomposition import NMF, LatentDirichletAllocation

# # Step 1 - 数据准备
# file_path = '/hdd0/tyt/datasets/filtered_image_filenames3.xlsx'
# try:
#     data = pd.read_excel(file_path, engine='openpyxl').astype(str)
#     print("数据加载成功，列名如下：", data.columns)
# except FileNotFoundError:
#     raise FileNotFoundError(f"Error: File '{file_path}' not found. 请检查文件路径是否正确！")

# # 检查列名是否包含 '文件名'
# if "文件名" not in data.columns:
#     raise KeyError("Error: '文件名' 列不存在，请检查数据文件是否正确！")

# # 停用词路径
# stop_file = "/ssd0/tyt/CogVLM/data_process/stop_words.txt"

# # 加载停用词并初始化 stop_list
# try:
#     with open(stop_file, encoding='utf-8') as stopword_list:
#         stop_list = [line.strip() for line in stopword_list]
#         print(f"停用词加载成功，数量: {len(stop_list)}")
# except FileNotFoundError:
#     print(f"Warning: 停用词文件 '{stop_file}' 未找到，将不使用停用词。")
#     stop_list = None  # 如果文件不存在，设置为 None

# # 定义分词函数
# def chinese_word_cut(mytext):
#     """
#     分词函数，支持移除停用词和短词
#     """
#     word_list = []
#     words = re.findall(r'[\u4e00-\u9fa5]+', mytext)  # 匹配中文词语
#     for word in words:
#         if stop_list and word in stop_list:  # 移除停用词
#             continue
#         if len(word) < 2:  # 移除长度小于2的词
#             continue
#         word_list.append(word)
#     return " ".join(word_list)

# # 分词并生成 content_cutted 列
# try:
#     data["content_cutted"] = data["文件名"].apply(chinese_word_cut)
#     print("分词完成，示例数据如下：")
#     print(data[["文件名", "content_cutted"]].head())
# except Exception as e:
#     raise RuntimeError(f"分词处理出错: {e}")

# # 检查 content_cutted 列是否生成
# if "content_cutted" not in data.columns:
#     raise KeyError("Error: 'content_cutted' 列未生成，分词可能失败！")

# # 输出调试信息
# print("分词后的数据框列名：", data.columns)
# print("数据样例：")
# print(data.head())

# # Step 2 - 特征表示（TF-IDF 或 CountVectorizer）
# vectorizer = TfidfVectorizer(
#     token_pattern=r'[\u4e00-\u9fa5]{2,}',  # 匹配连续两个及以上的中文字符
#     stop_words=stop_list,                  # 传入停用词列表
#     max_features=5000                      # 限制特征数量为 5000
# )

# # 生成 TF-IDF 矩阵
# try:
#     tfidf_matrix = vectorizer.fit_transform(data["content_cutted"])
#     print("TF-IDF 矩阵生成成功！")
# except Exception as e:
#     raise RuntimeError(f"TF-IDF 矩阵生成失败: {e}")

# feature_names = vectorizer.get_feature_names_out()  # 获取特征词列表

# # Step 3 - 选择主题建模方法（NMF 或 LDA）

# USE_NMF = True  # 设置为 True 使用 NMF；设置为 False 使用 LDA

# if USE_NMF:
#     # 使用 NMF 进行主题建模
#     n_topics = 15  # 自定义主题数量
#     nmf_model = NMF(n_components=n_topics, random_state=42)
#     nmf_topic_matrix = nmf_model.fit_transform(tfidf_matrix)

#     # 打印每个主题的关键词
#     print("NMF 主题关键词:")
#     for topic_idx, topic in enumerate(nmf_model.components_):
#         print(f"主题 {topic_idx}: {' '.join([feature_names[i] for i in topic.argsort()[:-6:-1]])}")

#     # 为每个文档分配主题
#     dominant_topics = nmf_topic_matrix.argmax(axis=1)
#     data["主题"] = dominant_topics

# else:
#     # 使用 LDA 进行主题建模
#     n_topics = 15  # 自定义主题数量
#     lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
#     lda_topic_matrix = lda_model.fit_transform(tfidf_matrix)

#     # 打印每个主题的关键词
#     print("LDA 主题关键词:")
#     for topic_idx, topic in enumerate(lda_model.components_):
#         print(f"主题 {topic_idx}: {' '.join([feature_names[i] for i in topic.argsort()[:-6:-1]])}")

#     # 为每个文档分配主题
#     dominant_topics = lda_topic_matrix.argmax(axis=1)
#     data["主题"] = dominant_topics

# # Step 4 - 保存结果
# output_path = "/hdd0/tyt/datasets/nmf_lda_topic_results.xlsx"
# try:
#     data.to_excel(output_path, index=False)
#     print(f"主题建模结果已保存至: {output_path}")
# except Exception as e:
#     raise RuntimeError(f"Error saving results to '{output_path}': {e}")




import os
import pandas as pd
import re

# Step 1 - 数据准备
file_path = '/hdd0/tyt/datasets/filtered_image_filenames1.xlsx'

# 读取 Excel 文件
try:
    data = pd.read_excel(file_path, engine='openpyxl').astype(str)
    print("数据加载成功，列名如下：", data.columns)
except FileNotFoundError:
    raise FileNotFoundError(f"Error: File '{file_path}' not found. 请检查路径是否正确！")

if "文件名" not in data.columns:
    raise KeyError("Error: '文件名' 列不存在，请检查数据文件是否正确！")

# 停用词路径
stop_file = "/ssd0/tyt/CogVLM/data_process/stop_words.txt"

# 加载停用词
try:
    with open(stop_file, encoding='utf-8') as stopword_list:
        stop_list = [line.strip() for line in stopword_list]
        print(f"停用词加载成功，数量: {len(stop_list)}")
except FileNotFoundError:
    print(f"Warning: 停用词文件 '{stop_file}' 未找到，将不使用停用词。")
    stop_list = None

# 定义分词函数
def chinese_word_cut(mytext):
    word_list = []
    words = re.findall(r'[\u4e00-\u9fa5]+', mytext)
    for word in words:
        if stop_list and word in stop_list:
            continue
        if len(word) < 2:
            continue
        word_list.append(word)
    return " ".join(word_list)

# 对文本进行分词
try:
    data["content_cutted"] = data["文件名"].apply(chinese_word_cut)
    print("分词完成，示例数据如下：")
    print(data[["文件名", "content_cutted"]].head())
except Exception as e:
    raise RuntimeError(f"分词处理出错: {e}")

predefined_topics = {
    # 高处作业风险
    "Working at Height": [
        "架体", "高处作业", "高空作业", "临边", "吊篮", "防坠网", "登高车", "作业面", 
        "洞口", "窗边", "采光井", "阳台", "人字梯", "木梯", "梯子", "悬挑", "防护棚", 
        "踢脚板", "安全网", "安全兜网", "安全防护网", "安全防护绿网", "立网", 
        "防护栏杆", "安全绳", "安全带", "生命绳", "生命线"
    ],

    # 脚手架与支架安全风险
    "Scaffold and Support Safety": [
        "脚手架", "扫地杆", "斜撑", "水平杆", "水平横杆", "立杆", "连墙件", "支模", 
        "支座", "支护", "支撑", "外架", "模板支架", "杆件", "杆件间距", "盘扣", 
        "扣件", "移动平台", "卸料平台", "操作架", "操作平台"
    ],

    # 起重吊装设备风险
    "Lifting and Hoisting Equipment": [
        "吊运设备", "吊运", "吊车", "吊装", "塔吊", "汽车吊", "吊篮", "重锤", 
        "起重", "钢丝绳", "吊装设备", "支腿", "警戒线", "起重机"
    ],

    # 基坑与土方作业风险
    "Excavation and Earthwork": [
        "基坑", "边坡", "桩机", "桩口", "桩孔", "雨水井", "集水井", "降排水", 
        "支护", "渣土", "土方"
    ],

    # 施工用电安全风险
    "Electrical Safety": [
        "电缆", "电线", "电源线", "配电箱", "漏电保护器", "插座", "变电箱", 
        "二级箱", "接地线", "插头", "插排", "电工", "电动", "油箱", "电瓶车", 
        "充电", "大功率", "电焊", "配电线路", "外电", "电箱"
    ],

    # 机械设备操作风险
    "Machinery Operation Safety": [
        "机械设备", "挖机", "叉车", "吊车", "铲车", "搅拌机", "圆盘锯", "焊机", 
        "桩机", "机具", "切割", "操作人员", "设备检查", "土方设备", "柴油"
    ],

    # 材料堆放与吊运风险
    "Material Storage and Transport": [
        "材料堆放", "堆放", "材料", "吊运设备", "吊运", "吊篮", "垂直", "防护棚", 
        "物料", "钢筋", "模板支架", "卸料平台"
    ],

    # 消防与防火安全风险
    "Fire Safety": [
        "易燃", "易燃物品", "消防", "灭火器", "防火隔离", "吸烟", "烟", "乙炔瓶", 
        "气瓶", "可燃气体", "厨房", "火", "电焊"
    ],

    # 环境污染与现场管理风险
    "Environment and Site Management": [
        # 环境污染
        "扬尘", "灰尘", "垃圾", "泥浆", "排水沟", "裸土", "清理", "清扫", 
        "通风", "灯光", "灯带", "照明", "灯",
        # 现场管理
        "标识", "警戒线", "警告标志", "文明施工", "场清", "旁站", "记录", 
        "施工方案", "报审", "应急预案", "作业许可", "许可", "证","食堂", "渣土", "木凳", "木制", "生活区"
    ],

    # 个人防护风险
    "Personal Protective Equipment": [
        "安全帽", "防护手套", "防护鞋", "防护服", "帽带", "反光衣", "安全绳", 
        "安全带", "生命绳", "生命线"
    ]
}


# 文档分类函数
def classify_document(row, topics):
    """
    根据预定义主题对文档分类：检查文档中是否包含主题关键词。
    如果包含多个主题的关键词，按顺序取第一个匹配的主题。
    """
    for topic, keywords in topics.items():
        for keyword in keywords:
            if keyword in row:
                return topic
    return "Others"  # 如果没有匹配到任何关键词，归为“其他”

# 对每个文档分类
data["主题名称"] = data["content_cutted"].apply(lambda x: classify_document(x, predefined_topics))

# 统计总文档数量
total_documents = len(data)
print(f"\n总文档数量: {total_documents} 篇")

# 统计每个主题的文档数量
print("\n每个主题的文档数量:")
topic_counts = data["主题名称"].value_counts().sort_index()
for topic, count in topic_counts.items():
    print(f"主题 {topic}: {count} 篇")

# 打印分类结果示例
print("\n文档分类结果示例:")
print(data[["文件名", "content_cutted", "主题名称"]].head())

# Step 3 - 保存结果
output_path = "/hdd0/tyt/datasets/predefined_topic_results08.xlsx"
try:
    data.to_excel(output_path, index=False)
    print(f"基于给定主题的分类结果已保存至: {output_path}")
except Exception as e:
    raise RuntimeError(f"Error saving results to '{output_path}': {e}")

