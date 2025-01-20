# import os
# import re
# import pandas as pd

# def save_image_filenames_to_excel(image_folder, excel_save_path):
#     # 获取图片文件的文件名
#     file_names = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    
#     # 按名称排序
#     file_names.sort()

#     # 过滤文件名
#     def filter_filename(name):
#         return re.sub(r'^\d+[^\w]*|[^\w]*[a-zA-Z]*$', '', name)

#     filtered_file_names = [filter_filename(f) for f in file_names]
    
#     # 生成编号和文件名列表
#     data = [(str(i+1).zfill(5), file_name) for i, file_name in enumerate(filtered_file_names)]

#     # 创建DataFrame
#     df = pd.DataFrame(data, columns=["编号", "文件名"])

#     # 保存为Excel文件
#     excel_file_path = os.path.join(excel_save_path, "image_filenames.xlsx")
#     df.to_excel(excel_file_path, index=False, engine='openpyxl')
#     print(f"Excel文件已保存到: {excel_file_path}")

# # 定义图片文件夹和保存路径
# image_folder = "/hdd0/tyt/datasets/2023-12"
# excel_save_path = "/hdd0/tyt/datasets/"

# save_image_filenames_to_excel(image_folder, excel_save_path)


import re
import pandas as pd

# 过滤Excel文件第二列文件名并保存到新文件
def filter_excel_file_column(input_file_path, output_file_path):
    # 读取Excel文件
    df = pd.read_excel(input_file_path, engine='openpyxl')

    # 遍历并过滤第二列的文件名
    def filter_name(name):
        # 删除开头的字母、数字、标点符号和汉字"层"
        return re.sub(r'^[\W\dA-Za-z一二三四五六七八九十]+', '', name)

    df["文件名"] = df["文件名"].apply(lambda x: filter_name(str(x)))

    # 保存修改后的结果到新文件
    df.to_excel(output_file_path, index=False, engine='openpyxl')
    print(f"已更新文件名并保存至: {output_file_path}")

# 定义输入和输出Excel文件路径
input_excel_file = "/hdd0/tyt/datasets/filtered_image_filenames2.xlsx"
output_excel_file = "/hdd0/tyt/datasets/filtered_image_filenames3.xlsx"

filter_excel_file_column(input_excel_file, output_excel_file)



