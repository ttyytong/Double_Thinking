# import os
# import shutil

# def find_all_files(path, suffix=".jpg"):
#     target_files = []
#     for cur_dir, _, files in os.walk(path, followlinks=True):
#         for f in files:
#             if f.endswith(suffix):
#                 target_files.append(os.path.join(cur_dir, f))
#     print(f'find {len(target_files)} files...')
#     return target_files

# all_files = find_all_files('/ssd0/tyt/CogVLM/datasets/construction_images')
# os.makedirs("/ssd0/tyt/CogVLM/datasets/construction_images_split", exist_ok=True)
# os.makedirs("/ssd0/tyt/CogVLM/datasets/construction_images_split/train", exist_ok=True)
# os.makedirs("/ssd0/tyt/CogVLM/datasets/construction_images_split/valid", exist_ok=True)
# os.makedirs("/ssd0/tyt/CogVLM/datasets/construction_images_split/test", exist_ok=True)

# import random
# random.seed(2023)
# random.shuffle(all_files)
# train = all_files[:8000]
# valid = all_files[8000:8000+500]
# test = all_files[8000+500:8000+500+1500]

# print("building train")
# for file in train:
#     shutil.move(file, os.path.join("/ssd0/tyt/CogVLM/datasets/construction_images_split/train", file.split("/")[-1]))
# print("building valid")
# for file in valid:
#     shutil.move(file, os.path.join("/ssd0/tyt/CogVLM/datasets/construction_images_split/valid", file.split("/")[-1]))
# print("building test")
# for file in test:
#     shutil.move(file, os.path.join("/ssd0/tyt/CogVLM/datasets/construction_images_split/test", file.split("/")[-1]))
# print("done")



# import os
# import shutil
# import random

# def find_all_files(path, suffix=".jpg"):
#     target_files = []
#     for cur_dir, _, files in os.walk(path, followlinks=True):
#         for f in files:
#             if f.endswith(suffix):
#                 target_files.append(os.path.join(cur_dir, f))
#     print(f'find {len(target_files)} files...')
#     return target_files

# # 查找所有文件
# all_files = find_all_files('/hdd0/tyt/datasets/construction/test1')

# # 创建目标文件夹
# os.makedirs("/hdd0/tyt/datasets/construction/test1_split", exist_ok=True)
# os.makedirs("/hdd0/tyt/datasets/construction/test1_split/train", exist_ok=True)
# os.makedirs("/hdd0/tyt/datasets/construction/test1_split/valid", exist_ok=True)
# os.makedirs("/hdd0/tyt/datasets/construction/test1_split/test", exist_ok=True)

# # 随机打乱数据并按照比例划分
# random.seed(2023)
# random.shuffle(all_files)

# # 计算每个子集的大小
# total_size = len(all_files)
# train_size = int(0.7 * total_size)
# valid_size = int(0.1 * total_size)
# test_size = total_size - train_size - valid_size  # 确保总数不变

# # 划分数据集
# train = all_files[:train_size]
# valid = all_files[train_size:train_size + valid_size]
# test = all_files[train_size + valid_size:]

# # 输出划分结果
# print(f"训练集大小: {len(train)}")
# print(f"验证集大小: {len(valid)}")
# print(f"测试集大小: {len(test)}")

# # 移动文件到相应的文件夹
# print("building train")
# for file in train:
#     shutil.move(file, os.path.join("/ssd0/tyt/CogVLM/datasets/construction_images_split/train", os.path.basename(file)))

# print("building valid")
# for file in valid:
#     shutil.move(file, os.path.join("/ssd0/tyt/CogVLM/datasets/construction_images_split/valid", os.path.basename(file)))

# print("building test")
# for file in test:
#     shutil.move(file, os.path.join("/ssd0/tyt/CogVLM/datasets/construction_images_split/test", os.path.basename(file)))

# print("done")


import os
import shutil
import random

def find_all_pairs(path):
    jpg_files = [f for f in os.listdir(path) if f.endswith('.jpg')]
    pairs = [(os.path.join(path, f), os.path.join(path, f.replace('.jpg', '.txt'))) for f in jpg_files]
    return pairs

# 查找所有文件对
all_pairs = find_all_pairs('/hdd0/tyt/datasets/construction/test')

# 创建目标文件夹
os.makedirs("/hdd0/tyt/datasets/construction/test_split/train", exist_ok=True)
os.makedirs("/hdd0/tyt/datasets/construction/test_split/valid", exist_ok=True)
os.makedirs("/hdd0/tyt/datasets/construction/test_split/test", exist_ok=True)

# 随机打乱数据并按照比例划分
random.seed(2023)
random.shuffle(all_pairs)

# 计算每个子集的大小
total_size = len(all_pairs)
train_size = int(0.7 * total_size)
valid_size = int(0.1 * total_size)
test_size = total_size - train_size - valid_size  # 确保总数不变

# 划分数据集
train = all_pairs[:train_size]
valid = all_pairs[train_size:train_size + valid_size]
test = all_pairs[train_size + valid_size:]

# 输出划分结果
print(f"训练集大小: {len(train)}")
print(f"验证集大小: {len(valid)}")
print(f"测试集大小: {len(test)}")

# 移动文件到相应的文件夹
def move_files(pairs, subset_name):
    for jpg_file, txt_file in pairs:
        shutil.move(jpg_file, os.path.join(f"/hdd0/tyt/datasets/construction/test_split/{subset_name}", os.path.basename(jpg_file)))
        shutil.move(txt_file, os.path.join(f"/hdd0/tyt/datasets/construction/test_split/{subset_name}", os.path.basename(txt_file)))

print("building train")
move_files(train, 'train')

print("building valid")
move_files(valid, 'valid')

print("building test")
move_files(test, 'test')

print("done")
