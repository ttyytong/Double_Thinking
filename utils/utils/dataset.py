# import os
# import logging
# import random
# import logging
# import jsonlines
# from io import BytesIO
# from PIL import Image
# from torch.utils.data import Dataset
# from sat.helpers import print_rank0

# def find_all_files(path, suffix=".jpg"):
#     target_files = []
#     for cur_dir, _, files in os.walk(path, followlinks=True):
#         for f in files:
#             if f.endswith(suffix):
#                 target_files.append(os.path.join(cur_dir, f))
#     print_rank0(f'find {len(target_files)} files...')
#     return target_files

# class ItemDataset(Dataset):
#     def __init__(self, image_processor, text_processor, args, data_dirs, cross_image_processor=None, **kwargs):
#         super().__init__()
#         self.data = self.load_data(data_dirs)
#         self.image_processor, self.text_processor, self.cross_image_processor = image_processor, text_processor, cross_image_processor
    
#     def process_img(self, img):
#         img_dict = {'vision': self.image_processor(img)}
#         if self.cross_image_processor:
#             img_dict.update({'cross': self.cross_image_processor(img)})
#         return img_dict
    
#     def process_text(self, answer, prompt):
#         return self.text_processor(answer, prompt)
    
#     def load_data(self, data_dir):
#         all_files = find_all_files(data_dir, suffix=".jpg")
#         print_rank0(f"find {len(all_files)} samples in all...")
#         return all_files
    
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         data = self.data[index]
#         # img
#         try:
#             img = Image.open(data).convert('RGB')
#         except Exception as e:
#             print_rank0(e, level=logging.WARNING)
#             return {}
#         img_dict = self.process_img(img)
#         # text
#         label = data.split('/')[-1].split('.')[0]
#         uni_key = label
#         text_dict = self.process_text(label, "CAPTCHA:")
#         if text_dict is None:
#             print_rank0(f"Process text failed. Please check the max_target_length & max_source_length.\n The data is {data}", level=logging.WARNING)
#             return {}
#         # other attr
#         ret = {**img_dict, **text_dict, "question_id": uni_key}
#         return ret
    

# import os
# import logging
# from PIL import Image
# from torch.utils.data import Dataset
# from sat.helpers import print_rank0

# class ItemDataset(Dataset):
#     def __init__(self, image_processor, text_processor, args, data_dir, **kwargs):
#         super().__init__()
#         self.image_processor = image_processor
#         self.text_processor = text_processor
#         self.data_dir = data_dir

#         # Load all image and caption paths
#         self.data = self.load_data()

#     def process_img(self, img):
#         img_dict = {'vision': self.image_processor(img)}
#         return img_dict

#     def process_text(self, caption_text, prompt):
#         return self.text_processor(caption_text, prompt)

#     def load_data(self):
#         image_files = []
#         for filename in os.listdir(self.data_dir):
#             if filename.endswith(".jpg"):
#                 image_path = os.path.join(self.data_dir, filename)
#                 caption_path = os.path.join(self.data_dir, f"{filename.split('.')[0]}.txt")
#                 if os.path.exists(caption_path):
#                     image_files.append((image_path, caption_path))
#                 else:
#                     print_rank0(f"Caption file not found for {filename}", level=logging.WARNING)
#         print_rank0(f"Found {len(image_files)} image-caption pairs in {self.data_dir}")
#         return image_files

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         image_path, caption_path = self.data[index]

#         # Load image
#         try:
#             img = Image.open(image_path).convert('RGB')
#         except Exception as e:
#             print_rank0(f"Error loading image {image_path}: {e}", level=logging.WARNING)
#             return {}

#         img_dict = self.process_img(img)

#         # Load caption
#         try:
#             with open(caption_path, 'r', encoding='utf-8') as f:
#                 caption_text = f.read().strip()
#                 print_rank0(f"Loaded caption: {caption_text}")
#         except Exception as e:
#             print_rank0(f"Error loading caption {caption_path}: {e}", level=logging.WARNING)
#             return {}

#         text_dict = self.process_text(caption_text, "CAPTION:")

#         if text_dict is None:
#             print_rank0(f"Process text failed for caption {caption_path}. Please check the processing logic.", level=logging.WARNING)
#             return {}

#         # Construct return dictionary
#         ret = {**img_dict, **text_dict, "question_id": os.path.basename(image_path).split('.')[0]}
#         return ret

# import os
# import logging
# import random
# import logging
# import jsonlines
# from io import BytesIO
# from PIL import Image
# from torch.utils.data import Dataset
# from sat.helpers import print_rank0

# class ItemDataset(Dataset):
#     def __init__(self, image_processor, text_processor, args, data_dir, **kwargs):
#         super().__init__()
#         self.image_processor = image_processor
#         self.text_processor = text_processor
#         self.data_dir = data_dir

#         # 加载所有图像和对应的文本路径
#         self.data = self.load_data()

#     def process_img(self, img):
#         img_dict = {'vision': self.image_processor(img)}
#         return img_dict

#     def process_text(self, caption_text, prompt):
#         return self.text_processor(caption_text, prompt)

#     def load_data(self):
#         image_files = []
#         for filename in os.listdir(self.data_dir):
#             if filename.endswith(".jpg"):
#                 # 提取图像文件的基本名称（不含扩展名）
#                 base_name = os.path.splitext(filename)[0]
#                 image_path = os.path.join(self.data_dir, filename)
#                 caption_path = os.path.join(self.data_dir, f"{base_name}.txt")
                
#                 # 检查对应的文本文件是否存在
#                 if os.path.exists(caption_path):
#                     image_files.append((image_path, caption_path))
#                 else:
#                     print_rank0(f"Caption file not found for {filename}", level=logging.WARNING)
        
#         print_rank0(f"Found {len(image_files)} image-caption pairs in {self.data_dir}")
#         return image_files

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         image_path, caption_path = self.data[index]

#         # 加载图像
#         try:
#             img = Image.open(image_path).convert('RGB')
#         except Exception as e:
#             print_rank0(f"Error loading image {image_path}: {e}", level=logging.WARNING)
#             return {}

#         img_dict = self.process_img(img)

#         # 加载文本
#         try:
#             with open(caption_path, 'r', encoding='utf-8') as f:
#                 caption_text = f.read().strip()
#                 print_rank0(f"Loaded caption: {caption_text}")
#         except Exception as e:
#             print_rank0(f"Error loading caption {caption_path}: {e}", level=logging.WARNING)
#             return {}

#         text_dict = self.process_text(caption_text, "CAPTION:")

#         if text_dict is None:
#             print_rank0(f"Process text failed for caption {caption_path}. Please check the processing logic.", level=logging.WARNING)
#             return {}

#         # 构造返回字典
#         ret = {**img_dict, **text_dict, "question_id": os.path.basename(image_path).split('.')[0]}
        
#         # # 打印返回的字典内容
#         # print_rank0(f"Data dict: {ret}")
        
#         return ret


import os
import logging
from PIL import Image
from torch.utils.data import Dataset
from sat.helpers import print_rank0

class ItemDataset(Dataset):
    def __init__(self, image_processor, text_processor, args, data_dir, **kwargs):
        super().__init__()
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.data_dir = data_dir

        # 加载所有图像和对应的文本路径
        self.data = self.load_data()

    def process_img(self, img):
        img_dict = {'vision': self.image_processor(img)}
        #print_rank0(f"Processed Image Dict: {img_dict}")
        return img_dict

    def process_text(self, caption_text, prompt):
        text_dict = self.text_processor(caption_text, prompt)
        #print_rank0(f"Processed Text Dict: {text_dict}")
        return text_dict

    def load_data(self):
        image_files = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".jpg"):
                base_name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.data_dir, filename)
                caption_path = os.path.join(self.data_dir, f"{base_name}.txt")
                if os.path.exists(caption_path):
                    image_files.append((image_path, caption_path))
                else:
                    print_rank0(f"Caption file not found for {filename}", level=logging.WARNING)
        
        print_rank0(f"Found {len(image_files)} image-caption pairs in {self.data_dir}")
        return image_files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, caption_path = self.data[index]
        try:
            img = Image.open(image_path).convert('RGB')
            print_rank0(f"Loaded image: {image_path}")
        except Exception as e:
            print_rank0(f"Error loading image {image_path}: {e}", level=logging.WARNING)
            return {}

        img_dict = self.process_img(img)

        try:
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption_text = f.read().strip()
                print_rank0(f"Loaded caption: {caption_text}")
        except Exception as e:
            print_rank0(f"Error loading caption {caption_path}: {e}", level=logging.WARNING)
            return {}

        text_dict = self.process_text(caption_text, "CAPTION:")

        if text_dict is None:
            print_rank0(f"Process text failed for caption {caption_path}. Please check the processing logic.", level=logging.WARNING)
            return {}

        ret = {**img_dict, **text_dict, "question_id": os.path.basename(image_path).split('.')[0]}
        #print_rank0(f"Data dict: {ret}")
        
        return ret
