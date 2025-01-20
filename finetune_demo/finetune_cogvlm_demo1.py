# import os
# import torch
# import argparse
# from functools import partial
# import sys
# import numpy as np
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from sat import mpu, get_args, get_tokenizer
# from sat.training.deepspeed_training import training_main
# from sat.helpers import print_rank0
# from utils.models import FineTuneTrainCogVLMModel
# from utils.utils import llama2_text_processor, llama2_text_processor_inference, get_image_processor, llama2_tokenizer  # 确保导入llama2_tokenizer
# from transformers import AutoTokenizer, AutoModel  # 添加对AutoTokenizer和AutoModel的导入
# from text2vec import Similarity


# def disable_untrainable_params(self):
#     total_trainable = 0
#     enable = [('mlp', 'vit')]
#     if self.args.use_ptuning:
#         enable.extend(['ptuning'])
#     if self.args.use_lora or self.args.use_qlora:
#         enable.extend(['matrix_A', 'matrix_B'])
#     for n, p in self.named_parameters():
#         flag = False
#         for e in enable:
#             if type(e) is tuple:
#                 if e[0].lower() in n.lower() and e[1].lower() in n.lower() and 55 > int(n[:n.find('.mlp')].split('.')[-1]) > 45:
#                     flag = True
#                     break
#             else:
#                 if e.lower() in n.lower():
#                     flag = True
#                     break
#         if not flag:
#             p.requires_grad_(False)
#         else:
#             total_trainable += p.numel()
#             print_rank0(n)
#     print_rank0("***** Total trainable parameters: "+str(total_trainable)+" *****")

# FineTuneTrainCogVLMModel.disable_untrainable_params = disable_untrainable_params

# def data_collator(examples):
#     examples = [ex for ex in examples if len(ex) > 0] # drop {}
#     for example in examples:
#         for k in example:
#             if isinstance(example[k], list):
#                 example[k] = torch.tensor(example[k])
#             elif isinstance(example[k], np.ndarray):
#                 example[k] = torch.from_numpy(example[k])
#     img_args = {}
#     tmp_example = examples[0]
#     for k in tmp_example['vision']:
#         if type(tmp_example['vision'][k]) is torch.Tensor:
#             img_args['vision_'+k] = torch.cat([example['vision'][k] for example in examples])
#         else:
#             img_args['vision_'+k] = example['vision'][k]
#     for example in examples:
#         example.pop('vision')
#         if 'cross' in example:
#             example.pop('cross')

#     model_args = {}
#     tmp_example = examples[0]
#     for k in tmp_example:
#         if type(tmp_example[k]) is torch.Tensor:
#             model_args[k] = torch.cat([example[k] for example in examples])
#         else:
#             model_args[k] = tmp_example[k]
#     model_args.update(img_args)

#     #print("Collated Labels:", model_args['labels'])  # 在汇总标签后打印标签
#     return model_args

# from collections import defaultdict

# def broadcast_auto(data_dict):
#     type2list = defaultdict(list)
#     other = []
#     for k in data_dict:
#         if type(data_dict[k]) is torch.Tensor:
#             type2list[data_dict[k].dtype].append(k)
#         else:
#             other.append(k)
#     new_data = {}
#     for k in type2list:
#         new_data.update(mpu.broadcast_data(type2list[k], data_dict, k))
#     for k in other:
#         new_data[k] = data_dict[k]
#     return new_data

# def get_batch(data_iterator, args, timers):
#     # Broadcast data.
#     timers('data loader').start()
#     if data_iterator is not None:
#         data = next(data_iterator)
#     else:
#         data = None
#     timers('data loader').stop()
#     data_b = broadcast_auto(data)
#     for k in data_b:
#         if type(data_b[k]) is torch.Tensor and data_b[k].dtype is not torch.int32 and data_b[k].dtype is not torch.long:
#             if args.fp16:
#                 data_b[k] = data_b[k].half()
#             elif args.bf16:
#                 data_b[k] = data_b[k].bfloat16()
#     return data_b

# from torch.nn import CrossEntropyLoss
# from sat.model.mixins import CachedAutoregressiveMixin
# from sat.generation.autoregressive_sampling import filling_sequence
# from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy

# def chat(model, tokenizer, tokens,
#          max_length: int = 1800, num_beams=5, top_p=0.95, top_k=0, temperature=0.8, **kwargs):
#     inputs = tokens.to(model.parameters().__next__().device)[0]
#     seq = torch.cat(
#         [inputs, torch.tensor([-1] * (max_length - len(inputs)), device=inputs.device)], dim=0
#     )
#     strategy = BaseStrategy(temperature=temperature, top_p=0.4, top_k=1, end_tokens=[tokenizer.eos_token_id])
#     get_func = llama2_text_processor_inference.get_func(None, None, image_rope_mask=kwargs['image_rope_mask'])
#     output = filling_sequence(
#         model, seq,
#         batch_size=1,
#         strategy=strategy,
#         get_masks_and_position_ids=get_func,
#         **kwargs
#     )[0]  # drop memory

#     return output

# # 修改 forward_step_eval 函数，增加 sim_model 参数
# def forward_step_eval(data_iterator, model, args, timers, tokenizer, sim_model):
#     def compute_metrics(eval_preds):
#         preds, labels, device = eval_preds
#         preds = preds.unsqueeze(0)
#         if isinstance(preds, tuple):
#             preds = preds[0]
#         decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#         if args.ignore_pad_token_for_loss:
#             labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#         decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#         # 初始化评价指标的字典
#         score_dict = {
#             "avg_similarity": [],
#             "above_threshold_ratio": []
#         }

#         above_threshold_count = 0

#         for pred, label in zip(decoded_preds, decoded_labels):
#             if args.rank == 0:
#                 print('pred', pred, 'label', label, flush=True)
#             score = sim_model.get_score(pred, label)
#             score_dict["avg_similarity"].append(score)
#             if score >= 0.8:
#                 above_threshold_count += 1

#         avg_similarity = float(np.mean(score_dict["avg_similarity"]))
#         above_threshold_ratio = above_threshold_count / len(score_dict["avg_similarity"])

#         # 更新评价指标字典
#         score_dict["avg_similarity"] = avg_similarity
#         score_dict["above_threshold_ratio"] = above_threshold_ratio

#         return score_dict

#     timers('batch generator').start()
#     data_b = get_batch(data_iterator, args, timers)
#     timers('batch generator').stop()

#     context_len = int(data_b['context_length'][0])
#     tokens = data_b['input_ids'][:, :context_len]
#     data_b['vision_expert_mask'] = data_b['vision_expert_mask'][:, :context_len]
#     data_b['image_embed_mask'] = data_b['image_embed_mask'][:, :context_len]
#     data_b['image_rope_mask'] = data_b['image_rope_mask'][:, :context_len]

#     data_b.pop('input_ids')
#     data_b.pop('attention_mask')
#     data_b.pop('position_ids')
#     labels = data_b.pop('labels')
#     qid = data_b.pop('question_id')

#     model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
#     outputs = chat(model, tokenizer, tokens, **data_b)[0][context_len:]
#     model.del_mixin('auto-regressive')

#     metrics = compute_metrics((outputs.cpu(), labels.cpu(), outputs.device))

#     return torch.tensor(0, device=outputs.device), {k: torch.tensor(v, device=outputs.device) for k, v in metrics.items()}


# # 下载并缓存模型
# tokenizer = AutoTokenizer.from_pretrained("/hdd0/tyt/huggingface/text2vec/chinese-macbert-base")
# # Initialize similarity model
# sim_model = Similarity("/hdd0/tyt/huggingface/text2vec/chinese-macbert-base")


# def forward_step(data_iterator, model, args, timers, tokenizer):
#     """Forward step."""
#     # 获取批次数据
#     timers('batch generator').start()
#     data_b = get_batch(data_iterator, args, timers)
#     labels = data_b.pop('labels')
#     timers('batch generator').stop()

#     # 将 -100 替换为 tokenizer.pad_token_id
#     processed_labels = torch.where(labels == -100, tokenizer.pad_token_id, labels)

#     # 打印处理后的标签以便调试
#     # print("Processed Labels Shape:", processed_labels.shape)
#     # print("Processed Labels:", processed_labels)

#     # 模型前向传播
#     logits = model(**data_b)[0]
#     lm_logits = logits.to(torch.float32)

#     # 打印 logits 的形状和一些统计信息
#     print("Logits Shape:", logits.shape)
#     print("Logits Sample:", logits[0, :10])  # 打印前10个预测值

#     # 获取预测和标签文本
#     decoded_preds = tokenizer.batch_decode(logits.argmax(dim=-1), skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(processed_labels.tolist(), skip_special_tokens=True)

#     # 打印解码后的标签和预测以便调试
#     # print("Decoded Predictions:", decoded_preds)
#     print("Decoded Labels:", decoded_labels)

#     # 计算语义相似度
#     sim_scores = []
#     for pred, label in zip(decoded_preds, decoded_labels):
#         score = sim_model.get_score(pred, label)
#         sim_scores.append(score)
#         # 打印每个预测和标签的相似度
#         #print(f"Prediction: {pred}, Label: {label}, Similarity Score: {score}")

#     # 将相似度转换为损失（可以取1 - 相似度作为损失）
#     sim_scores = torch.tensor(sim_scores, device=lm_logits.device, dtype=torch.float32)
#     loss = 1 - sim_scores.mean()

#     # 确保损失具有梯度
#     loss.requires_grad_(True)

#     # 打印损失以便调试
#     print("Sim Scores:", sim_scores)
#     print("Loss:", loss)

#     return loss, {'loss': loss}


# from utils.utils import ItemDataset
# def create_dataset_function(image_processor, text_processor, path, args):
#     dataset = ItemDataset(image_processor, text_processor, args, path)
#     return dataset

# from sat.model.finetune.lora2 import LoraMixin
# from sat.model.finetune.prompt_tuning import PTuningV2Mixin

# if __name__ == '__main__':
#     py_parser = argparse.ArgumentParser(add_help=False)
#     py_parser.add_argument('--max_length', type=int)
#     py_parser.add_argument('--ignore_pad_token_for_loss', action='store_false')
#     py_parser.add_argument("--version", type=str, default="chat_old", help='version to interact with')
#     py_parser.add_argument("--from_pretrained", type=str, default="/hdd0/tyt/SAT/cogvlm-base-490", help='pretrained ckpt')
#     py_parser.add_argument("--local_tokenizer", type=str, default="/hdd1/huggingface/models/lmsys/vicuna-7b-v1.5", help='tokenizer path')
#     py_parser.add_argument("--vit_checkpoint_activations", action='store_true')
#     py_parser = FineTuneTrainCogVLMModel.add_model_specific_args(py_parser)
#     known, args_list = py_parser.parse_known_args()
#     args = get_args(args_list)
#     args = argparse.Namespace(**vars(args), **vars(known))
#     if args.use_qlora:
#         args.device = 'cpu'

#     model, args = FineTuneTrainCogVLMModel.from_pretrained(args.from_pretrained, args, overwrite_args={'model_parallel_size': args.model_parallel_size} if args.model_parallel_size != 1 else {})
#     if args.use_ptuning:
#         model.add_mixin("ptuning", PTuningV2Mixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.pre_seq_len))
#     if args.use_lora:
#         model.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, layer_range=args.layer_range), reinit=True)
#         model.get_mixin("eva").vit_model.add_mixin("lora", LoraMixin(args.eva_args['num_layers'], args.lora_rank, layer_range=args.layer_range), reinit=True)
#     elif args.use_qlora:
#         model.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, layer_range=args.layer_range, qlora=True), reinit=True)
        
#     if args.use_qlora and torch.cuda.is_available():
#         model = model.to('cuda')
    
#     tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)  # 确保导入并初始化tokenizer
#     image_processor = get_image_processor(args.eva_args["image_size"][0])
#     text_processor = llama2_text_processor(tokenizer, args.max_length, args.image_length)

#     model = training_main(
#         args,
#         model_cls=model,
#         forward_step_function=partial(forward_step, tokenizer=tokenizer),  # 确保传递tokenizer
#         create_dataset_function=partial(create_dataset_function, image_processor, text_processor),
#         collate_fn=data_collator,
#         forward_step_eval=partial(forward_step_eval, tokenizer=tokenizer, sim_model=sim_model)  # 使用partial传递tokenizer和sim_model
#     )
#     if args.use_lora:
#         model.get_mixin("lora").merge_lora()
#         model.get_mixin("eva").vit_model.get_mixin("lora").merge_lora()
#         args.use_lora = False
#         args.save = "/hdd0/tyt/checkpoints/merged_lora_cogvlm{}".format(args.eva_args["image_size"][0])
#         from sat.training.model_io import save_checkpoint
#         save_checkpoint(1, model, None, None, args)


# import os
# import torch
# import argparse
# from functools import partial
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from sat import mpu, get_args, get_tokenizer
# from sat.training.deepspeed_training import training_main
# from sat.helpers import print_rank0
# from utils.models import FineTuneTrainCogVLMModel
# from utils.utils import llama2_text_processor, llama2_text_processor_inference, get_image_processor, llama2_tokenizer  # 确保导入llama2_tokenizer
# from transformers import AutoTokenizer, AutoModel  # 添加对AutoTokenizer和AutoModel的导入
# from text2vec import Similarity


# def disable_untrainable_params(self):
#     total_trainable = 0
#     enable = [('mlp', 'vit')]
#     if self.args.use_ptuning:
#         enable.extend(['ptuning'])
#     if self.args.use_lora or self.args.use_qlora:
#         enable.extend(['matrix_A', 'matrix_B'])
#     for n, p in self.named_parameters():
#         flag = False
#         for e in enable:
#             if type(e) is tuple:
#                 if e[0].lower() in n.lower() and e[1].lower() in n.lower() and 55 > int(n[:n.find('.mlp')].split('.')[-1]) > 45:
#                     flag = True
#                     break
#             else:
#                 if e.lower() in n.lower():
#                     flag = True
#                     break
#         if not flag:
#             p.requires_grad_(False)
#         else:
#             total_trainable += p.numel()
#             print_rank0(n)
#     print_rank0("***** Total trainable parameters: "+str(total_trainable)+" *****")

# FineTuneTrainCogVLMModel.disable_untrainable_params = disable_untrainable_params

# def data_collator(examples):
#     examples = [ex for ex in examples if len(ex) > 0] # drop {}
#     for example in examples:
#         for k in example:
#             if isinstance(example[k], list):
#                 example[k] = torch.tensor(example[k])
#             elif isinstance(example[k], np.ndarray):
#                 example[k] = torch.from_numpy(example[k])
#     img_args = {}
#     tmp_example = examples[0]
#     for k in tmp_example['vision']:
#         if type(tmp_example['vision'][k]) is torch.Tensor:
#             img_args['vision_'+k] = torch.cat([example['vision'][k] for example in examples])
#         else:
#             img_args['vision_'+k] = example['vision'][k]
#     for example in examples:
#         example.pop('vision')
#         if 'cross' in example:
#             example.pop('cross')

#     model_args = {}
#     tmp_example = examples[0]
#     for k in tmp_example:
#         if type(tmp_example[k]) is torch.Tensor:
#             model_args[k] = torch.cat([example[k] for example in examples])
#         else:
#             model_args[k] = tmp_example[k]
#     model_args.update(img_args)

#     #print("Collated Labels:", model_args['labels'])  # 在汇总标签后打印标签
#     return model_args

# from collections import defaultdict

# def broadcast_auto(data_dict):
#     type2list = defaultdict(list)
#     other = []
#     for k in data_dict:
#         if type(data_dict[k]) is torch.Tensor:
#             type2list[data_dict[k].dtype].append(k)
#         else:
#             other.append(k)
#     new_data = {}
#     for k in type2list:
#         new_data.update(mpu.broadcast_data(type2list[k], data_dict, k))
#     for k in other:
#         new_data[k] = data_dict[k]
#     return new_data

# def get_batch(data_iterator, args, timers):
#     # Broadcast data.
#     timers('data loader').start()
#     if data_iterator is not None:
#         data = next(data_iterator)
#     else:
#         data = None
#     timers('data loader').stop()
#     data_b = broadcast_auto(data)
#     for k in data_b:
#         if type(data_b[k]) is torch.Tensor and data_b[k].dtype is not torch.int32 and data_b[k].dtype is not torch.long:
#             if args.fp16:
#                 data_b[k] = data_b[k].half()
#             elif args.bf16:
#                 data_b[k] = data_b[k].bfloat16()
#     return data_b

# from torch.nn import CrossEntropyLoss
# from sat.model.mixins import CachedAutoregressiveMixin
# from sat.generation.autoregressive_sampling import filling_sequence
# from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy

# def chat(model, tokenizer, tokens,
#          max_length: int = 1800, num_beams=5, top_p=0.95, top_k=0, temperature=0.8, **kwargs):
#     inputs = tokens.to(model.parameters().__next__().device)[0]
#     seq = torch.cat(
#         [inputs, torch.tensor([-1] * (max_length - len(inputs)), device=inputs.device)], dim=0
#     )
#     strategy = BaseStrategy(temperature=temperature, top_p=0.4, top_k=1, end_tokens=[tokenizer.eos_token_id])
#     get_func = llama2_text_processor_inference.get_func(None, None, image_rope_mask=kwargs['image_rope_mask'])
#     output = filling_sequence(
#         model, seq,
#         batch_size=1,
#         strategy=strategy,
#         get_masks_and_position_ids=get_func,
#         **kwargs
#     )[0]  # drop memory

#     return output

# # 修改 forward_step_eval 函数，增加 sim_model 参数
# def forward_step_eval(data_iterator, model, args, timers, tokenizer, sim_model):
#     def compute_metrics(eval_preds):
#         preds, labels, device = eval_preds
#         preds = preds.unsqueeze(0)
#         if isinstance(preds, tuple):
#             preds = preds[0]
#         decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#         if args.ignore_pad_token_for_loss:
#             labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#         decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#         # 初始化评价指标的字典
#         score_dict = {
#             "avg_similarity": [],
#             "above_threshold_ratio": []
#         }

#         above_threshold_count = 0

#         for pred, label in zip(decoded_preds, decoded_labels):
#             if args.rank == 0:
#                 print('pred', pred, 'label', label, flush=True)
#             score = sim_model.get_score(pred, label)
#             score_dict["avg_similarity"].append(score)
#             if score >= 0.85:
#                 above_threshold_count += 1

#         avg_similarity = float(np.mean(score_dict["avg_similarity"]))
#         above_threshold_ratio = above_threshold_count / len(score_dict["avg_similarity"])

#         # 更新评价指标字典
#         score_dict["avg_similarity"] = avg_similarity
#         score_dict["above_threshold_ratio"] = above_threshold_ratio

#         return score_dict

#     timers('batch generator').start()
#     data_b = get_batch(data_iterator, args, timers)
#     timers('batch generator').stop()

#     context_len = int(data_b['context_length'][0])
#     tokens = data_b['input_ids'][:, :context_len]
#     data_b['vision_expert_mask'] = data_b['vision_expert_mask'][:, :context_len]
#     data_b['image_embed_mask'] = data_b['image_embed_mask'][:, :context_len]
#     data_b['image_rope_mask'] = data_b['image_rope_mask'][:, :context_len]

#     data_b.pop('input_ids')
#     data_b.pop('attention_mask')
#     data_b.pop('position_ids')
#     labels = data_b.pop('labels')
#     qid = data_b.pop('question_id')

#     model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
#     outputs = chat(model, tokenizer, tokens, **data_b)[0][context_len:]
#     model.del_mixin('auto-regressive')

#     metrics = compute_metrics((outputs.cpu(), labels.cpu(), outputs.device))

#     return torch.tensor(0, device=outputs.device), {k: torch.tensor(v, device=outputs.device) for k, v in metrics.items()}


# # 下载并缓存模型
# tokenizer = AutoTokenizer.from_pretrained("/hdd0/tyt/huggingface/text2vec/chinese-macbert-base")
# # Initialize similarity model
# sim_model = Similarity("/hdd0/tyt/huggingface/text2vec/chinese-macbert-base")


# # def forward_step(data_iterator, model, args, timers, tokenizer, train_loss_list):
# #     """Forward step."""
# #     # 获取批次数据
# #     timers('batch generator').start()
# #     data_b = get_batch(data_iterator, args, timers)
# #     labels = data_b.pop('labels')
# #     timers('batch generator').stop()

# #     # 将 -100 替换为 tokenizer.pad_token_id
# #     processed_labels = torch.where(labels == -100, tokenizer.pad_token_id, labels)

# #     # 打印处理后的标签以便调试
# #     # print("Processed Labels Shape:", processed_labels.shape)
# #     # print("Processed Labels:", processed_labels)

# #     # 模型前向传播
# #     logits = model(**data_b)[0]
# #     lm_logits = logits.to(torch.float32)

# #     # 打印 logits 的形状和一些统计信息
# #     print("Logits Shape:", logits.shape)
# #     print("Logits Sample:", logits[0, :10])  # 打印前10个预测值

# #     # 获取预测和标签文本
# #     decoded_preds = tokenizer.batch_decode(logits.argmax(dim=-1), skip_special_tokens=True)
# #     decoded_labels = tokenizer.batch_decode(processed_labels.tolist(), skip_special_tokens=True)

# #     # 打印解码后的标签和预测以便调试
# #     # print("Decoded Predictions:", decoded_preds)
# #     print("Decoded Labels:", decoded_labels)

# #     # 计算语义相似度
# #     sim_scores = []
# #     for pred, label in zip(decoded_preds, decoded_labels):
# #         score = sim_model.get_score(pred, label)
# #         sim_scores.append(score)
# #         # 打印每个预测和标签的相似度
# #         #print(f"Prediction: {pred}, Label: {label}, Similarity Score: {score}")

# #     # 将相似度转换为损失（可以取1 - 相似度作为损失）
# #     sim_scores = torch.tensor(sim_scores, device=lm_logits.device, dtype=torch.float32)
# #     loss = 1 - sim_scores.mean()

# #     # 确保损失具有梯度
# #     loss.requires_grad_(True)

# #     # 记录训练损失
# #     train_loss_list.append(loss.item())

# #     # 打印损失以便调试
# #     print("Sim Scores:", sim_scores)
# #     print("Loss:", loss)

# #     return loss, {'loss': loss}

# def forward_step(data_iterator, model, args, timers, tokenizer, train_loss_list):
#     """Forward step."""
#     # 获取批次数据
#     timers('batch generator').start()
#     data_b = get_batch(data_iterator, args, timers)
#     labels = data_b.pop('labels')
#     timers('batch generator').stop()

#     # 将 -100 替换为 tokenizer.pad_token_id
#     processed_labels = torch.where(labels == -100, tokenizer.pad_token_id, labels)

#     # 模型前向传播
#     logits = model(**data_b)[0]
#     lm_logits = logits.to(torch.float32)

#     # 获取预测和标签文本
#     decoded_preds = tokenizer.batch_decode(logits.argmax(dim=-1), skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(processed_labels.tolist(), skip_special_tokens=True)

#     # 计算语义相似度
#     sim_scores = []
#     for pred, label in zip(decoded_preds, decoded_labels):
#         score = sim_model.get_score(pred, label)
#         sim_scores.append(score)

#     # 将相似度转换为损失（可以取1 - 相似度作为损失）
#     sim_scores = torch.tensor(sim_scores, device=lm_logits.device, dtype=torch.float32)
#     loss = 1 - sim_scores.mean()

#     # 添加正则化项 (L2正则化)
#     l2_lambda = 0.01
#     l2_reg = torch.tensor(0., device=lm_logits.device)
#     for param in model.parameters():
#         l2_reg += torch.norm(param, 2)
    
#     total_loss = loss + l2_lambda * l2_reg

#     # 归一化损失
#     normalized_loss = total_loss / (1 + l2_lambda)

#     # 确保损失具有梯度
#     normalized_loss.requires_grad_(True)

#     # 仅在主进程中记录训练损失
#     if torch.distributed.get_rank() == 0:
#         train_loss_list.append(normalized_loss.item())

#     # 打印调试信息
#     print("Sim Scores:", sim_scores)
#     print("Loss:", loss)
#     print("L2 Regularization:", l2_reg)
#     print("Total Loss:", total_loss)
#     print("Normalized Loss:", normalized_loss)

#     return normalized_loss, {'loss': normalized_loss}






# from utils.utils import ItemDataset
# def create_dataset_function(image_processor, text_processor, path, args):
#     dataset = ItemDataset(image_processor, text_processor, args, path)
#     return dataset

# from sat.model.finetune.lora2 import LoraMixin
# from sat.model.finetune.prompt_tuning import PTuningV2Mixin

# if __name__ == '__main__':
#     py_parser = argparse.ArgumentParser(add_help=False)
#     py_parser.add_argument('--max_length', type=int)
#     py_parser.add_argument('--ignore_pad_token_for_loss', action='store_false')
#     py_parser.add_argument("--version", type=str, default="chat_old", help='version to interact with')
#     py_parser.add_argument("--from_pretrained", type=str, default="/hdd0/tyt/SAT/cogvlm-base-490", help='pretrained ckpt')
#     py_parser.add_argument("--local_tokenizer", type=str, default="/hdd1/huggingface/models/lmsys/vicuna-7b-v1.5", help='tokenizer path')
#     py_parser.add_argument("--vit_checkpoint_activations", action='store_true')
#     py_parser = FineTuneTrainCogVLMModel.add_model_specific_args(py_parser)
#     known, args_list = py_parser.parse_known_args()
#     args = get_args(args_list)
#     args = argparse.Namespace(**vars(args), **vars(known))
    

#     # 从预训练模型加载
#     model, args = FineTuneTrainCogVLMModel.from_pretrained(
#         args.from_pretrained, 
#         args, 
#         overwrite_args={'model_parallel_size': args.model_parallel_size} if args.model_parallel_size != 1 else {}
#     )

#     # 添加模型的微调Mixin
#     if args.use_ptuning:
#         model.add_mixin("ptuning", PTuningV2Mixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.pre_seq_len))
#     if args.use_lora:
#         model.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, layer_range=args.layer_range), reinit=True)
#         model.get_mixin("eva").vit_model.add_mixin("lora", LoraMixin(args.eva_args['num_layers'], args.lora_rank, layer_range=args.layer_range), reinit=True)
#     elif args.use_qlora:
#         model.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, layer_range=args.layer_range, qlora=True), reinit=True)
    
#     # 初始化tokenizer和数据处理器
#     tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)  # 确保导入并初始化tokenizer
#     image_processor = get_image_processor(args.eva_args["image_size"][0])
#     text_processor = llama2_text_processor(tokenizer, args.max_length, args.image_length)

#     # 初始化训练损失列表
#     train_loss_list = []

#     # 训练模型
#     # 训练模型
#     model = training_main(
#         args,
#         model_cls=model,
#         forward_step_function=partial(forward_step, tokenizer=tokenizer, train_loss_list=train_loss_list),  # 确保传递tokenizer和train_loss_list
#         create_dataset_function=partial(create_dataset_function, image_processor, text_processor),
#         collate_fn=data_collator,
#         forward_step_eval=partial(forward_step_eval, tokenizer=tokenizer, sim_model=sim_model)  # 使用partial传递tokenizer和sim_model
#     )


#     # 如果使用LoRA，则合并LoRA权重
#     if args.use_lora:
#         model.get_mixin("lora").merge_lora()
#         model.get_mixin("eva").vit_model.get_mixin("lora").merge_lora()
#         args.use_lora = False
#         save_path = "/hdd0/tyt/checkpoints/merged_lora_cogvlm{}".format(args.eva_args["image_size"][0])
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保路径存在
#         from sat.training.model_io import save_checkpoint
#         save_checkpoint(1, model, None, None, args)

#  # 绘制并保存训练损失曲线
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_loss_list, label='Training Loss')
#     plt.xlabel('Iterations')
#     plt.ylabel('Loss')
#     plt.title('Training Loss Curve')
#     plt.legend()
#     plt.savefig('/ssd0/tyt/CogVLM/training_loss_curve.png')  # 保存损失曲线图像
#     plt.show()



import os
import torch
import argparse
from functools import partial
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sat import mpu, get_args, get_tokenizer
from sat.training.deepspeed_training import training_main
from sat.helpers import print_rank0
from utils.models import FineTuneTrainCogVLMModel
from utils.utils import llama2_text_processor, llama2_text_processor_inference, get_image_processor, llama2_tokenizer  # 确保导入llama2_tokenizer
from transformers import AutoTokenizer, AutoModel  # 添加对AutoTokenizer和AutoModel的导入
from text2vec import Similarity


def disable_untrainable_params(self):
    total_trainable = 0
    enable = [('mlp', 'vit')]
    if self.args.use_ptuning:
        enable.extend(['ptuning'])
    if self.args.use_lora or self.args.use_qlora:
        enable.extend(['matrix_A', 'matrix_B'])
    for n, p in self.named_parameters():
        flag = False
        for e in enable:
            if type(e) is tuple:
                if e[0].lower() in n.lower() and e[1].lower() in n.lower() and 55 > int(n[:n.find('.mlp')].split('.')[-1]) > 45:
                    flag = True
                    break
            else:
                if e.lower() in n.lower():
                    flag = True
                    break
        if not flag:
            p.requires_grad_(False)
        else:
            total_trainable += p.numel()
            print_rank0(n)
    print_rank0("***** Total trainable parameters: "+str(total_trainable)+" *****")

FineTuneTrainCogVLMModel.disable_untrainable_params = disable_untrainable_params

def data_collator(examples):
    examples = [ex for ex in examples if len(ex) > 0] # drop {}
    for example in examples:
        for k in example:
            if isinstance(example[k], list):
                example[k] = torch.tensor(example[k])
            elif isinstance(example[k], np.ndarray):
                example[k] = torch.from_numpy(example[k])
    img_args = {}
    tmp_example = examples[0]
    for k in tmp_example['vision']:
        if type(tmp_example['vision'][k]) is torch.Tensor:
            img_args['vision_'+k] = torch.cat([example['vision'][k] for example in examples])
        else:
            img_args['vision_'+k] = example['vision'][k]
    for example in examples:
        example.pop('vision')
        if 'cross' in example:
            example.pop('cross')

    model_args = {}
    tmp_example = examples[0]
    for k in tmp_example:
        if type(tmp_example[k]) is torch.Tensor:
            model_args[k] = torch.cat([example[k] for example in examples])
        else:
            model_args[k] = tmp_example[k]
    model_args.update(img_args)

    #print("Collated Labels:", model_args['labels'])  # 在汇总标签后打印标签
    return model_args

from collections import defaultdict

def broadcast_auto(data_dict):
    type2list = defaultdict(list)
    other = []
    for k in data_dict:
        if type(data_dict[k]) is torch.Tensor:
            type2list[data_dict[k].dtype].append(k)
        else:
            other.append(k)
    new_data = {}
    for k in type2list:
        new_data.update(mpu.broadcast_data(type2list[k], data_dict, k))
    for k in other:
        new_data[k] = data_dict[k]
    return new_data

def get_batch(data_iterator, args, timers):
    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = broadcast_auto(data)
    for k in data_b:
        if type(data_b[k]) is torch.Tensor and data_b[k].dtype is not torch.int32 and data_b[k].dtype is not torch.long:
            if args.fp16:
                data_b[k] = data_b[k].half()
            elif args.bf16:
                data_b[k] = data_b[k].bfloat16()
    return data_b

from torch.nn import CrossEntropyLoss
from sat.model.mixins import CachedAutoregressiveMixin
from sat.generation.autoregressive_sampling import filling_sequence
from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy

def chat(model, tokenizer, tokens,
         max_length: int = 1800, num_beams=5, top_p=0.95, top_k=0, temperature=0.8, **kwargs):
    inputs = tokens.to(model.parameters().__next__().device)[0]
    seq = torch.cat(
        [inputs, torch.tensor([-1] * (max_length - len(inputs)), device=inputs.device)], dim=0
    )
    strategy = BaseStrategy(temperature=temperature, top_p=0.4, top_k=1, end_tokens=[tokenizer.eos_token_id])
    get_func = llama2_text_processor_inference.get_func(None, None, image_rope_mask=kwargs['image_rope_mask'])
    output = filling_sequence(
        model, seq,
        batch_size=1,
        strategy=strategy,
        get_masks_and_position_ids=get_func,
        **kwargs
    )[0]  # drop memory

    return output

# 修改 forward_step_eval 函数，增加 sim_model 参数
def forward_step_eval(data_iterator, model, args, timers, tokenizer, sim_model):
    def compute_metrics(eval_preds):
        preds, labels, device = eval_preds
        preds = preds.unsqueeze(0)
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if args.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 初始化评价指标的字典
        score_dict = {
            "avg_similarity": [],
            "above_threshold_ratio": []
        }

        above_threshold_count = 0

        for pred, label in zip(decoded_preds, decoded_labels):
            if args.rank == 0:
                print('pred', pred, 'label', label, flush=True)
            score = sim_model.get_score(pred, label)
            score_dict["avg_similarity"].append(score)
            if score >= 0.8:
                above_threshold_count += 1

        avg_similarity = float(np.mean(score_dict["avg_similarity"]))
        above_threshold_ratio = above_threshold_count / len(score_dict["avg_similarity"])

        # 更新评价指标字典
        score_dict["avg_similarity"] = avg_similarity
        score_dict["above_threshold_ratio"] = above_threshold_ratio

        return score_dict

    timers('batch generator').start()
    data_b = get_batch(data_iterator, args, timers)
    timers('batch generator').stop()

    context_len = int(data_b['context_length'][0])
    tokens = data_b['input_ids'][:, :context_len]
    data_b['vision_expert_mask'] = data_b['vision_expert_mask'][:, :context_len]
    data_b['image_embed_mask'] = data_b['image_embed_mask'][:, :context_len]
    data_b['image_rope_mask'] = data_b['image_rope_mask'][:, :context_len]

    data_b.pop('input_ids')
    data_b.pop('attention_mask')
    data_b.pop('position_ids')
    labels = data_b.pop('labels')
    qid = data_b.pop('question_id')

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    outputs = chat(model, tokenizer, tokens, **data_b)[0][context_len:]
    model.del_mixin('auto-regressive')

    metrics = compute_metrics((outputs.cpu(), labels.cpu(), outputs.device))

    return torch.tensor(0, device=outputs.device), {k: torch.tensor(v, device=outputs.device) for k, v in metrics.items()}


# 下载并缓存模型
tokenizer = AutoTokenizer.from_pretrained("/hdd0/tyt/huggingface/text2vec/chinese-macbert-base")
# Initialize similarity model
sim_model = Similarity("/hdd0/tyt/huggingface/text2vec/chinese-macbert-base")


recorded_batches = set()

# def forward_step(data_iterator, model, args, timers, tokenizer, train_loss_list, sim_scores_list, loss_file, sim_file):
#     """Forward step."""
#     # 获取批次数据
#     timers('batch generator').start()
#     data_b = get_batch(data_iterator, args, timers)
#     labels = data_b.pop('labels')
#     timers('batch generator').stop()

#     # 将 -100 替换为 tokenizer.pad_token_id
#     processed_labels = torch.where(labels == -100, tokenizer.pad_token_id, labels)

#      # 模型前向传播
#     logits = model(**data_b)[0]
#     lm_logits = logits.to(torch.float32)

#     # 获取预测和标签文本
#     decoded_preds = tokenizer.batch_decode(logits.argmax(dim=-1), skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(processed_labels.tolist(), skip_special_tokens=True)

#     # 计算语义相似度
#     sim_scores = []
#     for pred, label in zip(decoded_preds, decoded_labels):
#         score = sim_model.get_score(pred, label)
#         sim_scores.append(score)

#     # 记录语义相似度值
#     avg_sim_score = torch.mean(torch.tensor(sim_scores)).item()
    
#     # 获取当前批次编号（可以根据你自己的逻辑确定 batch_idx）
#     batch_idx = args.batch_idx if hasattr(args, 'batch_idx') else 0
#     args.batch_idx = batch_idx + 1
    
#     # 将相似度转换为损失（可以取1 - 相似度作为损失）
#     sim_scores = torch.tensor(sim_scores, device=lm_logits.device, dtype=torch.float32)
#     loss = 1 - sim_scores.mean()

#     # 添加正则化项 (L2正则化)
#     l2_lambda = 0.01
#     l2_reg = torch.tensor(0., device=lm_logits.device)
#     for param in model.parameters():
#         l2_reg += torch.norm(param, 2)
    
#     total_loss = loss + l2_lambda * l2_reg

#     # 归一化损失
#     normalized_loss = total_loss / (1 + l2_lambda)

#     # 确保损失具有梯度
#     normalized_loss.requires_grad_(True)

#     # 仅在主进程中记录训练损失
#     if torch.distributed.get_rank() == 0:
#         if batch_idx not in recorded_batches:
#             recorded_batches.add(batch_idx)
#             train_loss_list.append(normalized_loss.item())
#             sim_scores_list.append(avg_sim_score)
#         loss_file.write(f"{normalized_loss.item()}\n")  # 将损失写入文件

#     # 打印调试信息
#     print("Sim Scores:", sim_scores)
#     print("Loss:", loss)
#     print("L2 Regularization:", l2_reg)
#     print("Total Loss:", total_loss)
#     print("Normalized Loss:", normalized_loss)

#     return normalized_loss, {'loss': normalized_loss}

def forward_step(data_iterator, model, args, timers, tokenizer, train_loss_list, sim_scores_list, loss_file, sim_file):
    """Forward step."""
    # 获取批次数据
    timers('batch generator').start()
    data_b = get_batch(data_iterator, args, timers)
    labels = data_b.pop('labels')
    timers('batch generator').stop()

    # 将 -100 替换为 tokenizer.pad_token_id
    processed_labels = torch.where(labels == -100, tokenizer.pad_token_id, labels)

    # 模型前向传播
    logits = model(**data_b)[0]
    lm_logits = logits.to(torch.float32)

    # 获取预测和标签文本
    decoded_preds = tokenizer.batch_decode(logits.argmax(dim=-1), skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(processed_labels.tolist(), skip_special_tokens=True)

    # 计算语义相似度
    sim_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        score = sim_model.get_score(pred, label)
        sim_scores.append(score)

    # 记录语义相似度值
    avg_sim_score = torch.mean(torch.tensor(sim_scores)).item()
    
    # 获取当前批次编号（可以根据你自己的逻辑确定 batch_idx）
    batch_idx = args.batch_idx if hasattr(args, 'batch_idx') else 0
    args.batch_idx = batch_idx + 1
    
    # 将相似度转换为损失（可以取1 - 相似度作为损失）
    sim_scores = torch.tensor(sim_scores, device=lm_logits.device, dtype=torch.float32)
    loss = 1 - sim_scores.mean()

    # 添加正则化项 (L2正则化)
    l2_lambda = 0.01
    l2_reg = torch.tensor(0., device=lm_logits.device)
    for param in model.parameters():
        l2_reg += torch.norm(param, 2)
    
    total_loss = loss + l2_lambda * l2_reg

    # 归一化损失
    normalized_loss = total_loss / (1 + l2_lambda)

    # 确保损失具有梯度
    normalized_loss.requires_grad_(True)

    # 仅在主进程中记录训练损失和语义相似度
    if torch.distributed.get_rank() == 0:
        if batch_idx not in recorded_batches:
            recorded_batches.add(batch_idx)
            train_loss_list.append(normalized_loss.item())
            sim_scores_list.append(avg_sim_score)
            loss_file.write(f"{batch_idx},{normalized_loss.item()}\n")  # 将损失写入文件，加入批次编号
            sim_file.write(f"{batch_idx},{avg_sim_score}\n")  # 将语义相似度写入文件，加入批次编号

    # 打印调试信息
    print("Sim Scores:", sim_scores)
    print("Loss:", loss)
    print("L2 Regularization:", l2_reg)
    print("Total Loss:", total_loss)
    print("Normalized Loss:", normalized_loss)

    return normalized_loss, {'loss': normalized_loss}


from utils.utils import ItemDataset
def create_dataset_function(image_processor, text_processor, path, args):
    dataset = ItemDataset(image_processor, text_processor, args, path)
    return dataset

from sat.model.finetune.lora2 import LoraMixin
from sat.model.finetune.prompt_tuning import PTuningV2Mixin

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--max_length', type=int)
    py_parser.add_argument('--ignore_pad_token_for_loss', action='store_false')
    py_parser.add_argument("--version", type=str, default="chat_old", help='version to interact with')
    py_parser.add_argument("--from_pretrained", type=str, default="/hdd0/tyt/SAT/cogvlm-base-490", help='pretrained ckpt')
    py_parser.add_argument("--local_tokenizer", type=str, default="/hdd1/huggingface/models/lmsys/vicuna-7b-v1.5", help='tokenizer path')
    py_parser.add_argument("--vit_checkpoint_activations", action='store_true')
    py_parser = FineTuneTrainCogVLMModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    
    # 从预训练模型加载
    model, args = FineTuneTrainCogVLMModel.from_pretrained(
        args.from_pretrained, 
        args, 
        overwrite_args={'model_parallel_size': args.model_parallel_size} if args.model_parallel_size != 1 else {}
    )

    # 添加模型的微调Mixin
    if args.use_ptuning:
        model.add_mixin("ptuning", PTuningV2Mixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.pre_seq_len))
    if args.use_lora:
        model.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, layer_range=args.layer_range), reinit=True)
        model.get_mixin("eva").vit_model.add_mixin("lora", LoraMixin(args.eva_args['num_layers'], args.lora_rank, layer_range=args.layer_range), reinit=True)
    elif args.use_qlora:
        model.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, layer_range=args.layer_range, qlora=True), reinit=True)
    
    # 初始化tokenizer和数据处理器
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)  # 确保导入并初始化tokenizer
    image_processor = get_image_processor(args.eva_args["image_size"][0])
    text_processor = llama2_text_processor(tokenizer, args.max_length, args.image_length)

    # 初始化训练损失列表和语义相似度列表
    train_loss_list = []
    sim_scores_list = []

    # 打开文件用于记录损失和语义相似度
    loss_file = open('/ssd0/tyt/CogVLM/training_loss_values.txt', 'w')
    sim_file = open('/ssd0/tyt/CogVLM/semantic_similarity_values.txt', 'w')

    # 训练模型
    model = training_main(
        args,
        model_cls=model,
        forward_step_function=partial(forward_step, tokenizer=tokenizer, train_loss_list=train_loss_list, sim_scores_list=sim_scores_list, loss_file=loss_file, sim_file=sim_file),
        create_dataset_function=partial(create_dataset_function, image_processor, text_processor),
        collate_fn=data_collator,
        forward_step_eval=partial(forward_step_eval, tokenizer=tokenizer, sim_model=sim_model)
    )

    # 关闭文件
    loss_file.close()
    sim_file.close()

    # 如果使用LoRA，则合并LoRA权重
    if args.use_lora:
        model.get_mixin("lora").merge_lora()
        model.get_mixin("eva").vit_model.get_mixin("lora").merge_lora()
        args.use_lora = False
        save_path = "/hdd0/tyt/checkpoints/merged_lora_cogvlm{}".format(args.eva_args["image_size"][0])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        from sat.training.model_io import save_checkpoint
        save_checkpoint(1, model, None, None, args)

    # 绘制并保存训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_list, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig('/ssd0/tyt/CogVLM/training_loss_curve.png')
    plt.show()

    # 绘制并保存语义相似度变化曲线
    plt.figure(figsize=(10, 5))
    plt.plot(sim_scores_list, label='Semantic Similarity')
    plt.xlabel('Iterations')
    plt.ylabel('Similarity Score')
    plt.title('Semantic Similarity Curve')
    plt.legend()
    plt.savefig('/ssd0/tyt/CogVLM/semantic_similarity_curve.png')
    plt.show()
