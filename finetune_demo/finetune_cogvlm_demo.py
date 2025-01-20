# import os
# import torch
# import argparse
# from functools import partial
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from sat import mpu, get_args, get_tokenizer
# from sat.training.deepspeed_training import training_main
# from sat.helpers import print_rank0
# from utils.models import FineTuneTrainCogVLMModel
# from utils.utils import llama2_text_processor, llama2_text_processor_inference, get_image_processor

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
# import numpy as np

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
#     # strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[tokenizer.eos_token_id],
#     #                               num_beams=num_beams, consider_end=True)
#     get_func = llama2_text_processor_inference.get_func(None, None, image_rope_mask=kwargs['image_rope_mask'])
#     output = filling_sequence(
#         model, seq,
#         batch_size=1,
#         strategy=strategy,
#         get_masks_and_position_ids=get_func,
#         **kwargs
#     )[0]  # drop memory

#     return output


# def forward_step_eval(data_iterator, model, args, timers):
#     def compute_metrics(eval_preds):
#         preds, labels, device = eval_preds
#         preds = preds.unsqueeze(0)
#         if isinstance(preds, tuple):
#             preds = preds[0]
#         decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#         if args.ignore_pad_token_for_loss:
#             # Replace -100 in the labels as we can't decode them.
#             labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#         decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#         score_dict = {
#             "acc": [],
#             "acc_w/o_case": [],
#         }
#         for pred, label in zip(decoded_preds, decoded_labels):
#             if args.rank == 0:
#                 print('pred', pred, 'label', label, flush=True)
#             if pred == label:
#                 score_dict['acc'].append(1.)
#             else:
#                 score_dict['acc'].append(0.)
#             if pred.lower() == label.lower():
#                 score_dict['acc_w/o_case'].append(1.)
#             else:
#                 score_dict['acc_w/o_case'].append(0.)
            

#         for k, v in score_dict.items():
#             score_dict[k] = float(np.mean(v))
#         return score_dict

#     # Get the batch.
#     timers('batch generator').start()
#     data_b = get_batch(
#         data_iterator, args, timers)
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
#     # print(outputs)
#     model.del_mixin('auto-regressive')
#     return torch.tensor(0, device=outputs.device), {k: torch.tensor(v, device=outputs.device) for k, v in
#                                                     compute_metrics(
#                                                         (outputs.cpu(), labels.cpu(), outputs.device)).items()}


# from torch.nn import CrossEntropyLoss
# # def forward_step(data_iterator, model, args, timers):
# #     """Forward step."""

# #     # Get the batch.
# #     timers('batch generator').start()
# #     data_b = get_batch(
# #         data_iterator, args, timers)
# #     labels = data_b.pop('labels')
# #     timers('batch generator').stop()
    

# #     # # Print input data shapes and content
# #     # print("Input data:")
# #     # for k, v in data_b.items():
# #     #     if isinstance(v, torch.Tensor):
# #     #         print(f"{k}: shape={v.shape}, dtype={v.dtype}")
# #     #     else:
# #     #         print(f"{k}: {v}")

# #     # print("Labels:")
# #     # if isinstance(labels, torch.Tensor):
# #     #     print(f"shape={labels.shape}, dtype={labels.dtype}")
# #     # else:
# #     #     print(labels)
    
# #     logits = model(**data_b)[0]
# #     lm_logits = logits.to(torch.float32)

# #     # print("Logits shape:", lm_logits.shape)
# #     # print("Labels shape:", labels.shape)


# #     # Shift so that tokens < n predict n
# #     shift_labels = labels[..., 1:].contiguous()
# #     shift_logits = lm_logits[..., -1-shift_labels.size(-1):-1, :].contiguous()

# #     # print("Shift logits shape:", shift_logits.shape)
# #     # print("Shift labels shape:", shift_labels.shape)
    

# #     # Flatten the tokens
# #     loss_fct = CrossEntropyLoss(ignore_index=-100)
# #     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
# #     loss = loss.to(torch.float32)

# #     # print("Loss:", loss.item())

# #     return loss, {'loss': loss}


# # List to store loss values
# train_losses = []

# def forward_step(data_iterator, model, args, timers):
#     """Forward step."""

#     # Get the batch.
#     timers('batch generator').start()
#     data_b = get_batch(
#         data_iterator, args, timers)
#     labels = data_b.pop('labels')
#     timers('batch generator').stop()
    
#     logits = model(**data_b)[0]
#     lm_logits = logits.to(torch.float32)

#     # Shift so that tokens < n predict n
#     shift_labels = labels[..., 1:].contiguous()
#     shift_logits = lm_logits[..., -1-shift_labels.size(-1):-1, :].contiguous()
    
#     # Flatten the tokens
#     loss_fct = CrossEntropyLoss(ignore_index=-100)
#     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#     loss = loss.to(torch.float32)

#     # Append the loss value to the list
#     train_losses.append(loss.item())

#     return loss, {'loss': loss}


# from utils.utils import ItemDataset
# def create_dataset_function(image_processor, text_processor, path, args):
#     dataset = ItemDataset(image_processor, text_processor, args, path)
#     return dataset

# from sat.model.finetune.lora2 import LoraMixin
# from sat.model.finetune.prompt_tuning import PTuningV2Mixin

# import matplotlib.pyplot as plt

# def save_and_plot_losses(train_losses, save_path):
#     # Save losses to a file
#     with open(save_path, 'w') as f:
#         for loss in train_losses:
#             f.write(f"{loss}\n")
    
#     # Plot the loss curve
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Training Loss')
#     plt.xlabel('Training Steps')
#     plt.ylabel('Loss')
#     plt.title('Training Loss Curve')
#     plt.legend()
#     plt.savefig(save_path.replace('.txt', '.png'))
#     plt.show()





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
#     from utils.utils import llama2_tokenizer
#     tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
#     image_processor = get_image_processor(args.eva_args["image_size"][0])
#     text_processor = llama2_text_processor(tokenizer, args.max_length, args.image_length)

#     model = training_main(args, model_cls=model, forward_step_function=forward_step, create_dataset_function=partial(create_dataset_function, image_processor, text_processor), collate_fn=data_collator, forward_step_eval=forward_step_eval)
    
#     # Save and plot the losses
#     save_and_plot_losses(train_losses, '/ssd0/tyt/CogVLM/train_losses.txt')

    
    
#     if args.use_lora:
#         model.get_mixin("lora").merge_lora()
#         model.get_mixin("eva").vit_model.get_mixin("lora").merge_lora()
#         args.use_lora = False
#         args.save = "/hdd0/tyt/checkpoints/merged_lora_cogvlm{}".format(args.eva_args["image_size"][0])
#         from sat.training.model_io import save_checkpoint
#         save_checkpoint(1, model, None, None, args)



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
from utils.utils import llama2_text_processor, llama2_text_processor_inference, get_image_processor, llama2_tokenizer
from transformers import AutoTokenizer
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
            if score >= 0.85:
                above_threshold_count += 1

        avg_similarity = float(np.mean(score_dict["avg_similarity"]))
        above_threshold_ratio = above_threshold_count / len(score_dict["avg_similarity"])

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
tokenizer_sim = AutoTokenizer.from_pretrained("/hdd0/tyt/huggingface/text2vec/chinese-macbert-base")
# Initialize similarity model
sim_model = Similarity("/hdd0/tyt/huggingface/text2vec/chinese-macbert-base")

# 训练损失和相似度列表
train_losses = []
similarity_scores = []

def forward_step(data_iterator, model, args, timers, current_step):
    """Forward step."""
    # 获取批次数据
    timers('batch generator').start()
    data_b = get_batch(data_iterator, args, timers)
    labels = data_b.pop('labels')
    timers('batch generator').stop()

    logits = model(**data_b)[0]
    lm_logits = logits.to(torch.float32)

    # Shift so that tokens < n predict n
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = lm_logits[..., -1-shift_labels.size(-1):-1, :].contiguous()

    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.to(torch.float32)

    # Append the loss value to the list if the current step is a multiple of 100
    if current_step % 100 == 0:
        train_losses.append(loss.item())

        # Compute similarity
        preds = shift_logits.argmax(dim=-1)
        
        # 限制范围在 [0, tokenizer_sim.vocab_size) 之间
        shift_labels_int = [[int(x) % tokenizer_sim.vocab_size for x in seq] for seq in shift_labels.cpu().numpy()]
        preds_int = [[int(x) % tokenizer_sim.vocab_size for x in seq] for seq in preds.cpu().numpy()]

        decoded_preds = tokenizer_sim.batch_decode(preds_int, skip_special_tokens=True)
        decoded_labels = tokenizer_sim.batch_decode(shift_labels_int, skip_special_tokens=True)
        
        similarities = [sim_model.get_score(pred, label) for pred, label in zip(decoded_preds, decoded_labels)]
        similarity_scores.append(np.mean(similarities))

    return loss, {'loss': loss}

def create_dataset_function(image_processor, text_processor, path, args):
    dataset = ItemDataset(image_processor, text_processor, args, path)
    return dataset

def save_and_plot_losses(train_losses, similarity_scores, save_path_loss, save_path_sim):
    # Save losses to a file
    with open(save_path_loss, 'w') as f:
        for loss in train_losses:
            f.write(f"{loss}\n")
    
    # Save similarity scores to a file
    with open(save_path_sim, 'w') as f:
        for score in similarity_scores:
            f.write(f"{score}\n")
    
    # Plot the loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(train_losses) * 100, 100), train_losses, label='Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(save_path_loss.replace('.txt', '.png'))
    plt.show()

    # Plot the similarity score curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(similarity_scores) * 100, 100), similarity_scores, label='Similarity Score')
    plt.xlabel('Training Steps')
    plt.ylabel('Similarity Score')
    plt.title('Similarity Score Curve')
    plt.legend()
    plt.savefig(save_path_sim.replace('.txt', '.png'))
    plt.show()

from utils.utils import ItemDataset
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
    
    if args.use_qlora:
        args.device = 'cpu'

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
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    image_processor = get_image_processor(args.eva_args["image_size"][0])
    text_processor = llama2_text_processor(tokenizer, args.max_length, args.image_length)

    current_step = 0
    def forward_step_wrapper(data_iterator, model, args, timers):
        global current_step
        current_step += 1
        return forward_step(data_iterator, model, args, timers, current_step)

    model = training_main(
        args,
        model_cls=model,
        forward_step_function=forward_step_wrapper,
        create_dataset_function=partial(create_dataset_function, image_processor, text_processor),
        collate_fn=data_collator,
        forward_step_eval=partial(forward_step_eval, tokenizer=tokenizer, sim_model=sim_model)
    )

    if args.use_lora:
        model.get_mixin("lora").merge_lora()
        model.get_mixin("eva").vit_model.get_mixin("lora").merge_lora()
        args.use_lora = False
        save_path = "/hdd0/tyt/checkpoints/merged_lora_cogvlm{}".format(args.eva_args["image_size"][0])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        from sat.training.model_io import save_checkpoint
        save_checkpoint(1, model, None, None, args)

    save_and_plot_losses(train_losses, similarity_scores, '/ssd0/tyt/CogVLM/train_losses.txt', '/ssd0/tyt/CogVLM/similarity_scores.txt')
