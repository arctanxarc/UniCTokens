import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset, load_from_disk
# from transformers import Qwen2VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import argparse
from typing import Union
from pdata import get_personalized_mmu_dataloader, get_personalized_t2i_dataloader, get_concept_info, get_concept_all_training_images_path, resize_img
from lightning.pytorch.utilities import CombinedLoader
from insightface.app import FaceAnalysis
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.nn import Parameter
from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, create_attention_mask_for_mmu
from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter
from transformers import AutoTokenizer
from llava.llava import conversation as conversation_lib
import copy
from omegaconf import DictConfig, ListConfig, OmegaConf
from grpo import unic_grpo
conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
if not dist.is_initialized():
    dist.init_process_group(backend='nccl', init_method='env://')
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
# os.environ['http_proxy'] = 'http://127.0.0.1:2333'
# os.environ['https_proxy'] = 'http://127.0.0.1:2333'
if hasattr(torch, 'compile'):
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.verbose = False
# add image_generation_prompt in the GRPOConfig
@dataclass
class GrpoConfig(GRPOConfig):
    """
    Configuration class for the GRPO training script.
    """
    cfg_weight: float = field(default=3.0, metadata={"help": "The cfg weight for image generation"})
    img_size: int = field(default=512, metadata={"help": "The size of the image to generate"})
    patch_size: int = field(default=16, metadata={"help": "The patch size of the image to generate"})
    deepspeed: bool = field(default=False)
    
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    semantic:str=field(default=True)
    llm:str=field(default='glm')
    accelerate:str=field(default=True)
    data_root:str=field(
        default="path/to/unictokens_data"
    )
    ablation_mode:str=field(default='random',metadata={"help":"Chosen from: ur,ru,r,u,random"})
    inverse_prompt:bool=field(default=True)
    save_dir:str=field(default="./tmp_result_accelerate/")
    work_dir:str=field(default="./")
    batch_num:int=field(default=10)
    batch_size:int=field(default=1)
    num_gen: int = field(default=9, metadata={"help": "The number of new generations of image to generate"})
    num_gpus: int=field(default=3,metadata={"help":"The number of gpus"})
    image_size: int = field(default=512, metadata={"help": "The size of the image to generate"})
    reward_funcs: list[str] = field(
        default_factory=lambda: ["test"],
        metadata={"help": "List of reward functions. Possible values: 'test'"},
    )
    config_file: str = field(
        default="configs/showo_demo_512x512.yaml",
        metadata={"help": "Path to the configuration file"}
    )
    concept: str = field(
        default="bo",
        metadata={"help": "Concept for the model"}
    )
    pre_trained_ckpt_name: str = field(
        default="path/to/second_stage_checkpoint_dir",
        metadata={"help": "Name of the pre-trained checkpoint"}
    )
    device: str = field(
        default=torch.device(f"cuda:{local_rank}"),
        metadata={"help": "Device to use for training"}
    )
    save_training_image: bool = field(
        default=False,
        metadata={"help": "Save training images"}
    )
    interval_epochs: int = field(
        default=2,
        metadata={"help": "Number of epochs between evaluations"}
    )
    epoch: int = field(
        default=5,
        metadata={"help": "Total number of training epochs"}
    )
    epoch_to_load: int = field(
        default=15,
        metadata={"help": "Epoch to load from the checkpoint"}
    )
    lr: float = field(
        default=1e-6,
        metadata={"help": "Learning rate"}
    )
    nums_new_token_i_stage_1: int = field(
        default=16,
        metadata={"help": "Number of new tokens in stage 1"}
    )
    nums_new_token_i_stage_2: int = field(
        default=8,
        metadata={"help": "Number of new tokens in stage 2"}
    )

def make_detection_prompt(nouns):
    if len(nouns) == 0:
        return '', []
    
    token_spans = []
    pointer = 0
    for noun in nouns:
        n_split = noun.strip().split(" ")
        if len(n_split) == 1:
            length = len(n_split[0])
            token_spans.append([[pointer, pointer + length]])
            pointer += length + 3 # on the blank space after the noun
        else: # multiple words
            beg_len = len(n_split[0])
            total_length = len(noun)
            end_len = len(n_split[-1])
            token_spans.append([[pointer, pointer + beg_len], [pointer + total_length - end_len, pointer + total_length]])
            pointer += total_length + 3 # on the blank space after the noun
    text_prompt = ' . '.join(nouns) + "." # need to end with '.
    return text_prompt, token_spans


reward_funcs_registry = {
    'test':'test'
}

def setup_model(args, config):
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side ="left")
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)
    vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(args.device)
    model = Showo.from_pretrained(config.model.showo.pretrained_model_path,low_cpu_mem_usage=False).to(args.device)
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            output_device=local_rank
        )
    return tokenizer, uni_prompting, vq_model, model.module


def face_embed_init(args, concept, nums_new_token_i_stage_2):
    training_images = get_concept_all_training_images_path(concept) # ["str", "str", ...]
    training_images = [Image.open(img).convert("RGB") for img in training_images]
    training_images = [resize_img(img) for img in training_images]
    
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    face_embs = []
    for img in training_images:
        face_info = app.get(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
        face_emb = face_info['embedding']   # np array, shape: (512,)
        face_emb = torch.from_numpy(face_emb) # shape: (512)
        face_embs.append(face_emb)
    face_embs_mean = torch.stack(face_embs).mean(dim=0) # torch.Size([512])
    
    # 512 / nums_new_token_i_stage_2 = n, face_embs_multi = 512 = [n, 512/nums_new_token_i_stage_2]
    face_embs_multi = face_embs_mean.view(nums_new_token_i_stage_2, -1) # shape: [8, 64]
    face_embs_multi = torch.nn.functional.interpolate(face_embs_multi.unsqueeze(0), size=(2048,), mode='linear').squeeze(0) # shape: [8, 2048]
    
    assert face_embs_multi.shape == (nums_new_token_i_stage_2, 2048)
    return face_embs_multi
def update_tokens_load_from_pretrained(args,
                                       concept, 
                                       tokenizer, 
                                       model, 
                                       pre_trained_ckpt_name, 
                                       epoch_to_load, 
                                       nums_new_token_i_stage_1=16,
                                       nums_new_token_i_stage_2=8,
                                       need_init=True,
                                       second_time=False):
    ckpt_path = os.path.join("../", concept, pre_trained_ckpt_name)
    ckpt_embed_path = os.path.join(ckpt_path, f"epoch_{epoch_to_load}_embed.pt")
    ckpt_lm_head_weight_path = os.path.join(ckpt_path, f"epoch_{epoch_to_load}_lm_head_weight.pt")
    ckpt_lm_head_bias_path = os.path.join(ckpt_path, f"epoch_{epoch_to_load}_lm_head_bias.pt")

    nums_total_token_i = nums_new_token_i_stage_1 + nums_new_token_i_stage_2
    adj_tokens = [f"<token_{i}>" for i in range(nums_total_token_i)]
    sks_token = [f"<{concept}>"]
    new_tokens = sks_token + adj_tokens
    num_new_tokens = len(new_tokens)  # 16 + 8 + 1 

    # 文本 token 数量（ID 0-50304）
    if second_time:
        original_text_vocab_size = len(tokenizer)-num_new_tokens
    else:
        original_text_vocab_size = len(tokenizer)
    # Image token 数量（原 ID 50305-58497）
    original_image_vocab_size = model.showo.get_input_embeddings().num_embeddings - len(tokenizer)
    original_total_vocab = original_text_vocab_size + original_image_vocab_size  # 58498
    
    # 新的参数
    new_text_vocab_size = original_text_vocab_size + num_new_tokens  # 50305 + 25
    new_total_vocab = original_total_vocab + num_new_tokens          # 58498 + 25

    # ------------------------------
    # Step 1: 修改 Tokenizer 的词汇表
    # ------------------------------

    # 添加新 token 到 50305-50321 的位置
    if not second_time:
        num_new_tokens = tokenizer.add_tokens(new_tokens)
    new_token_ids = tokenizer.convert_tokens_to_ids(new_tokens)
    print("新 token ID:", new_token_ids)  # 应输出 50305-50329
    sks_token_id = tokenizer.convert_tokens_to_ids(sks_token)
    print("sks_token_id:", sks_token_id)  # 应输出 50305
    
    # ------------------------------
    # Step 2: 调整模型的权重
    # ------------------------------
    embed_dim = model.showo.get_input_embeddings().weight.shape[1] 
    with torch.no_grad():
        # 获取嵌入层权重
        embeddings = model.showo.get_input_embeddings().weight.data
        
        # 扩展嵌入层（58498 -> 58522）
        model.showo.resize_token_embeddings(new_total_vocab)
        # new_embeddings = model.showo.get_input_embeddings().weight.data

        # 将原 Image Token 权重后移 17 位
        original_image_weights = embeddings[original_text_vocab_size:original_total_vocab].clone()
        model.showo.get_input_embeddings().weight.data[new_text_vocab_size:new_total_vocab] = original_image_weights
        print(original_text_vocab_size,original_total_vocab,new_text_vocab_size,new_total_vocab)
        if os.path.exists(ckpt_embed_path):
            ckpt_embed_weight = torch.load(ckpt_embed_path)
            model.showo.get_input_embeddings().weight.data[original_text_vocab_size:original_text_vocab_size + 1 + nums_total_token_i] = ckpt_embed_weight.to(model.showo.get_input_embeddings().weight.device)
        elif need_init and get_concept_info(concept)[2] == "human" and nums_new_token_i_stage_2 > 0:
            model.showo.get_input_embeddings().weight.data[original_text_vocab_size + 1 + nums_new_token_i_stage_1:new_text_vocab_size] = face_embed_init(args, concept, nums_new_token_i_stage_2).to(model.showo.get_input_embeddings().weight.device)
        else:
            #
            #
            #
            # raise ValueError("Embedding weights do not exist!")
            new_embed_weights = Parameter(torch.randn(num_new_tokens, embed_dim, 
                                           dtype=model.showo.get_input_embeddings().weight.dtype,
                                           device=model.showo.get_input_embeddings().weight.device))
            # 保持与原模型相同的初始化范围（通常为均匀分布或正态分布）
            nn.init.normal_(new_embed_weights, mean=0.0, std=0.02)
            model.showo.get_input_embeddings().weight.data[original_text_vocab_size:new_text_vocab_size] = new_embed_weights
        

        # 处理 lm_head（假设与嵌入层共享权重）
        if model.showo.lm_head.weight.data.shape[0] == new_total_vocab:
            # 扩展 lm_head 权重
            lm_head = model.showo.lm_head
            new_lm_head = torch.nn.Linear(
                lm_head.in_features, 
                new_total_vocab, 
                bias=hasattr(lm_head, 'bias')
            )
            new_lm_head.weight.data = lm_head.weight.data.clone()
            new_lm_head.weight.data[new_text_vocab_size:new_total_vocab] = lm_head.weight.data[original_text_vocab_size:original_total_vocab]

            if os.path.exists(ckpt_lm_head_weight_path):
                ckpt_lm_head_weight = torch.load(ckpt_lm_head_weight_path)
                new_lm_head.weight.data[original_text_vocab_size:original_text_vocab_size + 1 + nums_total_token_i] = ckpt_lm_head_weight.to(new_lm_head.weight.device)
            else:
                # raise ValueError("lm_head weights do not exist!")
                # 
                # 
                # 
                print(f"警告: {ckpt_lm_head_weight_path} 不存在，使用随机初始化")
                # 只随机初始化新增部分，而不是整个weight
                new_lm_head.weight.data[original_text_vocab_size:new_text_vocab_size] = torch.randn(
                    new_text_vocab_size - original_text_vocab_size, 
                    embed_dim,
                    dtype=model.showo.lm_head.weight.dtype,
                    device=model.showo.lm_head.weight.device
                )
                nn.init.normal_(new_lm_head.weight.data[original_text_vocab_size:new_text_vocab_size], 
                                mean=0.0, std=0.02)
                # model.showo.lm_head.weight.data[original_text_vocab_size:new_text_vocab_size] = new_lm_head.weight
        

            if hasattr(lm_head, 'bias'):
                new_lm_head.bias.data = lm_head.bias.data.clone()
                new_lm_head.bias.data[new_text_vocab_size:new_total_vocab] = lm_head.bias.data[original_text_vocab_size:original_total_vocab]
                
                if os.path.exists(ckpt_lm_head_bias_path):
                    ckpt_lm_head_bias = torch.load(ckpt_lm_head_bias_path)
                    new_lm_head.bias.data[original_text_vocab_size:original_text_vocab_size + 1 + nums_total_token_i] = ckpt_lm_head_bias.to(new_lm_head.weight.device)
                else:
                    # raise ValueError("lm_head bias do not exist!")   
                    # 
                    # 
                    # 
                    # 
                    print(f"警告: {ckpt_lm_head_bias_path} 不存在，使用随机初始化")
                    # 只初始化新增部分的偏置
                    new_lm_head.bias.data[original_text_vocab_size:new_text_vocab_size] = torch.zeros(
                        new_text_vocab_size - original_text_vocab_size,
                        dtype=model.showo.lm_head.bias.dtype,
                        device=model.showo.lm_head.bias.device
                    )
       
                         
            model.showo.lm_head = new_lm_head
        else:
            raise ValueError("lm_head weights do not match the input embeddings!")

    index_no_updates = torch.ones((new_total_vocab,), dtype=torch.bool)
    index_no_updates[new_token_ids] = False
    adj_token_ids = tokenizer.convert_tokens_to_ids(adj_tokens) # shape: [16]
    
    
    # ------------------------------
    # 验证
    # ------------------------------
    # 检查新 token 的 ID
    print("新增文本 token ID:", [tokenizer.convert_tokens_to_ids(t) for t in new_tokens])  # 应输出 50305-50321

    # 检查一个原 Image Token 的新 ID
    sample_image_token = tokenizer.convert_ids_to_tokens(original_text_vocab_size)  # 原 ID 50305
    print(sample_image_token)
    print(f"Concept Token '{sample_image_token}' 的新 ID:", tokenizer.convert_tokens_to_ids(sample_image_token))  # 应输出 50322

    # 检查嵌入层形状
    print("嵌入层大小:", model.showo.get_input_embeddings().weight.shape)  # 应显示 torch.Size([58515, 2048])

    # 检查 index_no_updates 中 True 的位置和数量，True 应该是 new token ids
    print("index_no_updates 中 False 的位置:", torch.nonzero(~index_no_updates).squeeze())  # 应输出 50305-50321
    print("index_no_updates 中 True 的数量:", torch.sum(index_no_updates))  # 应输出 58498

    with torch.no_grad():
        orig_embeds = model.showo.get_input_embeddings().weight.data.clone()
        orig_lm_head_weight = model.showo.lm_head.weight.data.clone()
        orig_lm_head_bias = model.showo.lm_head.bias.data.clone()
        
    return tokenizer, model, orig_embeds, orig_lm_head_weight, orig_lm_head_bias, \
           index_no_updates, new_total_vocab, new_token_ids, adj_token_ids, sks_token_id
def check_param(model,args):
    #need to modify
    # for name, param in model.named_parameters():
    #     if "embed_tokens" in name or "lm_head" in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    #statistic
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)

    #statistic
    # for names, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(f"{names} requires_grad") # embed_token, lm_head会更新

    # 统计所有可训练参数数量
    trainable_params_num = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {trainable_params_num}")
    optimizer = torch.optim.AdamW(
        trainable_params, # for optimize the embeddings and the head
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-3,
        eps=1e-08,
    )
    return optimizer
def test_showo(model,image_path):
    pass
def main(args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    if training_args.deepspeed:
        with open(training_args.deepspeed, "r") as f:
            training_args.deepspeed = json.load(f)
    #get initial model
    config = OmegaConf.load(args.config_file)
    tokenizer, uni_prompting, vq_model, model= setup_model(args, config)
    # ref_tokenizer,_,_,ref_model=setup_model(args, config)
    # make filepath to save the result
    concept = args.concept
    if "adrien_brody" in args.pre_trained_ckpt_name:
        args.pre_trained_ckpt_name = args.pre_trained_ckpt_name.replace("adrien_brody", args.concept)
    #set up training arch
    tokenizer, model, orig_embeds, orig_lm_head_weight, \
    orig_lm_head_bias, index_no_updates, new_total_vocab, new_token_ids, adj_token_ids, sks_token_id \
    = update_tokens_load_from_pretrained(args, concept, tokenizer, model, 
                                         args.pre_trained_ckpt_name, 
                                         args.epoch_to_load,
                                         nums_new_token_i_stage_1=args.nums_new_token_i_stage_1,
                                         nums_new_token_i_stage_2=args.nums_new_token_i_stage_2,
                                         need_init=True
                                         )
    # ref_tokenizer, ref_model,_,_, \
    # _,_,_,_,_,_ \
    # = update_tokens_load_from_pretrained(args, concept, ref_tokenizer, ref_model, 
    #                                      args.pre_trained_ckpt_name, 
    #                                      args.epoch_to_load,
    #                                      nums_new_token_i_stage_1=args.nums_new_token_i_stage_1,
    #                                      nums_new_token_i_stage_2=args.nums_new_token_i_stage_2,
    #                                      need_init=True
    #                                      )
    config.new_total_vocab=new_total_vocab
    # set up parameters
    vq_model.requires_grad_ = False
    vq_model.eval()
    model.train()
    optimizer=check_param(model,args)


    trainer_cls = unic_grpo
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model,
        # ref_model=ref_model,
        reward_funcs=reward_funcs,
        args=args,
        train_args=training_args,
        config=config,
        vq_model=vq_model,
        uni_prompting=uni_prompting,
        optimizer=optimizer,
        tokenizer=tokenizer,
        # ref_tokenizer=ref_tokenizer,
        # eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        # attn_implementation=model_args.attn_implementation,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    # trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)





# def print_args(args_obj, prefix=""):
#     print(f"{prefix}{type(args_obj).__name__}")
#     for key in vars(args_obj):
#         value = getattr(args_obj, key)
#         if isinstance(value, (GRPOScriptArguments, GRPOConfig, ModelConfig)):
#             print(f"{prefix}{key}:")
#             print_args(value, prefix)
#         elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (GRPOScriptArguments, GRPOConfig, ModelConfig)):
#             print(f"{prefix} {key} ({len(value)}):")
#             for i, item in enumerate(value):
#                 print(f"{prefix}{i}:")
#                 print_args(item, prefix)
#         else:
#             print(f"{prefix}{key}: {value}")
#     print(f"{prefix}")

import sys
import json
if __name__ == "__main__":
    print(sys.executable)
    parser = TrlParser((GRPOScriptArguments, GrpoConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
