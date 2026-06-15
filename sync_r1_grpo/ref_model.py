import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import random
import json
# import wandb
from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config, image_transform
from transformers import AutoTokenizer
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
import argparse
from argparse import Namespace
from pathlib import Path

def mkdir(path):
    folder_path=Path(path)
    if not folder_path.exists():
        folder_path.mkdir(parents=True)

def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/showo_demo_512x512.yaml")
    # _512x512
    parser.add_argument("--data_root", type=str, default="path/to/unictokens_data")
    parser.add_argument("--concept", type=str, default="None")
    parser.add_argument("--pre_trained_ckpt_name", type=str, default="path/to/second_stage_checkpoint_dir")
    parser.add_argument("--epoch_to_load", type=int, default=15)
    parser.add_argument("--nums_new_token_i_stage_1", type=int, default=16)
    parser.add_argument("--nums_new_token_i_stage_2", type=int, default=8)
    parser.add_argument("--num_gen_images", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="gen_saved")
    parser.add_argument("--inverse_prompt", default=True, action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--t2i_batch_size", type=int, default=4)
    
    return parser.parse_args()

NAME={"concept","pre_trained_ckpt_name","epoch_to_load","inverse_prompt"}
class ref_model:
    def __init__(self,ref_args=None,num_gen=1,num_batch=1,device='cuda:3'):
        self.args = get_test_args()
        args_dict=vars(self.args)
        if ref_args:
            ref_dict=vars(ref_args)
            for n in NAME:
                args_dict[n]=ref_dict[n]
        self.args = Namespace(**args_dict)
        self.args.num_gen_images=num_gen
        self.args.t2i_batch_size=num_batch
        self.args.device=device
       
        path=os.path.join('./','ref_image',self.args.concept)
        mkdir(path)
        self.save_dir=path
        self.config = OmegaConf.load(self.args.config_file)
        self.device = torch.device(self.args.device)
        
        self.config.mode = 't2i'
        self.config.batch_size = 2
        self.config.generation_timesteps = 50
        self.config.guidance_scale = 5
        
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.abspath(self.config.model.showo.llm_model_path), padding_side="left",local_files_only=True)

        self.uni_prompting = UniversalPrompting(self.tokenizer, max_text_len=self.config.dataset.preprocessing.max_seq_length,
                                            special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                            ignore_id=-100, cond_dropout_prob=self.config.training.cond_dropout_prob)

        self.vq_model = MAGVITv2.from_pretrained(os.path.abspath(self.config.model.vq_model.vq_model_name),local_files_only=True).to(self.device)
        self.vq_model.requires_grad_(False)
        self.vq_model.eval()

        self.model = Showo.from_pretrained(os.path.abspath(self.config.model.showo.pretrained_model_path),local_files_only=True,low_cpu_mem_usage=False).to(self.device)
        self.model.eval()

        # load from users passed arguments
        self.config.training.batch_size = self.config.batch_size
        self.config.training.guidance_scale = self.config.guidance_scale
        self.config.training.generation_timesteps = self.config.generation_timesteps
        
        data_root = self.args.data_root
        concept = self.args.concept
        ckpt_name = self.args.pre_trained_ckpt_name
        epoch2load = self.args.epoch_to_load
        
        ckpt_path = ckpt_name
        ckpt_embed_path = os.path.join(ckpt_path, f"epoch_{epoch2load}_embed.pt")
        ckpt_lm_head_weight_path = os.path.join(ckpt_path, f"epoch_{epoch2load}_lm_head_weight.pt")
        ckpt_lm_head_bias_path = os.path.join(ckpt_path, f"epoch_{epoch2load}_lm_head_bias.pt")

        nums_new_token_i_stage_1 = self.args.nums_new_token_i_stage_1
        nums_new_token_i_stage_2 = self.args.nums_new_token_i_stage_2
        
        new_tokens_total = nums_new_token_i_stage_1 + nums_new_token_i_stage_2
        new_tokens_total = [f"<{concept}>"] + [f"<token_{i}>" for i in range(new_tokens_total)]
        num_new_tokens_total = len(new_tokens_total)  # 21
        sks_token = [f"<{concept}>"]
        
        new_tokens_stage_1 = [f"<token_{i}>" for i in range(nums_new_token_i_stage_1)]
        new_tokens_stage_2 = [f"<token_{i}>" for i in range(nums_new_token_i_stage_1, nums_new_token_i_stage_1 + nums_new_token_i_stage_2)]
        

        # Known original parameters
        # Text token count (ID 0-50304)
        original_text_vocab_size = len(self.tokenizer)  
        # Image token count (original ID 50305-58497)
        original_image_vocab_size = self.model.showo.get_input_embeddings().num_embeddings - len(self.tokenizer)

        original_total_vocab = original_text_vocab_size + original_image_vocab_size  # 58498

        # New parameters
        new_text_vocab_size = original_text_vocab_size + num_new_tokens_total  # 50305 + 17 = 50322
        new_total_vocab = original_total_vocab + num_new_tokens_total          # 58498 + 17 = 58515

        # ------------------------------
        # Step 1: Modify the Tokenizer's vocabulary
        # ------------------------------

        # Add new tokens to positions 50305-50321
        num_new_tokens = self.tokenizer.add_tokens(new_tokens_total)
        new_token_ids_total = self.tokenizer.convert_tokens_to_ids(new_tokens_total)
        print("New token IDs:", new_token_ids_total)  # Should output 50305-50321
        sks_token_id = self.tokenizer.convert_tokens_to_ids(sks_token)
        print("sks_token_id:", sks_token_id)  # Should output <concept> token ID
        stage_1_token_ids = self.tokenizer.convert_tokens_to_ids(new_tokens_stage_1)  # Should output 50305-50320
        stage_2_token_ids = self.tokenizer.convert_tokens_to_ids(new_tokens_stage_2)  # Should output 50305-50320

        print("stage_1_token_ids:", stage_1_token_ids)  # Should output 50305-50320
        print("stage_2_token_ids:", stage_2_token_ids)  # Should output 50305-50320
        
        # ------------------------------
        # Step 2: Adjust model weights
        # ------------------------------
        with torch.no_grad():
            # Get embedding layer weights
            embeddings = self.model.showo.get_input_embeddings().weight.data
            
            # Expand embedding layer (58498 -> 58515)
            self.model.showo.resize_token_embeddings(new_total_vocab)
            # new_embeddings = model.showo.get_input_embeddings().weight.data

            # Move original Image Token weights back by 17 positions
            original_image_weights = embeddings[original_text_vocab_size:original_total_vocab].clone()
            self.model.showo.get_input_embeddings().weight.data[new_text_vocab_size:new_total_vocab] = original_image_weights
            
            # Initialize new token weights (using original last 17 text tokens)
            if os.path.exists(ckpt_embed_path):
                ckpt_embed_weight = torch.load(ckpt_embed_path)
                with torch.no_grad():
                    self.model.showo.get_input_embeddings().weight.data[original_text_vocab_size:new_text_vocab_size] = ckpt_embed_weight.to(self.model.showo.get_input_embeddings().weight.device)
            else:
                raise ValueError("Embedding weights do not exist!")
                
            # new_text_weights = embeddings[original_text_vocab_size - num_new_tokens : original_text_vocab_size].clone()
            # model.showo.get_input_embeddings().weight.data[original_text_vocab_size : new_text_vocab_size] = new_text_weights
            # print(model.showo.lm_head.weight.data.shape[1])
            # Handle lm_head (assuming weight sharing with embedding layer)
            if self.model.showo.lm_head.weight.data.shape[0] == new_total_vocab:
                # Expand lm_head weights
                lm_head = self.model.showo.lm_head
                new_lm_head = torch.nn.Linear(
                    lm_head.in_features, 
                    new_total_vocab, 
                    bias=hasattr(lm_head, 'bias')
                )
                new_lm_head.weight.data = lm_head.weight.data.clone()
                new_lm_head.weight.data[new_text_vocab_size:new_total_vocab] = lm_head.weight.data[original_text_vocab_size:original_total_vocab]
                
                if os.path.exists(ckpt_lm_head_weight_path):
                    ckpt_lm_head_weight = torch.load(ckpt_lm_head_weight_path)
                    with torch.no_grad():
                        new_lm_head.weight.data[original_text_vocab_size:new_text_vocab_size] = ckpt_lm_head_weight.to(new_lm_head.weight.device)
                else:
                    raise ValueError("lm_head weights do not exist!")
                # new_lm_head.weight.data[original_text_vocab_size:new_text_vocab_size] = lm_head.weight.data[original_text_vocab_size - num_new_tokens : original_text_vocab_size]
                if hasattr(lm_head, 'bias'):
                    new_lm_head.bias.data = lm_head.bias.data.clone()
                    new_lm_head.bias.data[new_text_vocab_size:new_total_vocab] = lm_head.bias.data[original_text_vocab_size:original_total_vocab]
                    
                    if os.path.exists(ckpt_lm_head_bias_path):
                        ckpt_lm_head_bias = torch.load(ckpt_lm_head_bias_path)
                        with torch.no_grad():
                            new_lm_head.bias.data[original_text_vocab_size:new_text_vocab_size] = ckpt_lm_head_bias.to(new_lm_head.weight.device)
                    else:
                        raise ValueError("lm_head bias do not exist!")
                    # new_lm_head.bias.data[original_text_vocab_size:new_text_vocab_size] = lm_head.bias.data[original_text_vocab_size - num_new_tokens : original_text_vocab_size]
                
                self.model.showo.lm_head = new_lm_head
            else:
                raise ValueError("lm_head weights do not match the input embeddings!")

        self.config.model.showo.llm_vocab_size = len(self.tokenizer) - 10
        self.init_param_snapshot = self._save_param_snapshot()  # 存储初始化时的参数状怄1�7
        print("ref_model 初始化完成，已保存参数")
    def _save_param_snapshot(self):
        snapshot = {}
        device = self.device

        # 1. 输入嵌入层参数（朢�易变化的新token权重＄1�7
        embed_layer = self.model.showo.get_input_embeddings()
        # 计算嵌入层权重的哈希值（避免存储全量权重，用哈希快��对比）
        embed_weight_hash = torch.sum(embed_layer.weight.data).item()  # 用求和哈希快速校验（也可用md5，求和更轻量＄1�7
        embed_weight_shape = embed_layer.weight.data.shape  # 记录形状（防止维度变化）
        snapshot["input_embedding"] = {
            "hash": round(embed_weight_hash, 6),  # 保留6位小数，避免浮点精度问题
            "shape": embed_weight_shape
        }

        # 2. lm_head层参数（输出层，与嵌入层权重关联＄1�7
        lm_head = self.model.showo.lm_head
        lm_weight_hash = torch.sum(lm_head.weight.data).item()
        lm_weight_shape = lm_head.weight.data.shape
        snapshot["lm_head_weight"] = {
            "hash": round(lm_weight_hash, 6),
            "shape": lm_weight_shape
        }
        # 若lm_head有bias，也记录bias状��1�7
        if hasattr(lm_head, "bias") and lm_head.bias is not None:
            lm_bias_hash = torch.sum(lm_head.bias.data).item()
            snapshot["lm_head_bias"] = {
                "hash": round(lm_bias_hash, 6),
                "shape": lm_head.bias.data.shape
            }

        # 3. 新增token的权重（重点关注用户添加的1�7<concept>咄1�7<token_x>＄1�7
        # 获取新增token的ID范围（original_text_vocab_size 刄1�7 new_text_vocab_size＄1�7
        original_text_vocab_size = len(self.tokenizer) - len(self.tokenizer.added_tokens_encoder)  # 原始词表大小（减去新增token＄1�7
        new_text_vocab_size = len(self.tokenizer)  # 新增后的词表大小
        new_token_embed = embed_layer.weight.data[original_text_vocab_size:new_text_vocab_size]
        snapshot["new_token_embedding"] = {
            "hash": round(torch.sum(new_token_embed).item(), 6),
            "shape": new_token_embed.shape,
            "token_ids": list(range(original_text_vocab_size, new_text_vocab_size))  # 新增token的ID范围
        }

        return snapshot
    # ------------------------------
    def reference(self,prompt,epoch,batch_idx,rank,group_id):
        self.model.config.mask_token_id = self.model.showo.get_input_embeddings().num_embeddings - 1
        mask_token_id = self.model.showo.get_input_embeddings().num_embeddings - 1
        self.config.model.showo.llm_vocab_size = len(self.tokenizer) - 10
                
        condition = prompt
        
        for i in range(0, self.args.num_gen_images, self.args.t2i_batch_size):
            image_tokens = torch.ones((self.args.t2i_batch_size, self.config.model.showo.num_vq_tokens),
                                        dtype=torch.long, device=self.device) * mask_token_id    # shape [batch, num_vq_tokens] [1, 256], fill with mask token
            conditions = [condition] * self.args.t2i_batch_size
            input_ids, _ = self.uni_prompting((conditions, image_tokens), 't2i_gen')   # [1, 387]

            if self.config.training.guidance_scale > 0:
                uncond_input_ids, _ = self.uni_prompting(([''] * self.args.t2i_batch_size, image_tokens), 't2i_gen')
                # [1, 387], == [PAD] * 126 + <|t2i|> + <|endoftext|> + <|endoftext|> + <|soi|> + [MASK] * 256 + <|eoi|> ## no prompt
                attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),    # [2, 387]
                                                                    pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
            else:
                attention_mask = create_attention_mask_predict_next(input_ids,
                                                                    pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
                uncond_input_ids = None
            # attention_mask [2, 1, 387, 387]

            if self.config.get("mask_schedule", None) is not None:
                schedule = self.config.mask_schedule.schedule
                args = config.mask_schedule.get("params", {})
                mask_schedule = get_mask_chedule(schedule, **args)
            else:
                mask_schedule = get_mask_chedule(self.config.training.get("mask_schedule", "cosine"))

            with torch.inference_mode():
                gen_token_ids,logits = self.model.t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=self.config.training.guidance_scale,
                    temperature=self.config.training.get("generation_temperature",1.0),
                    timesteps=self.config.training.generation_timesteps,
                    noise_schedule=mask_schedule,
                    noise_type=self.config.training.get("noise_type", "mask"),
                    seq_len=self.config.model.showo.num_vq_tokens,
                    uni_prompting=self.uni_prompting,
                    config=self.config,
                    return_logits=True,
                )
            if batch_idx % 5 ==4:
                gen_token_ids = torch.clamp(gen_token_ids, max=self.config.model.showo.codebook_size - 1, min=0)
                images = self.vq_model.decode_code(gen_token_ids)
                images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                images *= 255.0
                images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                pil_images = [Image.fromarray(image) for image in images]
                for j in range(self.args.t2i_batch_size):
                    gen_image = pil_images[j]
                    gen_image.save(os.path.join(self.save_dir, f"E{epoch}B{batch_idx}R{rank}G{group_id}N{self.args.t2i_batch_size * i + j}.png"))
            return logits
    def check_param_change(self, verbose=True) -> bool:
        current_snapshot = self._save_param_snapshot()
        has_change = False
        change_log = []

        for param_key in self.init_param_snapshot.keys():
            init_data = self.init_param_snapshot[param_key]
            current_data = current_snapshot.get(param_key, None)

            if current_data is None:
                has_change = True
                continue

            if init_data["shape"] != current_data["shape"]:
                has_change = True
                change_log.append(
                    f"变化！初始化时 {init_data['shape']}当前 {current_data['shape']}"
                )
                continue

            if init_data["hash"] != current_data["hash"]:
                has_change = True
                change_log.append(
                    f"变化！初始化时{init_data['hash']}，当前哈希{current_data['hash']}"
                )
            else:
                if verbose:
                    change_log.append(f"参数 {param_key} 无变化")

        # 打印对比结果
        if verbose:
            print("\n" + "="*50)
            print("📊 ref_model 参数变化校验结果")
            print("="*50)
            for log in change_log:
                print(log)
            print("="*50)
            if not has_change:
                print("🎉 参数未变化")
            else:
                print("⚠️  发现参数变化")
            print("="*50 + "\n")

        return has_change


if __name__ == "__main__":
    model=ref_model()
    model.reference('a photo of cat')
