import os
import textwrap
from glm_api import *
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pdata import image_transform
import numpy as np
from pathlib import Path
import torch
import torch.utils.data
import json
import transformers
# import deepspeed
from datasets import Dataset, IterableDataset
from packaging import version
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from clip_eval import SHOWO_P_CLIPEvaluator
from models import Showo, MAGVITv2, get_mask_chedule
from models.sampling import mask_by_random_topk
from clip.model import build_model
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, create_attention_mask_for_mmu
from transformers import get_cosine_schedule_with_warmup
# from transformers.integrations import is_deepspeed_zero3_enabled 
# from gpttest import chat_with_images_gpt
# from api import evaluate,extract
# from record_best import update_best_results
from torch.utils.checkpoint import checkpoint
from transformers import (

    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,

    Trainer,
    TrainerCallback,
    is_wandb_available,
)
# from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
# from deepspeed import zero
from transformers.utils import is_peft_available
from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model,  unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
import shutil
from utils import *
import copy
import random
import re
from typing import List, Any
import logging
from ref_model import ref_model
tt=local_rank = int(os.environ.get("LOCAL_RANK", 0))
# 1. 初始化日志配置（仅需执行一次，通常在程序入口）


generate_prompt='''
You are asked to generate an image based on this prompt: "{}"
Provide a brief, precise visualization of all elements in the prompt. Your description should:
1. Include every object mentioned in the prompt
2. Specify visual attributes (color, number, shape, texture) if specified in the prompt
3. Clarify relationships (e.g., spatial) between objects if specified in the prompt
4. Be concise (50 words or less)
5. Focus only on what's explicitly stated in the prompt
6. Do not elaborate beyond the attributes or relationships specified in the prompt
Do not miss objects. Output your visualization directly without explanation: 
'''


# Example local image path placeholder

class unic_grpo(Trainer):
    def __init__(
        self,
        model,
        reward_funcs,
        args,
        train_args,
        config,
        vq_model,
        uni_prompting,
        optimizer,
        tokenizer,
        peft_config,
    ):
        self.local_rank = local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.args=args
        if self.local_rank == 0:
            self.target_dtype = torch.float32
        else:
            self.target_dtype = torch.float32  # 临时占位
        
        # 广播target_dtype到所有进程（用整数标识：0=fp32,1=fp16,2=bf16）
        if dist.is_initialized():
            dtype_code = torch.tensor(
                2 if self.target_dtype == torch.float32 else 1 if self.target_dtype == torch.float16 else 0,
                device=self.args.device
            )
            dist.broadcast(dtype_code, src=0)
            # 非0进程更新target_dtype
            if self.local_rank != 0:
                code = dtype_code.item()
                self.target_dtype = torch.float32 if code == 2 else torch.float16 if code == 1 else torch.float32
        mkdir(os.path.join(self.args.save_dir,'logs/',self.args.concept))
        logging.basicConfig(
            level=logging.INFO,  # 日志级别：只记录 >= INFO 的信息（过滤 DEBUG）
            format="%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",  # 日志格式
            handlers=[
                logging.StreamHandler(),  # 输出到控制台
                logging.FileHandler(os.path.join(self.args.save_dir,'logs/',self.args.concept,f"{tt}"+"_record.log"), encoding="utf-8")  # 输出到文件（utf-8避免中文乱码）
            ]
        )
        logging.info(f"进程 {self.local_rank} 最终target_dtype: {self.target_dtype}")

        # freeze all vision encoders
        for name, param in model.named_parameters():
            if name.startswith("vision_model") or name.startswith("aligner") or name.startswith("gen"): # choose whatever you like here
                param.requires_grad = False
        for name, param in vq_model.named_parameters():
            param.requires_grad = False

        
        # # Reference model
        self.model=model.to(self.target_dtype)
        self.model=check_dtype(model,self.target_dtype)
        self.optimizer=optimizer
        num_training_steps = args.batch_num * args.epoch
        num_warmup_steps = args.batch_num  # 10%的热身步数

        # 创建调度器
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        self.num_generations = args.num_gen  # = G in the GRPO paper
        self.group=int(self.num_generations/args.num_gpus)
        self.num_generations=int(self.num_generations/self.group)
        self.beta = train_args.beta
        self._metrics = defaultdict(list)
        self.model_accepts_loss_kwargs = False
        self.config=config
        self.threshold=0.50
        self.accelerate_rate=0.25
        self.vq_model=vq_model
        self.vq_model=check_dtype(self.vq_model,self.target_dtype)
        self.uni_prompting=uni_prompting
        self.tokenizer=tokenizer
        self.beta=0.01
        self.rate=0.5
        self.epsilon=0.2
        self.split_rate=0.5
        self.ref_model=ref_model(self.args)
        self.ref_model.args.device = str(self.args.device)
        self.ref_model.device = torch.device(self.args.device)
        self.ref_model.model = self.ref_model.model.to(self.args.device)
        self.ref_model.vq_model = self.ref_model.vq_model.to(self.args.device)
        self.ref_model.model=check_dtype(self.ref_model.model,self.target_dtype)
        self.info=read_json_to_dict(os.path.join(self.args.data_root,'concept/train',self.args.concept,'info.json'))
        self.t2i_condition=read_json_to_dict(os.path.join(self.args.data_root,'concept/test',self.args.concept,'t2i_conditions.json'))
        self.vqa=read_json_to_dict(os.path.join(self.args.data_root,'concept/train',self.args.concept,'conversations.json'))
        self.ablation_mode=self.args.ablation_mode
        self.mode=[]
        if self.ablation_mode=='ur':
            self.mode=['u' for _ in range(int(self.args.batch_num/2))]
            for _  in range(self.args.batch_num-int(self.args.batch_num/2)):
                self.mode.append('r')
        elif self.ablation_mode=='ru':
            self.mode=['r' for _ in range(self.args.batch_num-int(self.args.batch_num/2))]
            for _  in range(int(self.args.batch_num/2)):
                self.mode.append('u')
        elif self.ablation_mode=='r':
            for _ in range(self.args.batch_num):
                self.mode.append('r')
        elif self.ablation_mode=='u':
            for _ in range(self.args.batch_num):
                self.mode.append('u')
        else:
            for _ in range(self.args.batch_num):
                rand_seed=random.random()
                modified_rate=int(self.args.batch_num/2)/self.args.batch_num
                if rand_seed<=modified_rate:
                    self.mode.append('u')
                else:
                    self.mode.append('r')
        logging.info(f"using ablation mode:{self.ablation_mode}")

        mkdir(os.path.join(self.args.save_dir,'model_weights',self.args.concept))
        # if self.local_rank==0:
        #     save_distributed_model(self.model,self.optimizer,os.path.join(self.args.save_dir,'model_weights',self.args.concept),epoch=0)
        #     self.model,self.optimizer,_=load_distributed_model(self.model,self.optimizer,os.path.join(self.args.save_dir,'model_weights',self.args.concept),device=self.args.device)

        # if self.deepspeed_enabled:
        #     from deepspeed import zero
        #     self.model, _ = deepspeed.initialize(model=model, config_params=train_args.deepspeed)
        #     self.ref_model, _ = deepspeed.initialize(model=self.ref_model, config_params=train_args.deepspeed, eval_mode=True)
        self.concept_train_path=os.path.join(self.args.data_root,'concept/train',self.args.concept)
        self.BLACK_IMAGE_PATH=os.path.join(self.args.data_root,'black_512x512.png')
    def _prepare_inputs(self, inputs):
        return inputs

    def _build_policy_eval_inputs(self, conditions, image_token_ids):
        input_ids, _ = self.uni_prompting((conditions, image_token_ids), 't2i_gen')
        input_ids = input_ids.to(dtype=torch.long, device=self.args.device)

        if self.config.guidance_scale > 0:
            uncond_input_ids, _ = self.uni_prompting(([''] * len(conditions), image_token_ids), 't2i_gen')
            attention_mask = create_attention_mask_predict_next(
                torch.cat([input_ids, uncond_input_ids], dim=0),
                pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                rm_pad_in_image=True,
            ).to(self.target_dtype)
        else:
            uncond_input_ids = None
            attention_mask = create_attention_mask_predict_next(
                input_ids,
                pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                rm_pad_in_image=True,
            ).to(self.target_dtype)

        return input_ids, uncond_input_ids, attention_mask.type(self.target_dtype)

    def _maskgit_forward_logits(self, model, input_ids, attention_mask, uncond_prefix=None):
        if uncond_prefix is not None and self.config.guidance_scale > 0:
            uncond_input_ids = torch.cat(
                [uncond_prefix, input_ids[:, self.config.dataset.preprocessing.max_seq_length + 1:]], dim=1
            )
            model_input = torch.cat([input_ids, uncond_input_ids], dim=0)
            cond_logits, uncond_logits = model(model_input, attention_mask=attention_mask).chunk(2)
            logits = (1 + self.config.guidance_scale) * cond_logits - self.config.guidance_scale * uncond_logits
        else:
            logits = model(input_ids, attention_mask=attention_mask)

        num_vq_tokens = self.config.model.showo.num_vq_tokens
        num_new_special_tokens = self.config.model.showo.num_new_special_tokens
        return logits[
            :,
            -(num_vq_tokens + 1):-1,
            self.config.model.showo.llm_vocab_size + num_new_special_tokens:-1,
        ]

    def _run_maskgit_rollout(
        self,
        input_ids,
        attention_mask,
        noise_schedule,
        temperature,
        timesteps,
        uncond_input_ids=None,
        checkpoint_step=None,
        stop_at_checkpoint=False,
        state=None,
    ):
        token_slice = slice(-(self.config.model.showo.num_vq_tokens + 1), -1)
        num_new_special_tokens = self.config.model.showo.num_new_special_tokens
        mask_token_id = self.model.config.mask_token_id

        if state is None:
            input_ids = input_ids.clone()
            input_ids_minus_lm_vocab_size = input_ids[:, token_slice].clone()
            input_ids_minus_lm_vocab_size = torch.where(
                input_ids_minus_lm_vocab_size == mask_token_id,
                mask_token_id,
                input_ids_minus_lm_vocab_size
                - self.config.model.showo.llm_vocab_size
                - num_new_special_tokens,
            )
            uncond_prefix = None
            if uncond_input_ids is not None and self.config.guidance_scale > 0:
                uncond_prefix = uncond_input_ids[:, : self.config.dataset.preprocessing.max_seq_length + 1].clone()
            step_start = 0
            records = []
        else:
            input_ids = state["input_ids"].clone()
            input_ids_minus_lm_vocab_size = state["input_ids_minus_lm_vocab_size"].clone()
            uncond_prefix = state["uncond_prefix"]
            step_start = state["step"]
            temperature = state["temperature"]
            records = list(state["records"])

        with torch.no_grad():
            for step in range(step_start, timesteps):
                logits = self._maskgit_forward_logits(
                    self.model, input_ids, attention_mask, uncond_prefix=uncond_prefix
                )
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                probs = log_probs.exp()
                sampled = probs.reshape(-1, probs.size(-1))
                sampled_ids = torch.multinomial(sampled, 1)[:, 0].view(*probs.shape[:-1])

                unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
                chosen_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
                chosen_logps = torch.gather(log_probs, -1, chosen_ids.unsqueeze(-1)).squeeze(-1)

                records.append(
                    {
                        "input_ids": input_ids.detach().clone(),
                        "chosen_ids": chosen_ids.detach().clone(),
                        "old_logps": chosen_logps.detach().clone(),
                    }
                )

                ratio = 1.0 * (step + 1) / timesteps
                mask_ratio = noise_schedule(torch.tensor(ratio, device=probs.device))
                selected_probs = chosen_logps.exp()
                selected_probs = torch.where(
                    unknown_map,
                    selected_probs,
                    torch.full_like(selected_probs, torch.finfo(selected_probs.dtype).max),
                )
                mask_len = (self.config.model.showo.num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(probs.device)
                mask_len = torch.max(
                    torch.tensor([1], device=probs.device),
                    torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len),
                )

                if stop_at_checkpoint and checkpoint_step is not None and (step + 1) == checkpoint_step:
                    preview_input_ids = input_ids.clone()
                    preview_input_ids[:, token_slice] = (
                        chosen_ids + self.config.model.showo.llm_vocab_size + num_new_special_tokens
                    )
                    next_state = {
                        "step": step + 1,
                        "input_ids": preview_input_ids.detach().clone(),
                        "input_ids_minus_lm_vocab_size": chosen_ids.detach().clone(),
                        "uncond_prefix": None if uncond_prefix is None else uncond_prefix.detach().clone(),
                        "temperature": temperature * (1.0 - ratio),
                        "records": records,
                        "attention_mask": attention_mask.detach().clone(),
                    }
                    return chosen_ids.detach().clone(), next_state

                temperature = temperature * (1.0 - ratio)
                masking = mask_by_random_topk(mask_len, selected_probs, temperature)
                input_ids[:, token_slice] = torch.where(
                    masking,
                    torch.full_like(chosen_ids, mask_token_id),
                    chosen_ids + self.config.model.showo.llm_vocab_size + num_new_special_tokens,
                )
                input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, chosen_ids)

        trajectory = {
            "records": records,
            "attention_mask": attention_mask.detach().clone(),
            "uncond_prefix": None if uncond_prefix is None else uncond_prefix.detach().clone(),
        }
        return input_ids_minus_lm_vocab_size.detach().clone(), trajectory

    def _flatten_old_trajectory_logps(self, trajectories):
        flattened = [torch.cat([record["old_logps"] for record in traj["records"]], dim=1) for traj in trajectories]
        return torch.cat(flattened, dim=0)

    def _compute_trajectory_logps(self, model, trajectories, requires_grad):
        grad_context = torch.enable_grad if requires_grad else torch.no_grad
        flattened = []
        with grad_context():
            for traj in trajectories:
                attention_mask = traj["attention_mask"].to(device=self.args.device, dtype=self.target_dtype)
                uncond_prefix = traj["uncond_prefix"]
                if uncond_prefix is not None:
                    uncond_prefix = uncond_prefix.to(device=self.args.device, dtype=torch.long)

                step_logps = []
                for record in traj["records"]:
                    input_ids = record["input_ids"].to(device=self.args.device, dtype=torch.long)
                    chosen_ids = record["chosen_ids"].to(device=self.args.device, dtype=torch.long)
                    logits = self._maskgit_forward_logits(
                        model, input_ids, attention_mask, uncond_prefix=uncond_prefix
                    )
                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                    chosen_logps = torch.gather(log_probs, -1, chosen_ids.unsqueeze(-1)).squeeze(-1)
                    step_logps.append(chosen_logps)
                flattened.append(torch.cat(step_logps, dim=1))

        return torch.cat(flattened, dim=0)

    def _compute_selected_token_logps(self, model, conditions, sampled_token_ids, requires_grad):
        sampled_token_ids = sampled_token_ids.to(device=self.args.device, dtype=torch.long)
        num_new_special_tokens = self.config.model.showo.num_new_special_tokens
        model_image_token_ids = sampled_token_ids + self.config.model.showo.llm_vocab_size + num_new_special_tokens

        input_ids, uncond_input_ids, attention_mask = self._build_policy_eval_inputs(
            conditions, model_image_token_ids
        )

        grad_context = torch.enable_grad if requires_grad else torch.no_grad
        with grad_context():
            if uncond_input_ids is not None and self.config.guidance_scale > 0:
                model_input = torch.cat([input_ids, uncond_input_ids], dim=0)
                cond_logits, uncond_logits = model(model_input, attention_mask=attention_mask).chunk(2)
                logits = (1 + self.config.guidance_scale) * cond_logits - self.config.guidance_scale * uncond_logits
            else:
                logits = model(input_ids, attention_mask=attention_mask)

            logits = logits[
                :,
                -(self.config.model.showo.num_vq_tokens + 1):-1,
                self.config.model.showo.llm_vocab_size + num_new_special_tokens:-1,
            ]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            selected_logps = torch.gather(log_probs, -1, sampled_token_ids.unsqueeze(-1)).squeeze(-1)

        del input_ids, uncond_input_ids, attention_mask
        return selected_logps

    def train(self,return_outputs=False, num_items_in_batch=None):
        counter=0
        loss_final=[]
        accelerate_counter=0
        fm=''
        for f in ['0.png','0.jpg','0.jpeg']:
            ref_path = os.path.join(self.args.data_root,'concept/train',self.args.concept,f)
            if Path(ref_path).exists():
                fm=f
                break
        clip_model=SHOWO_P_CLIPEvaluator(
                # "cuda:3",
                self.local_rank,
                clip_model=os.path.join(self.args.work_dir,'ViT-B-32.pt'),
                data_root=self.args.data_root,
                work_dir=self.args.work_dir,
                save_dir=fm,
                dtype=self.target_dtype
            )
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # other setting
        self.model.config.mask_token_id = self.model.showo.get_input_embeddings().num_embeddings - 1
        self.model.mask_token_id = self.model.showo.get_input_embeddings().num_embeddings - 1
        mask_schedule = get_mask_chedule(self.config.training.get("mask_schedule", "cosine"))
        mask_id = self.model.mask_token_id
        mask_dtype = self.model.showo.get_input_embeddings().weight.dtype
        self.model.output_size = self.config.new_total_vocab
        save_dir=self.args.save_dir
        for epoch in range(self.args.epoch):
            print(f"Epoch {epoch+1}")
            loss_list = []
            counter=0
            for batch_idx in tqdm(range(self.args.batch_num)):
                accounter=0
                torch.cuda.empty_cache()
                batch_size_t2i = self.args.batch_size

                #realize t2i inference
                self.config.model.showo.llm_vocab_size = len(self.tokenizer) - 10
                self.config.generation_timesteps = 50
                self.config.guidance_scale = 5

                nums_new_token_i_stage_1 = self.args.nums_new_token_i_stage_1
                nums_new_token_i_stage_2 = self.args.nums_new_token_i_stage_2
                new_tokens_stage_1 = [f"<token_{i}>" for i in range(nums_new_token_i_stage_1)]
                new_tokens_stage_2 = [f"<token_{i}>" for i in range(nums_new_token_i_stage_1, nums_new_token_i_stage_1 + nums_new_token_i_stage_2)]

                global_id=[]
                global_trajectories=[]
                reward_text=[]
                for group_id in range(self.group):

                    logging.info(f"gpu {self.local_rank},number {group_id}")
                    mti=self.model.config.mask_token_id
                    self.model.config.mask_token_id = self.model.showo.get_input_embeddings().num_embeddings - 1
                    mask_token_id = self.model.showo.get_input_embeddings().num_embeddings - 1
                    image_tokens_infer = torch.ones((batch_size_t2i, self.config.model.showo.num_vq_tokens),
                                            dtype=torch.long, device=self.args.device) * mask_token_id
                    
                    condition='A photo of '
                    for token in new_tokens_stage_1:
                        condition+=token
                    for token in new_tokens_stage_2:
                        condition+=token
                    condition+=f'<{self.args.concept}>.\n'
                    if self.args.semantic:
                        if self.mode[batch_idx]=='r':
                            # us_prompt="what is in the image?"
                            # us_prompt="Describe the person in this image using 3-5 concise descriptive adjectives focused on appearance, expression, and demeanor.Do not output extra information except these adjectives."
                            r=random.randint(1,int((len(self.t2i_condition["personalized_driven_generation"])+1)*self.rate))
                            self.r=r
                            question=self.t2i_condition["personalized_driven_generation"][r-1]
                            separator = f"<{self.args.concept}>"

                            # 分割字符串（split返回列表，最多分割1次，确保只分前后两部分）
                            parts = question.split(separator, 1)  # split(sep, maxsplit=1)

                            # 提取前后内容（处理分割失败的边缘情况，如分隔符不存在）
                            if len(parts) == 2:
                                before = parts[0].strip()  # 前半部分（strip()可选：去除前后多余空格）
                                after = parts[1].strip()   # 后半部分
                            else:
                                before = question  # 若没有分隔符，前半部分为原字符串
                                after = ""             # 后半部分为空
                            condition=before+' '
                            for token in new_tokens_stage_1:
                                condition+=token
                            for token in new_tokens_stage_2:
                                condition+=token
                            condition=condition+f'<{self.args.concept}> '+after+'.'
                            us_prompt=f'''
                            Below is some information about <{self.args.concept}> : {self.info['extra_info']}
                            Please make inferences based on the following prompt: {question}. 
                            If the prompt relates to a specific item from the aforementioned information list, 
                            output and only output that exact item. 
                            If the prompt does not relate to any item in the list, 
                            output nothing (i.e., an empty response).
                            '''
                            image_ori = Image.open(self.BLACK_IMAGE_PATH).convert("RGB")
                            # tranforming the image to the required resolution
                            image = image_transform(image_ori, resolution = self.config.dataset.params.resolution).to(self.args.device)
                            image = image.unsqueeze(0)


                            image_tokens_mmu = self.vq_model.get_code(image)
                            image_tokens = image_tokens_mmu + len(self.uni_prompting.text_tokenizer)
                            us_input = self.uni_prompting.text_tokenizer(['USER: ' + us_prompt + ' ASSISTANT:'])['input_ids']

                            us_input = torch.tensor(us_input).to(self.args.device)
                            us_input = torch.cat([
                                (torch.ones(us_input.shape[0], 1) * self.uni_prompting.sptids_dict['<|mmu|>']).to(self.args.device),
                                (torch.ones(us_input.shape[0], 1) * self.uni_prompting.sptids_dict['<|soi|>']).to(self.args.device),
                                image_tokens,
                                (torch.ones(us_input.shape[0], 1) * self.uni_prompting.sptids_dict['<|eoi|>']).to(self.args.device),
                                (torch.ones(us_input.shape[0], 1) * self.uni_prompting.sptids_dict['<|sot|>']).to(self.args.device),
                                us_input
                            ], dim=1).long()
                            us_mask = create_attention_mask_for_mmu(us_input.to(self.args.device),
                                                        eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>'])).to(self.target_dtype)
                            us_mask = us_mask.type(self.target_dtype)
                            us_toks_list = self.model.mmu_generate(
                                us_input, 
                                attention_mask=us_mask,
                                top_k=5,
                                eot_token=self.uni_prompting.sptids_dict['<|eot|>'],
                            )
                            us_toks_list= torch.stack(us_toks_list).squeeze()[None]
                            more_prompt = self.uni_prompting.text_tokenizer.batch_decode(us_toks_list, skip_special_tokens=True)[0].strip()
                            
                            logging.info(f"Question:{question}")
                            logging.info(f"Answer:{more_prompt}")
                            text_score=[]
                            text_gt=[0 for _ in range(len(self.info['extra_info']))]
                            text_gt[r-1]=1
                            for r in range(len(self.info['extra_info'])):
                                text_score.append(calculate_bleu(self.info['extra_info'][r],more_prompt))
                            text_mean=sum(text_score)
                            t_score=[0]*len(self.info['extra_info'])
                            if text_mean>0:
                                t_score=[t/text_mean for t in text_score]
                            print('gtscore',text_gt,text_mean)
                            print('bleuscore',t_score)
                            final_score=(2.0-calculate_distance(text_gt,t_score))/2
                            if final_score<=0.5:
                                final_score=0
                            else:
                                final_score=(final_score-0.5)*2
                            reward_text.append(final_score)
                            logging.info(f"{self.local_rank}: Bleu score is {reward_text}")
                            if self.args.llm=='gemini':
                                more_prompt=extract(more_prompt,self.info["class"])
                            elif self.args.llm=='glm':
                                more_prompt=glm_extract(more_prompt,self.info["class"])
                            cla=self.info['class']
                            more_prompt=more_prompt.replace(f"{cla}",f"<{self.args.concept}>")
                            if more_prompt:
                                # 先按换行符分割，只保留第一行内容
                                first_line = more_prompt.split('\n', 1)[0]  # 分割一次，取索引0的部分
                                
                                # 定义句尾标点，按优先级排序
                                end_punctuations = ('.', '!', '?')
                                # 遍历标点，找到第一个出现的句尾标点
                                for punc in end_punctuations:
                                    end_idx = first_line.find(punc)
                                    if end_idx != -1:
                                        # 截取到标点后一位（包含标点），作为第一句话
                                        first_sentence = first_line[:end_idx + 1]
                                        break
                                else:
                                    # 若没有找到句尾标点，则保留第一行的整个字符串
                                    first_sentence = first_line
                                # 更新more_prompt为第一句话
                                more_prompt = first_sentence
                            if '<|begin_of_box|>' in more_prompt:
                                more_prompt=more_prompt.replace('<|begin_of_box|>','')
                            if '<|end_of_box|>' in more_prompt:
                                more_prompt=more_prompt.replace('<|end_of_box|>','')
                            if 'empty' in more_prompt:
                                more_prompt=''
                            logging.info(f"After extraction:{more_prompt}")
                            condition+=self.info['info']+more_prompt
                            
                        else:
                            _,self.GT_IMAGE_PATH,SELECT_NUM=get_image_path(self.concept_train_path)
                            key=''
                            for tmp_keys in self.vqa.keys():
                                if tmp_keys in self.GT_IMAGE_PATH:
                                    key=tmp_keys
                                    break
                            
                            select_vqa=random.choice(self.vqa[key])
                            us_prompt=select_vqa['query']
                            us_answer=select_vqa['answer']
                            image_ori = Image.open(self.GT_IMAGE_PATH).convert("RGB")
                            # tranforming the image to the required resolution
                            image = image_transform(image_ori, resolution = self.config.dataset.params.resolution).to(self.args.device)
                            image = image.unsqueeze(0)

                            image_tokens_mmu = self.vq_model.get_code(image)
                            image_tokens = (image_tokens_mmu + len(self.uni_prompting.text_tokenizer)).long()
                            us_input = self.uni_prompting.text_tokenizer(['USER: ' + us_prompt + ' ASSISTANT:'])['input_ids']
                            us_input = torch.tensor(us_input).to(self.args.device)
                            us_input = torch.cat([
                                (torch.ones(us_input.shape[0], 1) * self.uni_prompting.sptids_dict['<|mmu|>']).to(self.args.device),
                                (torch.ones(us_input.shape[0], 1) * self.uni_prompting.sptids_dict['<|soi|>']).to(self.args.device),
                                image_tokens,
                                (torch.ones(us_input.shape[0], 1) * self.uni_prompting.sptids_dict['<|eoi|>']).to(self.args.device),
                                (torch.ones(us_input.shape[0], 1) * self.uni_prompting.sptids_dict['<|sot|>']).to(self.args.device),
                                us_input
                            ], dim=1).long()
                            us_mask = create_attention_mask_for_mmu(us_input.to(self.args.device),
                                                        eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>'])).to(self.target_dtype)
                            us_mask = us_mask.type(self.target_dtype)
                            us_toks_list = self.model.mmu_generate(
                                us_input, 
                                attention_mask=us_mask,
                                top_k=5,
                                eot_token=self.uni_prompting.sptids_dict['<|eot|>'],
                            )
                            us_toks_list= torch.stack(us_toks_list).squeeze()[None]
                            more_prompt = self.uni_prompting.text_tokenizer.batch_decode(us_toks_list, skip_special_tokens=True)[0].strip()
                            
                            logging.info(f"Question:{us_prompt}")
                            logging.info(f"GT:{us_answer}")
                            logging.info(f"Answer:{more_prompt}")
                            reward_text.append(calculate_bleu(us_answer,more_prompt))
                            logging.info(f"{self.local_rank}: Bleu score is {reward_text}")
                            if self.args.llm=='gemini':
                                more_prompt=extract(more_prompt,self.info["class"])
                            elif self.args.llm=='glm':
                                more_prompt=glm_extract(more_prompt,self.info["class"])
                            cla=self.info['class']
                            more_prompt=more_prompt.replace(f"{cla}",f"<{self.args.concept}>")
                            if more_prompt:
                                # 先按换行符分割，只保留第一行内容
                                first_line = more_prompt.split('\n', 1)[0]  # 分割一次，取索引0的部分
                                
                                # 定义句尾标点，按优先级排序
                                end_punctuations = ('.', '!', '?')
                                # 遍历标点，找到第一个出现的句尾标点
                                for punc in end_punctuations:
                                    end_idx = first_line.find(punc)
                                    if end_idx != -1:
                                        # 截取到标点后一位（包含标点），作为第一句话
                                        first_sentence = first_line[:end_idx + 1]
                                        break
                                else:
                                    # 若没有找到句尾标点，则保留第一行的整个字符串
                                    first_sentence = first_line
                                # 更新more_prompt为第一句话
                                more_prompt = first_sentence
                            if '<|begin_of_box|>' in more_prompt:
                                more_prompt=more_prompt.replace('<|begin_of_box|>','')
                            if '<|end_of_box|>' in more_prompt:
                                more_prompt=more_prompt.replace('<|end_of_box|>','')
                            if 'empty' in more_prompt:
                                more_prompt=''
                            logging.info(f"After extraction:{more_prompt}")
                            condition+=self.info['info']+more_prompt
                    
                    conditions = [condition] * batch_size_t2i


                    input_ids_infer, _ = self.uni_prompting((conditions, image_tokens_infer), 't2i_gen')   # [1, 387]
                    input_ids_infer = input_ids_infer.to(dtype=torch.long, device=self.args.device)
                    check_embedding_dtype(self.model,input_ids_infer[:1], self.target_dtype)
                    if self.config.guidance_scale > 0:
                        uncond_input_ids, _ = self.uni_prompting(([''] * batch_size_t2i, image_tokens_infer), 't2i_gen')
                    # [1, 387], == [PAD] * 126 + <|t2i|> + <|endoftext|> + <|endoftext|> + <|soi|> + [MASK] * 256 + <|eoi|> ## no prompt
                        attention_mask1 = create_attention_mask_predict_next(torch.cat([input_ids_infer, uncond_input_ids], dim=0),    # [2, 387]
                                                                            pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                                                                            soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                                                                            eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                                                                            rm_pad_in_image=True).to(self.target_dtype)
                        attention_mask1 = attention_mask1.type(self.target_dtype)
                    else:
                        attention_mask1 = create_attention_mask_predict_next(input_ids_infer,
                                                                            pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                                                                            soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                                                                            eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                                                                            rm_pad_in_image=True).to(self.target_dtype)
                        attention_mask1 = attention_mask1.type(self.target_dtype)
                        uncond_input_ids = None
                    if self.config.get("mask_schedule", None) is not None:
                        schedule = self.config.mask_schedule.schedule
                        arg = self.config.mask_schedule.get("params", {})
                        mask_schedule = get_mask_chedule(schedule, **arg)
                    else:
                        mask_schedule = get_mask_chedule(self.config.training.get("mask_schedule", "cosine"))
                    

                    # input_ids_infer=input_ids_infer.repeat_interleave(self.num_generations,dim=0)
                    # attention_mask1=attention_mask1.repeat_interleave(self.num_generations,dim=0)
                    # uncond_input_ids=uncond_input_ids.repeat_interleave(self.num_generations,dim=0)

                    
                    # with torch.no_grad():
                    if self.args.accelerate:
                        while True:
                            accounter+=1
                            gen_token_ids_ac,checkpoint = self._run_maskgit_rollout(
                                input_ids=input_ids_infer,
                                attention_mask=attention_mask1,
                                temperature=1.8,
                                timesteps=self.config.generation_timesteps,
                                noise_schedule=mask_schedule,
                                uncond_input_ids=uncond_input_ids,
                                checkpoint_step=10,
                                stop_at_checkpoint=True,
                            )
                            gen_token_ids_clamped_ac = torch.clamp(
                                gen_token_ids_ac, 
                                min=0, 
                                max=self.config.model.showo.codebook_size - 1
                            )
                            images_tensor_ac = self.vq_model.decode_code(gen_token_ids_clamped_ac)
                            images_tensor_ac = torch.clamp((images_tensor_ac + 1.0) / 2.0, min=0.0, max=1.0)
                            images_tensor_ac = images_tensor_ac * 255.0
                            images_tensor_ac = images_tensor_ac.permute(0, 2, 3, 1)
                            images_np_ac = images_tensor_ac.detach().cpu().numpy().astype(np.uint8)
                            gen_image_ac = Image.fromarray(images_np_ac[0])
                            # 创建保存目录（防止路径不存在）
                            save_path_ac=os.path.join(self.args.save_dir,'accelerate/',self.args.concept,f"gpu{self.local_rank}_num{accelerate_counter}.png")
                            mkdir(os.path.join(self.args.save_dir,'accelerate/',self.args.concept))
                            accelerate_counter += 1 
                            # 保存图像
                            gen_image_ac.save(save_path_ac)
                            logging.info(f"临时图像已经保存至{save_path_ac}")
                            clip_model.save_dir=save_path_ac
                            try:
                                simg,stext=clip_model.evaluate_concept(self.args.concept,'',0,prompt=remove_token(condition))
                                s=simg*0.65+stext*0.35
                            except:
                                simg=clip_model.evaluate_concept(self.args.concept,'',0)
                                s=simg
                            logging.info(f"临时图片评分{s}")
                            if s > self.threshold:
                                gen_token_ids,trajectory = self._run_maskgit_rollout(
                                    attention_mask=attention_mask1,
                                    noise_schedule=mask_schedule,
                                    temperature=checkpoint["temperature"],
                                    timesteps=self.config.generation_timesteps,
                                    state=checkpoint,
                                )
                                # os.remove(temp_save_path)  # 删除临时图像
                                del gen_token_ids_ac, gen_image_ac, images_tensor_ac, images_np_ac  # 释放CPU/GPU变量
                                del checkpoint  # 断点已使用，释放内存
                                torch.cuda.empty_cache()  # 清理GPU缓存
                                break

                            else:
                                if os.path.exists(save_path_ac):
                                    os.remove(save_path_ac)
                                    logging.info(f"已删除临时图像: {save_path_ac}")
                                
                                # 2. 清理显存/内存（关键：避免累积占用）
                                # 释放第9步相关张量
                                del gen_token_ids_ac, gen_image_ac, images_tensor_ac, images_np_ac
                                # 释放断点（未使用，直接删除）
                                del checkpoint
                                torch.cuda.empty_cache()
                                # 强制Python垃圾回收（清理未引用的内存对象）
                                import gc
                                gc.collect()
                                
                                logging.info(f"重新生成")
                                # 重新调用生成逻辑（复用当前循环的参数，无需额外传参）
                                continue  # 回到循环开头，重新执行9步生成

                    else:
                        gen_token_ids,trajectory = self._run_maskgit_rollout(
                            input_ids=input_ids_infer,
                            attention_mask=attention_mask1,
                            temperature=self.config.training.get("generation_temperature",1.0),
                            timesteps=self.config.generation_timesteps,
                            noise_schedule=mask_schedule,
                            uncond_input_ids=uncond_input_ids,
                        )
                    local_gen_token_ids = gen_token_ids#[self.local_rank::self.world_size]
                    global_id.append(local_gen_token_ids)
                    global_trajectories.append(trajectory)
                    
                    del input_ids_infer, attention_mask1, uncond_input_ids
                    del gen_token_ids
                    torch.cuda.empty_cache()
                gen_token_ids=torch.cat(global_id)
                gen_token_ids = torch.clamp(gen_token_ids, max=self.config.model.showo.codebook_size - 1, min=0)
                trajectory_step_count = len(global_trajectories[0]["records"]) if global_trajectories else 0
                old_per_token_logps = self._flatten_old_trajectory_logps(global_trajectories).to(self.args.device)
                per_token_logps = self._compute_trajectory_logps(
                    self.model, global_trajectories, requires_grad=True
                )
                ref_per_token_logps = self._compute_trajectory_logps(
                    self.ref_model.model, global_trajectories, requires_grad=False
                )
                del global_id, global_trajectories
                torch.cuda.empty_cache()
                images = self.vq_model.decode_code(gen_token_ids)
                images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                images *= 255.0
                images = images.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
                pil_images = [Image.fromarray(image) for image in images]
                del gen_token_ids
                for j in range(len(pil_images)):
                    gen_image = pil_images[j]
                    mkdir(os.path.join(save_dir,f'images/{self.args.concept}',f"Epoch{epoch}"))
                    tmp_image_path=os.path.join(save_dir,f'images/{self.args.concept}',f"Epoch{epoch}",f"batch_{batch_idx}_{self.world_size*j+self.local_rank}.png")
                    gen_image.save(tmp_image_path)
                    # gen_image.save(os.path.join("path/to/ref_image_dir", f"{counter}.png"))
                    # counter+=1
                    del gen_image
                torch.cuda.empty_cache()
                dist.barrier()
                self.model.config.mask_token_id=mti
                
                # load save logit
                # load_dir = "path/to/ref_image_dir"
                # save_logits=[]
                # for idx in range(100):
                #     l=torch.load(os.path.join(load_dir,f"{idx}.pt"))
                #     print(idx,l.max(),l.min())
                # return
                # for j in range(self.group):
                #     random_numbers = random.sample(range(100), 10)
                #     tmp_list=[]
                #     for idx in random_numbers:
                #         l=torch.load(os.path.join(load_dir,f"{idx}.pt"))
                #         l=torch.log(l)
                #         tmp_list.append(l)
                #     tmp_tensor=torch.stack(tmp_list).mean(dim=0)
                #     save_logits.append(tmp_tensor)
                # ref_per_token_logps=torch.cat(save_logits).to(self.args.device)
                # del save_logits
                # if not signal:
                #     self.save_logits=logits
                #     signal=True
                # ref_per_token_logps=self.save_logits
                    
                
                #calculate the rewards
                # rewards=torch.zeros(self.num_generations*batch_size_t2i).to(self.args.device)
                reward_list=[]
                
                # question='Please output a score ranging from 0 to 10 to represent the correctness of the following question:\n'+'Is '
                # question='How much do you think that '
                # for token in new_tokens_stage_1:
                #     question+=token
                # for token in new_tokens_stage_2:
                #     question+=token
                # question+='<adrien_brody> in the image?\n'
                # question+='Please use a score ranging from 0 to 10 to represent.\n'
                # # question+='Only a score is needed,please don\'t output yes or no.\n'
                image_path=os.path.join(save_dir,f'images/{self.args.concept}',f"Epoch{epoch}")
                path_list=[os.path.join(image_path,f"batch_{batch_idx}_{self.world_size*j+self.local_rank}.png") for j in range(self.group)]
                all_path_list=[os.path.join(image_path,f"batch_{batch_idx}_{j}.png") for j in range(self.group*self.num_generations)]



                #clip reward
                reward2_list=[]
                fr_list=[]
                fn_list=[]
                mix_list=[]
                clip_list=[]
                name_list=[]

                for path in path_list:
                    if 'man' in self.info['class'].lower():
                        sig1=0
                        sig2=0
                        try:
                            fr_score=face_recognition_score(path,self.args.data_root,self.args.concept)
                            sig1=1
                        except Exception as e:
                            print(f'---------------Face recognition False! 错误信息: {str(e)}')
                            pass
                        try:
                            fn_score=facenet_score(path,self.args.data_root,self.args.concept)
                            sig2=1
                        except Exception as e:
                            print(f'---------------FaceNet False! 错误信息: {str(e)}')
                            pass
                        if sig1==1 and sig2==1:
                            sim=0.7*fn_score+0.3*fr_score
                            mix_list.append(sim)
                            name_list.append('mix')
                        elif sig1==1:
                            sim=fr_score
                            fr_list.append(sim)
                            name_list.append('fr')
                        elif sig2==1:
                            sim=fn_score
                            fn_list.append(sim)
                            name_list.append('fn')
                        else:
                            clip_model.save_dir=path
                            sim=clip_model.evaluate_concept(self.args.concept,'',0)
                            if sim<0.5:
                                sim=0
                            else:
                                sim=2*(sim-0.5)
                            clip_list.append(sim)
                            name_list.append('clip')
                        reward2_list.append(sim)
                    else:
                        clip_model.save_dir=path
                        sim=clip_model.evaluate_concept(self.args.concept,'',0)
                        if sim<0.5:
                            sim=0
                        else:
                            sim=2*(sim-0.5)
                        reward2_list.append(sim)
                        name_list.append('clip')
                        clip_list.append(sim)
                mix_mean = sum(mix_list)/len(mix_list) if len(mix_list) > 0 else 0.8
                fr_mean = sum(fr_list)/len(fr_list) if len(fr_list) > 0 else 0.8
                fn_mean = sum(fn_list)/len(fn_list) if len(fn_list) > 0 else 0.8
                clip_mean = sum(clip_list)/len(clip_list) if len(clip_list) > 0 else 0.8

                # 遍历奖励列表，根据类型进行调整
                for idx in range(len(reward2_list)):
                    if name_list[idx] == 'mix':
                        reward2_list[idx] -= (mix_mean - 0.8)
                    elif name_list[idx] == 'fr':
                        reward2_list[idx] -= (fr_mean - 0.8)
                    elif name_list[idx] == 'fn':
                        reward2_list[idx] -= (fn_mean - 0.8)
                    elif name_list[idx] == 'clip':
                        reward2_list[idx] -= (clip_mean - 0.8)
                # print('score',sum(reward2_list)/len(reward2_list))
                print('similarity',path,sim)
                rewards2=torch.tensor(reward2_list).float().reshape(self.group).to(self.args.device,self.target_dtype)
                

                #gpt-4o reward
                # try:
                #     ref_path = "path/to/reference_image.png"
                #     img_path = [ref_path]
                #     # 拼接所有图像路径（all_path_list是所有图像的地址集合）
                #     for path in path_list:
                #         img_path.append(path)
                    
                #     # 调用GPT评分（仅local_rank=1执行）
                #     answer = glm_evaluate(img_path)
                #     answer = extract_and_clean_list(answer)
                #     print(f'gpt score:', answer)
                    
                #     # 转换为tensor并移到对应设备（与其他进程保持设备一致）
                #     reward3_list = [float(i/100) for i in answer]
                #     mean_re3=sum(reward3_list)/len(reward3_list)
                #     diff=mean_re3-0.8
                #     reward3_list=[i-diff for i in reward3_list]
                #     rewards3 = torch.tensor(reward3_list, dtype=torch.float32, device=self.args.device)
                # except:
                #     rewards3 = torch.tensor([0.8 for _ in range(self.group)], dtype=torch.float32, device=self.args.device)
                # rewards=rewards2*0.2+rewards3*0.8
                signal=1
                try:
                    # 1. 仅在local_rank=1时调用GPT评分（避免所有进程重复调用）
                    if self.local_rank == 1:
                        ref_path=os.path.join(self.args.data_root,'concept/train',self.args.concept,fm)
                        
                        img_path = [ref_path]
                        # 拼接所有图像路径（all_path_list是所有图像的地址集合）
                        for path in all_path_list:
                            img_path.append(path)
                        print(img_path)
                        for _ in range(5):
                            # 调用GPT评分（仅local_rank=1执行）
                            if self.args.llm=='gemini':
                                answer = evaluate(img_path)
                            elif self.args.llm=='glm':
                                answer =glm_evaluate(img_path)
                            answer = extract_and_clean_list(answer)
                            if len(answer)==len(all_path_list):
                                break
                        # if len(answer)>len(all_path_list):
                        #     answer=answer[:-len(all_path_list)]
                        logging.info(f'[local_rank={self.local_rank}] gpt score: {answer}')
                        
                        # 转换为tensor并移到对应设备（与其他进程保持设备一致）
                        reward3_list = [float(i/100) for i in answer]
                        mean_re3=sum(reward3_list)/len(reward3_list)
                        diff=mean_re3-0.8
                        reward3_list=[i-diff for i in reward3_list]
                        rewards3 = torch.tensor(reward3_list, dtype=self.target_dtype, device=self.args.device)
                        signal=0
                    else:
                        # 其他进程初始化空tensor（用于接收广播结果）
                        # 注意：需提前知道rewards3的形状，假设长度为self.group
                        rewards3 = torch.tensor([0.8 for _ in range(self.group*self.num_generations)], dtype=self.target_dtype, device=self.args.device)

                except Exception as e:
                    print(f'[local_rank={self.local_rank}] 评分计算失败，使用默认rewards2: {e}')
                signal_tensor = torch.tensor([signal], dtype=torch.int, device=self.args.device)
                dist.broadcast(signal_tensor, src=1)  # 同步 signal 到所有进程
                signal = signal_tensor.item()  # 所有进程的 signal 现在一致
                if signal==0:
                    # 2. 将local_rank=1的rewards3广播到所有进程
                    # 广播源为local_rank=1（确保该进程已计算出rewards3）
                    dist.broadcast(rewards3, src=1)  # src=1表示从local_rank=1广播到所有进程

                    # 3. 每个进程提取自己的reward（按local_rank索引，或按进程分片）
                    # 假设总进程数=group_size，每个进程取对应位置的元素
                    # 例如：总长度为self.group，每个进程取self.group // world_size个元素
                    world_size = dist.get_world_size()
                    total_length = self.group * self.num_generations

                    # 计算当前进程应处理的索引（交叉分片）
                    # 例如：总长度6，3进程时，进程0取0、3；进程1取1、4；进程2取2、5
                    indices = [i for i in range(total_length) if i % world_size == self.local_rank]

                    # 提取当前进程的reward
                    my_rewards3 = rewards3[indices].reshape(-1).to(self.target_dtype)

                    # 4. 与rewards2合成（当前进程仅用自己的reward片段）

                    rewards = rewards2 * 0.4 + my_rewards3 * 0.6
                else:
                    rewards=rewards2
                if self.args.semantic:
                    reward_text=torch.tensor(reward_text).float().reshape(self.group).to(self.args.device,self.target_dtype)
                    rewards=rewards*0.65+reward_text*0.35
                counter+=1
                # update_best_results(path_list,rewards.cpu().detach().numpy().tolist(),counter)

                
                all_rewards_list = [torch.zeros_like(rewards) for _ in range(self.world_size)]
                dist.all_gather(all_rewards_list, rewards)
                all_rewards = torch.cat(all_rewards_list, dim=0)
                rewards=all_rewards
                if self.local_rank==0:
                    for r in range(len(rewards)):
                        logging.info(f"img {r}, reward {rewards[r]}")
                        rr=int(r/2)+self.num_generations*(r % 2)
                        manage_top_images(all_path_list[rr],rewards[r].detach().cpu().item(),folder_path=os.path.join('./tmp_result/best_image/',self.args.concept))

                # if self.local_rank==0:
                #     print('max',logits.max())
                #     print('min',logits.min())
                # per_token_logps=torch.log(logits)
                # if self.local_rank==0:
                #     print(per_token_logps.requires_grad)
                # print(rewards,per_token_logps,ref_per_token_logps)


                #grpo
                mean_grouped_rewards = rewards.view(-1, self.num_generations*self.group).mean(dim=1)

                # Dr.GRPO uses strict group centering without variance normalization.
                mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations*self.group, dim=0)
                advantages = rewards - mean_grouped_rewards
                dist.broadcast(advantages, src=0)


                per_rank_samples = advantages.size(0) // self.world_size
                # 2. 计算当前进程数据的起始和结束索引
                start_idx = self.local_rank * per_rank_samples
                end_idx = start_idx + per_rank_samples
                # 3. 提取当前进程对应的advantages（与原始rewards索引一一对应）
                advantages = advantages[start_idx:end_idx]
                # print(advantages)

                self.model.train()
                if self.local_rank==0:
                    print('afmax',per_token_logps.max(),ref_per_token_logps.max())
                    print('afmin',per_token_logps.min(),ref_per_token_logps.min())
                old_per_token_logps = old_per_token_logps.to(per_token_logps.dtype)
                advantages = advantages.to(per_token_logps.dtype)
                log_ratio = per_token_logps - old_per_token_logps
                log_ratio = torch.clamp(log_ratio, -10, 10)
                ratio = torch.exp(log_ratio)
                unclipped_objective = ratio * advantages.unsqueeze(1)
                clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)  # 典型ε=0.2
                clipped_objective = clipped_ratio * advantages.unsqueeze(1)
                surrogate_objective = torch.min(unclipped_objective, clipped_objective)

                # 5. 计算KL散度正则项（加到损失中）
                ref_log_ratio = ref_per_token_logps.to(per_token_logps.dtype) - per_token_logps
                per_token_kl = torch.exp(ref_log_ratio) - ref_log_ratio - 1.0
                kl_regularization = self.beta * per_token_kl  # self.beta是KL权重（如0.1）

                # 6. Dr.GRPO removes token-length normalization and treats the trajectory holistically.
                per_token_loss = -surrogate_objective + kl_regularization

                if self.local_rank==0:
                    with torch.no_grad():
                        clipped_mask = (ratio < (1 - self.epsilon)) | (ratio > (1 + self.epsilon))
                        sample_objective = surrogate_objective.sum(dim=1)
                        sample_kl = per_token_kl.sum(dim=1)
                        sample_loss = per_token_loss.sum(dim=1)
                        logging.info(
                            "GRPO stats | "
                            f"traj_steps={trajectory_step_count} "
                            f"traj_tokens={per_token_logps.shape[1]} "
                            f"reward_mean={rewards.mean().item():.4f} reward_min={rewards.min().item():.4f} reward_max={rewards.max().item():.4f} "
                            f"adv_mean={advantages.mean().item():.4f} adv_std={advantages.std().item():.4f} adv_min={advantages.min().item():.4f} adv_max={advantages.max().item():.4f}"
                        )
                        logging.info(
                            "Policy stats | "
                            f"old_logp_mean={old_per_token_logps.mean().item():.4f} "
                            f"cur_logp_mean={per_token_logps.mean().item():.4f} "
                            f"ref_logp_mean={ref_per_token_logps.mean().item():.4f} "
                            f"log_ratio_mean={log_ratio.mean().item():.4f} log_ratio_std={log_ratio.std().item():.4f} "
                            f"ratio_mean={ratio.mean().item():.4f} ratio_min={ratio.min().item():.4f} ratio_max={ratio.max().item():.4f} "
                            f"clip_fraction={clipped_mask.float().mean().item():.4f}"
                        )
                        logging.info(
                            "Loss stats | "
                            f"surrogate_mean={surrogate_objective.mean().item():.4f} "
                            f"surrogate_sum_mean={sample_objective.mean().item():.4f} "
                            f"kl_mean={per_token_kl.mean().item():.4f} "
                            f"kl_sum_mean={sample_kl.mean().item():.4f} "
                            f"per_token_loss_mean={per_token_loss.mean().item():.4f} "
                            f"sample_loss_mean={sample_loss.mean().item():.4f}"
                        )
                completion_mask = torch.ones_like(per_token_loss, dtype=per_token_loss.dtype)
                loss = (per_token_loss * completion_mask).sum(dim=1).mean()

                #log
                self.optimizer.zero_grad()
                loss.backward()
                # ---------------------- 1. 梯度裁剪（防止梯度爆炸） ----------------------
                # 注意：DDP 梯度同步后，所有进程梯度已一致，裁剪操作需在所有进程执行（确保参数更新一致）
                max_norm = 1.0  # 梯度裁剪阈值（可根据模型调整，如 0.5/2.0）
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(),  # 对 DDP 模型的所有可训练参数裁剪
                    max_norm=max_norm,
                    norm_type=2  # L2 范数（常用选择，也可根据需求用 L1）
                )

                # ---------------------- 2. 打印梯度均值（监控梯度健康度） ----------------------
                # 仅在 local_rank=0 打印（避免所有进程重复输出，降低日志冗余）
                if self.local_rank == 0:
                    # 统计所有可训练参数的梯度均值（过滤无梯度的参数，如冻结的视觉 encoder）
                    grad_means = []
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            # 计算当前参数的梯度均值（绝对值，避免正负抵消）
                            grad_mean = param.grad.abs().mean().item()
                            grad_means.append(grad_mean)
                            # 可选：打印关键参数的梯度（如 LLM 嵌入层、注意力层，定位异常梯度）
                            if any(keyword in name for keyword in ["embed", "attention", "mlp"]):
                                logging.info(f"Gradient - {name}: mean={grad_mean:.6f}")
                    
                    # 打印全局梯度均值（反映整体梯度规模）
                    global_grad_mean = sum(grad_means) / len(grad_means) if grad_means else 0.0
                    logging.info(f"Epoch {epoch}, Batch {batch_idx}: Global gradient mean={global_grad_mean:.6f}")

                # ---------------------- 3. 验证各进程模型参数一致性（确保 DDP 同步正常） ----------------------
                # 每 10 个 batch 验证一次（避免频繁验证影响性能，可调整频率）
                if batch_idx % 1 == 0:
                    # 选择一个关键参数作为"校验锚点"（如 LLM 嵌入层权重，确保所有进程该参数一致）
                    anchor_param_name = "showo.get_input_embeddings().weight"  # 对应 Showo 模型的嵌入层
                    try:
                        # 1. 获取当前进程的锚点参数值（DDP 模型需通过 .module 访问原始参数）
                        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                            anchor_param = getattr(self.model.module, "showo").get_input_embeddings().weight
                        else:
                            anchor_param = getattr(self.model, "showo").get_input_embeddings().weight
                        
                        # 2. 计算当前进程锚点参数的均值（用于跨进程对比）
                        local_param_mean = anchor_param.mean().item()
                        local_param_mean_tensor = torch.tensor(local_param_mean, device=self.args.device)
                        
                        # 3. 收集所有进程的参数均值（通过 all_gather 同步）
                        all_param_means = [torch.tensor(0.0, device=self.args.device) for _ in range(self.world_size)]
                        torch.distributed.all_gather(all_param_means, local_param_mean_tensor)
                        
                        # 4. 验证所有进程的均值是否一致（误差阈值设为 1e-6，适应浮点精度）
                        param_means = [mean.item() for mean in all_param_means]
                        is_consistent = all(abs(mean - param_means[0]) < 1e-6 for mean in param_means)
                        
                        # 5. 打印验证结果（仅 local_rank=0 输出，避免冗余）
                        if self.local_rank == 0:
                            if is_consistent:
                                logging.info(f"Parameter consistency check PASSED: All ranks have same mean={param_means[0]:.6f}")
                            else:
                                # 若不一致，打印各进程均值（定位异常进程）
                                logging.error(f"Parameter consistency check FAILED: Ranks' means={param_means}")
                    except Exception as e:
                        # 捕获参数访问异常（如参数名错误），避免训练中断
                        if self.local_rank == 0:
                            logging.error(f"Parameter consistency check ERROR: {str(e)}")
                ref_model_test=self.ref_model.check_param_change()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                # if self.args.t2i_data and (epoch+1) % 10 == 0:
                #     print('epoch',epoch+1,'loss',loss.item())
                loss_tensor = torch.tensor([loss.item()], device=loss.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                world_size = dist.get_world_size()
                avg_loss = loss_tensor.item() / world_size
                if dist.get_rank() == 0:
                    loss_list.append(avg_loss)
                    logging.info(f"{batch_idx} loss: {avg_loss:.4f}")
                if self.args.accelerate:
                    accounter_tensor = torch.tensor([accounter], dtype=torch.int64, device=self.args.device)
                    dist.all_reduce(accounter_tensor, op=dist.ReduceOp.SUM)
                    rr=(self.num_generations*self.group)/(accounter_tensor.item())
                    rrr=self.accelerate_rate-rr
                    delta=0.12*rrr
                    tmp_threshold=self.threshold**(1+delta)
                    if tmp_threshold-self.threshold>0.005:
                        self.threshold+=0.005
                    if tmp_threshold-self.threshold<-0.005:
                        self.threshold-=0.005
                    logging.info(f"threshold turns into {self.threshold}")
                # for (n1, p1), (n2, p2) in zip(self.model.named_parameters(), self.ref_model.named_parameters()):
                #     if torch.allclose(p1, p2)!=True:
                #         print(f"parameter {n1} is not equal")
                import gc
                del rewards, per_token_logps, old_per_token_logps, ref_per_token_logps
                gc.collect()
                torch.cuda.empty_cache()
            loss_final.append(loss_list)
            logging.info(f"loss: {loss_final}")
            if epoch%self.args.interval_epochs==0 and epoch>0:
                save_distributed_model(self.model,self.optimizer,os.path.join(self.args.save_dir,'model_weights',self.args.concept),epoch=epoch)
