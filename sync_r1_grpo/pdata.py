import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

import os
import json
import copy
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from functools import partial
from llava.llava import conversation as conversation_lib

# conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
DATA_ROOT = "/home/daigaole/code/ex/dataset/unictokens_data"
    
def image_transform_dict(sample, resolution=256, load_cropped_image=True):
    # input image is PIL image
    image = sample["images"]
    image = image_transform(image, resolution=resolution, normalize=True, crop_augmentation=load_cropped_image)
    sample["images"] = image
    return sample

def image_transform(image, resolution=256, normalize=True, flip_augmentation=True, crop_augmentation=False):
    if crop_augmentation:
        resolution_to_resize = int(resolution * 1.1)
        image = transforms.Resize(resolution_to_resize, interpolation=transforms.InterpolationMode.BICUBIC)(image)
        image = transforms.RandomCrop(size=(resolution, resolution))(image)
    else:
        image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
        image = transforms.CenterCrop(size=(resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    # 有 05 的概率对图像进行水平翻转
    if flip_augmentation:
        if torch.rand(1) < 0.5:
            image = transforms.RandomHorizontalFlip()(image)

    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image


def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


def get_concept_all_training_images(concept, resolution=256):
    training_img_dir_path = os.path.join(DATA_ROOT, "concept/train", concept)
    train_images_file = os.path.join(DATA_ROOT, "concept/train", "train_images.json")
    with open(train_images_file, "r") as f:
        train_images = json.load(f)
    img_paths = [os.path.join(training_img_dir_path, img) for img in train_images[concept]]
    
    # transform images
    for i in range(len(img_paths)):
        image = Image.open(img_paths[i]).convert("RGB")
        image = image_transform(image, resolution)
        img_paths[i] = image
    return img_paths


def get_concept_all_training_images_path(concept):
    training_img_dir_path = os.path.join(DATA_ROOT, "concept/train", concept)
    train_images_file = os.path.join(DATA_ROOT, "concept/train", "train_images.json")
    with open(train_images_file, "r") as f:
        train_images = json.load(f)
    img_paths = [os.path.join(training_img_dir_path, img) for img in train_images[concept]]
    return img_paths


def get_concept_info(concept):
    info_path = os.path.join(DATA_ROOT, "concept/train", concept, "info.json")
    with open(info_path, "r") as f:
        info = json.load(f)
    class_conept = info["class"]
    concept_info = info["info"]
    concept_info = concept_info.replace(f"<{concept}> is ", "").strip()
    conversation_type = info["conversation_type"]
    return class_conept, concept_info, conversation_type


def preprocess_multimodal(sources):
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

                # Customized operation, get rid of <image> special token. Edited by Zechen
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "")
                sentence['value'] = sentence['value'].strip()

    return sources


def preprocess_v0(
        sources,
        tokenizer,
        system_personalized_prompt,
):
    # Let's assume has_image is false, since we will process the image token separately
    has_image = False

    # Adapted from llava-phi/mipha/train/train.py
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conversation_str = str(conv.get_prompt()).strip()
        conversations.append(conversation_str)

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "                   # ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):        # loop for instances in a batch
        # total_len = int(target.ne(tokenizer.pad_token_id).sum()) + conversation.count(conv.sep2)  # in phi-2, pad_token_id == eos_token_id
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)              # handle multi-round conversation regarding one image
        cur_len = 0                                         # no bos token in phi, so set the initial len to 0
        if cur_len > 0:
            target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids) + 1  # +1 for <|endoftext|>
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(conversation)
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    input_ids_system = tokenizer(
        [system_personalized_prompt for _ in range(len(conversations))],
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    return dict(
        input_ids=input_ids,
        labels=targets,
        input_ids_system=input_ids_system
    )
    

class PersonalizedMMUDataset(Dataset):

    def __init__(self,
                 tokenizer,
                 data_root,
                 concept,
                 image_size,
                 new_tokens,
                 stage,
                 nums_new_token_i_stage_1=16,
                 nums_new_token_i_stage_2=0,
                 nums_new_token_i_stage_3=0
                 ):
        super(PersonalizedMMUDataset, self).__init__()

        self.tokenizer = tokenizer
        self.image_size = image_size
        self.stage = stage
        
        data_file_path = os.path.join(data_root, f"showo_training_data_stage_{self.stage}/{concept}.json")

        with open(data_file_path, 'r') as f:
            data = json.load(f)
        self.list_data_dict = []
        for item in data:
            if 'image' in item.keys():
                self.list_data_dict.append(item)
        
        if self.stage == 1:
            if new_tokens:
                self.system_personalized_prompt = f"<{concept}> is "
                for i in range(nums_new_token_i_stage_1):
                    self.system_personalized_prompt += f"<token_{i}>"
                    if i == nums_new_token_i_stage_1 - 1:
                        self.system_personalized_prompt += "."
            else:
                self.system_personalized_prompt = ""
        else:
            if new_tokens:
                self.system_personalized_prompt = f"<{concept}> is "
                for i in range(nums_new_token_i_stage_1):
                    self.system_personalized_prompt += f"<token_{i}>"
                for i in range(nums_new_token_i_stage_1 + nums_new_token_i_stage_2,
                               nums_new_token_i_stage_1 + nums_new_token_i_stage_2 + nums_new_token_i_stage_3):
                    self.system_personalized_prompt += f"<token_{i}>"
                    if i == nums_new_token_i_stage_1 + nums_new_token_i_stage_2 + nums_new_token_i_stage_3 - 1:
                        self.system_personalized_prompt += "."
            else:
                self.system_personalized_prompt = ""

        for item in self.list_data_dict:
            item['conversations'][0]["value"] =  self.system_personalized_prompt + item['conversations'][0]["value"]
        
        print("Formatting llava instruction data")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        assert 'image' in sources[0]
        image_file = self.list_data_dict[i]['image']
        try:
            image = Image.open(os.path.join(image_file)).convert('RGB')
            image = image_transform(image, self.image_size)
        except:
            print("Read image error. Use dummy data.")
            crop_size = self.image_size
            image = torch.zeros(3, crop_size, crop_size)

        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))

        data_dict = preprocess_v0(sources, self.tokenizer, self.system_personalized_prompt)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             input_ids_system=data_dict["input_ids_system"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.image_size
            data_dict['image'] = torch.zeros(3, crop_size, crop_size)

        return data_dict


def mmu_collate_fn(
        instances,
        tokenizer=None,
        max_length=128,
):
    # 提取 input_ids, labels 和 input_ids_system
    input_ids, labels, input_ids_system = tuple([instance[key] for instance in instances]
                                                for key in ("input_ids", "labels", "input_ids_system"))
    
    # 填充 input_ids 和 labels 到 max_length
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)  # 使用 pad_token_id 填充
    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=IGNORE_INDEX)  # 使用 IGNORE_INDEX 填充 labels

    # 确保填充到 max_length
    # 如果 input_ids 长度小于 max_length，则填充
    if input_ids.shape[1] < max_length:
        pad_tube = torch.ones(
            size=(input_ids.shape[0], max_length - input_ids.shape[1]),
            dtype=input_ids.dtype) * tokenizer.pad_token_id
        input_ids = torch.cat([input_ids, pad_tube], dim=1)
    
    # 同样处理 labels
    if labels.shape[1] < max_length:
        pad_tube = torch.ones(
            size=(labels.shape[0], max_length - labels.shape[1]),
            dtype=labels.dtype) * IGNORE_INDEX
        labels = torch.cat([labels, pad_tube], dim=1)
    
    # 将 input_ids_system 堆叠成一个 tensor
    input_ids_system = torch.stack(input_ids_system, dim=0)

    # 计算填充后的输入长度，确保它们不会超过 max_length
    # min_max_len = min(
    #     max_length - input_ids_system.shape[-1],
    #     tokenizer.model_max_length - input_ids_system.shape[-1],
    # )

    # input_ids = input_ids[:, :max_length]
    # labels = labels[:, :max_length]

    # 创建最终的 batch 字典
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),  # attention mask 用于指示哪些部分是填充
        input_ids_system=input_ids_system,
    )

    # 如果每个样本包含图像数据，进行堆叠
    if 'image' in instances[0]:
        images = [instance['image'] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images)  # 堆叠图像 tensor（bsz, 3, 256, 256）
        else:
            batch['images'] = images  # 如果图像尺寸不一致，则保留为列表

    return batch


def get_personalized_mmu_dataloader(
        data_root,
        concept,
        tokenizer,
        image_size,
        batch_size,
        num_workers,
        max_length,
        new_tokens,
        stage = 1,
        nums_new_token_i_stage_1: int = 16,
        nums_new_token_i_stage_2: int = 0,
        nums_new_token_i_stage_3: int = 0,
):
    train_dataset = PersonalizedMMUDataset(
        tokenizer,
        data_root,
        concept,
        image_size,
        new_tokens,
        stage,
        nums_new_token_i_stage_1=nums_new_token_i_stage_1,
        nums_new_token_i_stage_2=nums_new_token_i_stage_2,
        nums_new_token_i_stage_3=nums_new_token_i_stage_3,
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(
            mmu_collate_fn,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
    )

    return dataloader
    

class PersonalizedT2IDataset(Dataset):
    def __init__(self, 
                 data_root: str, 
                 concept: str, 
                 tokenizer, 
                 image_size: int = 256,
                 max_text_len: int = 128,
                 nums_new_token_i: int = 16,
                 more_data=False,
                 inited=False,
                 system_prompt_t2i=False,
                 caption_training=False,
                 load_caption_training_path=None,
                 load_cropped_image=True,
                 less_t2i_data=False,
                 ):
        self.data_root = data_root
        self.concept = concept
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.more_data = more_data
        self.image_size = image_size
        self.inited = inited
        self.system_prompt_t2i = system_prompt_t2i
        self.caption_training = caption_training
        self.load_caption_training_path = load_caption_training_path
        self.load_cropped_image = load_cropped_image
        self.less_t2i_data = less_t2i_data
        
        self.condition_texts = []

        if self.load_cropped_image:
            training_img_dir_path = os.path.join(self.data_root, "concept/train", concept, "cropped")
        else:
            training_img_dir_path = os.path.join(self.data_root, "concept/train", concept)
        info_path = os.path.join(self.data_root, "concept/train", concept, "info.json")
        with open(info_path, "r") as f:
            info = json.load(f)
        self.concept_class = info["class"]
        
        if self.caption_training:
            with open(self.load_caption_training_path, "r") as f:
                info_caption = json.load(f)
        
        # self.names = []
        tokens_str = "".join([f"<token_{i}>" for i in range(nums_new_token_i)])
        self.img_paths = []
        for img in sorted(os.listdir(training_img_dir_path)):
            if img.lower().endswith(('png', 'jpg', 'jpeg')) and "mask" not in img:
                if self.less_t2i_data and len(self.img_paths) >= 4:
                    break
                    
                img_path = os.path.join(training_img_dir_path, img)
                self.img_paths.append(img_path)
                if self.caption_training:
                    if info_caption.get(img) is not None:
                        self.condition_texts.append(info_caption[img])
                    else:
                        raise ValueError(f"Image {img} does not have a corresponding caption in the info.json file.")
                else:
                    if self.inited:
                        self.condition_texts.append(f"A photo of {tokens_str} <{self.concept}>.")
                    else:
                        self.condition_texts.append(f"A photo of {tokens_str} <{self.concept}> {self.concept_class}.")
                
        assert len(self.condition_texts) == len(self.img_paths), f"{len(self.condition_texts)} != {len(self.img_paths)}"
        
        if self.more_data:
            same_class_img_dir_path = os.path.join(self.data_root, "concept/train", concept, "negative")
            for img in os.listdir(same_class_img_dir_path):
                if img.lower().endswith(('png', 'jpg', 'jpeg')) and "mask" not in img:
                    img_path = os.path.join(same_class_img_dir_path, img)
                    self.img_paths.append(img_path)
                    self.condition_texts.append(f"A photo of {self.concept_class}.")
                    
        if self.system_prompt_t2i:   
            self.system_personalized_prompt = f"<{concept}> is "
            for i in range(nums_new_token_i):
                self.system_personalized_prompt += f"<token_{i}>"
                if i == nums_new_token_i - 1:
                    self.system_personalized_prompt += ".\n"
        else:
            self.system_personalized_prompt = ""

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # 读取图像
        img_path = self.img_paths[idx]
        condition_text = self.condition_texts[idx]
        img = Image.open(img_path).convert("RGB")
        # 定义条件文本
        item = {
            "conditions": self.system_personalized_prompt + condition_text,
            "images": img,  
        }
        # 对图像进行预处理（resize、ToTensor 等）
        item = image_transform_dict(item, self.image_size, self.load_cropped_image)
        # # 使用 tokenizer 对条件文本进行编码
        # tokens = self.tokenizer(
        #     condition_text,
        #     max_length=self.max_text_len,
        #     truncation=True,
        #     padding="max_length"
        # )
        # # tokens["input_ids"] 为列表，此处转成 tensor
        # item["input_ids"] = torch.tensor(tokens["input_ids"])
        return item


def get_personalized_t2i_dataloader(data_root, 
                                    concept, 
                                    tokenizer, 
                                    image_size, 
                                    batch_size, 
                                    num_workers, 
                                    max_length=128, 
                                    nums_new_token_i=16, 
                                    more_data=False, 
                                    inited=False,
                                    system_prompt_t2i=False,
                                    caption_training=False,
                                    load_caption_training_path=None,
                                    less_t2i_data=False,
                                    load_cropped_image=True):
    """
    :param data_root: 数据根目录
    :param concept: 概念名称
    :param tokenizer: 文本分词器
    :param batch_size: 每个批次样本数
    :param num_workers: DataLoader 使用的进程数
    :param max_text_len: 文本最大长度
    :return: 返回构建好的 DataLoader 对象
    """
    dataset = PersonalizedT2IDataset(data_root, concept, 
                                     tokenizer, image_size, max_length, 
                                     nums_new_token_i, more_data, inited, 
                                     system_prompt_t2i, caption_training, 
                                     load_caption_training_path, load_cropped_image,
                                     less_t2i_data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        # collate_fn=t2i_collate_fn
    )
    return dataloader
