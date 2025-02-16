#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   trainer_train.py
@Time    :   2025/02/14 21:16:45
@Author  :   zhangce
@Desc    :   trainer_train.py
"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

# 调试
import debugpy
import jsonlines
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from util import print_trainable_parameters

# 启动debug
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="OpenGVLab/InternVL2_5-4B")
    train_type: Optional[str] = field(
        default="full",
        metadata={
            "help": """
            1. use_lora:使用lora训练,
            2. none:全量参数训练;
            3. freeze_vision:只冻结vision_tower进行训练
            """
        }
    )
    grad_checkpoint: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={
            "help": "path to the train data."
        }
    )
    split_ratio: float = field(default=0.01)


@dataclass
class PeftArguments:
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size, ((i //
                                                                        (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

# 图片处理阶段
def transform_image(image, input_size=448, max_num=12):
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def findall(token_list: List[int], sub_token_list: Union[int, List[int]]) -> List[int]:
    """Find the index of a token in the token_list."""
    if isinstance(sub_token_list, int):
        sub_token_list = [sub_token_list]
    res = []
    idx = -1
    try:
        while True:
            idx = token_list.index(sub_token_list[0], idx + 1)
            if len(sub_token_list) == 1 or sub_token_list == token_list[idx:idx + len(sub_token_list)]:
                res.append(idx)
    except ValueError:
        pass
    return res

# 切分训练验证集
def train_test_split(dataset_dir, split_ratio, seed):
    random.seed(seed)
    train_dataset = []
    valid_dataset = []
    total_data = []
    with jsonlines.open(dataset_dir, "r") as fr:
        for line in fr:
            total_data.append(line)
    total_data_len = len(total_data)
    valid_data_len = int(total_data_len * split_ratio)
    train_data_len = total_data_len - valid_data_len
    total_dataset_index = list(range(total_data_len))
    valid_dataset_index = set(random.sample(total_dataset_index, valid_data_len))
    for item_index in total_dataset_index:
        if item_index not in valid_dataset_index:
            train_dataset.append(total_data[item_index])
        else:
            valid_dataset.append(total_data[item_index])
    assert train_data_len + valid_data_len == len(train_dataset) + len(valid_dataset)
    with jsonlines.open(dataset_dir + ".train", "w") as fw:
        for item in train_dataset:
            fw.write(item)
    with jsonlines.open(dataset_dir + ".valid", "w") as fw:
        for item in valid_dataset:
            fw.write(item)
    return len(train_dataset), len(valid_dataset)

# 自定义数据集
class CustomDataset(Dataset):
    """
    构建数据集
    """
    def __init__(self, dataset_dir, model_name_or_path):
        super().__init__()
        self.chat_data = self.build_dataset(dataset_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.num_image_token = 256
    
    def build_dataset(self, data_dir):
        """
        构建数据集
        """
        # data_dir = Path(data_dir)
        # chat_file = data_dir.joinpath("final_merge_data.jsonl")
        with jsonlines.open(data_dir, "r") as fr:
            chat_data = [line for line in fr]
        return chat_data
    
    def _encode(self, q_text, a_text):
        """
        编码数据
        """
        q_text_list = q_text.split("<image>")
        q_text_list[0] = q_text_list[0] + "<img>"
        q_text_list[-1] = "</img>" + q_text_list[-1]
        q_text_list.insert(1, -100)
        q_input_ids = []
        for q_text_item in q_text_list:
            if isinstance(q_text_item, str):
                q_input_ids += self.tokenizer(q_text_item)["input_ids"]
            else:
                q_input_ids += [q_text_item]
        a_input_ids = []
        for a_text_item in [a_text, "<|im_end|>"]:
            a_input_ids += self.tokenizer(a_text_item)["input_ids"]
        input_ids = q_input_ids + a_input_ids

        labels = [-100] * len(q_input_ids) + a_input_ids
        result_dict = {
            "input_ids": input_ids,
            "labels": labels
        }
        return result_dict
    
    def _process_one_piece(self, feature):
        """
        对单条文本进行处理
        """
        q_text = feature[0]
        a_text = feature[1]
        image = feature[2]
        image = Image.open(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        images = [image]
        encoded = self._encode(q_text, a_text)
        if len(encoded) == 0:
            return encoded
        input_ids = encoded['input_ids']
        idx_list = findall(input_ids, -100)
        labels = encoded['labels']
        if images:
            input_size = 448
            max_num = 12
            pixel_values = [transform_image(image, input_size, max_num) for image in images]
            num_patches = [pv.shape[0] for pv in pixel_values]
            # pixel_values = torch.cat(pixel_values).to(self.config.torch_dtype)
            pixel_values = torch.cat(pixel_values).to(torch.bfloat16)
        else:
            pixel_values = None
            num_patches = []
        assert len(num_patches) == len(
            idx_list), f'len(num_patches): {len(num_patches)}, len(idx_list): {len(idx_list)}'
        added_tokens_len = 0
        for idx, num_patch in zip(idx_list, num_patches):
            img_tokens: List[int] = self.tokenizer.encode(
                '<IMG_CONTEXT>', add_special_tokens=False) * self.num_image_token * num_patch
            input_ids = input_ids[:idx + added_tokens_len] + img_tokens + input_ids[idx + added_tokens_len + 1:]
            if labels is not None:
                labels = labels[:idx + added_tokens_len] + [-100] * len(img_tokens) + labels[idx + added_tokens_len
                                                                                            + 1:]
            added_tokens_len += len(img_tokens) - 1
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['pixel_values'] = pixel_values
        encoded['image_flags'] = torch.tensor([1] * pixel_values.size(0), dtype=torch.long)
        return encoded
    
    def __len__(self):
        return len(self.chat_data)
    
    def __getitem__(self, index):
        cur_data = self.chat_data[index]

        messages = cur_data["messages"][:2]
        human_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        gpt_output = cur_data["messages"][2]["content"]
        image_path = cur_data["images"][0]

        encode_res = self._process_one_piece([human_input, gpt_output, image_path])

        return encode_res
    

class CustomCollator(object):
    """
    自定义数据处理器
    """
    def __init__(self, tokenizer, IGNORE_INDEX=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.IGNORE_INDEX = IGNORE_INDEX

    def _pad_sequence(self, sequences: List[torch.Tensor], padding_value: float = 0.) -> torch.Tensor:
        """Pad sequence by some side

        Args:
            sequences: The input sequences in tensor.
            padding_value: The padding value

        Returns:
            A tensor after padding
        """
        padding_side = 'right'
        padding_right = padding_side == 'right'
        if padding_right:
            return pad_sequence(sequences, batch_first=True, padding_value=padding_value)

        max_len = max([s.shape[0] for s in sequences])

        padded_sequences = []
        for seq in sequences:
            pad_length = max_len - seq.shape[0]
            pad_tuple = [0] * ((seq.dim() - 1) * 2) + [pad_length, 0]
            padded_seq = F.pad(seq, tuple(pad_tuple), 'constant', padding_value)
            padded_sequences.append(padded_seq)

        return torch.stack(padded_sequences)

    def __call__(self, features):
        input_ids_list = []
        labels_list = []
        pixel_values_list = []
        image_flags_list = []
        for feature in features:
            input_ids_list.append(torch.tensor(feature["input_ids"]))
            labels_list.append(torch.tensor(feature["labels"]))
            pixel_values_list.append(feature["pixel_values"])
            image_flags_list.append(feature["image_flags"])
        input_ids = self._pad_sequence(input_ids_list, self.tokenizer.pad_token_id)
        labels = self._pad_sequence(labels_list, self.IGNORE_INDEX)
        pixel_values = torch.concat(pixel_values_list, dim=0)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids==self.tokenizer.pad_token_id] = 0
        image_flags = torch.concat(image_flags_list)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "image_flags": image_flags
        }


IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


def load_model_tokenizer(model_args: ModelArguments, peft_args: PeftArguments, torch_dtype=torch.bfloat16):
    """
    加载模型和tokenizer
    """
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path, 
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code = True
        )
    if model_args.grad_checkpoint:
        model.language_model.config.use_cache = False
        model.vision_model.gradient_checkpointing = True
        model.vision_model.encoder.gradient_checkpointing = True
        model.language_model._set_gradient_checkpointing()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    if model_args.train_type == "use_lora":
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=peft_args.lora_rank,
            lora_alpha=peft_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=peft_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=[],
        )
        model = get_peft_model(model, config)

    elif model_args.train_type == "full":
        pass
    elif model_args.train_type == "freeze_vision":

        for param in model.vision_tower.parameters():
            param.requires_grad = False
    print_trainable_parameters(model)

    return model, tokenizer


def load_dataset_collator(tokenizer, 
                          data_args: DataArguments, 
                          model_args: ModelArguments,
                          train_args: TrainingArguments):
    """
    加载数据集和collator
    """
    train_test_split(data_args.data_path, data_args.split_ratio, train_args.seed)
    train_customer_dataset = CustomDataset(dataset_dir=data_args.data_path + ".train", model_name_or_path=model_args.model_name_or_path)
    valid_customer_dataset = CustomDataset(dataset_dir=data_args.data_path + ".valid", model_name_or_path=model_args.model_name_or_path)

    customer_collator = CustomCollator(tokenizer, -100)
    
    return train_customer_dataset, valid_customer_dataset, customer_collator


def train():
    """
    开始训练
    """
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, PeftArguments)
    )
    model_args, data_args, training_args, peft_args = parser.parse_args_into_dataclasses()
    if training_args.bf16:
        torch_dtype = torch.bfloat16
    if training_args.fp16:
        torch_dtype = torch.float16
    model, tokenizer = load_model_tokenizer(model_args, peft_args, torch_dtype)
    train_dataset, valid_dataset, data_collator = load_dataset_collator(tokenizer, data_args, model_args, training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

