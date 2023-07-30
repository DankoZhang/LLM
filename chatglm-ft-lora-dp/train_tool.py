from email import header


#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File    :   data_helper.py
brief   :   brief
Date    :   2023/07/26 13:14:57
Author  :   zhangce06
Contact :   zhangce06@baidu.com
"""


import json
import random

import torch
from torch.utils.data import Dataset, DataLoader

from args import args
from utils import logger


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    batch = list(zip(*batch))
    input_ids = torch.tensor(batch[0], dtype=torch.long)
    labels = torch.tensor(batch[1], dtype=torch.long)

    return input_ids, labels


class DataHelper:
    """
    数据初始处理类
    """
    def __init__(self):
        """
        init
        """
        self.data_path = args.data_path
        self.val_set_size = args.val_set_size

    def load_data(self):
        """
        加载数据
        """
        with open(self.data_path, "r") as fr:
            data = json.load(fr)

        return data

    def gen_data(self):
        """
        生成训练、验证数据
        """
        data = self.load_data()
        random.shuffle(data)

        train_data = data[self.val_set_size:]
        valid_data = data[:self.val_set_size]
        return train_data, valid_data


class ChatGLMDataset(Dataset):
    """
    数据类
    """
    def __init__(self, tokenizer, data):
        """
        init
        """
        self.tokenizer = tokenizer
        self.data = data
        self.max_input_len = args.max_input_len
        self.max_output_len = args.max_output_len
        self.label_pad_token_id = -100  # pytorch 中label默认为-100时不会计算loss

    def generate_prompt(self, data_point):
        """
        构建输入、输出
        """
        # sorry about the formatting disaster gotta move fast
        if data_point["input"]:
            prompt = "{}: {}".format(data_point["instruction"], data_point["input"])
        else:
            prompt = "{}".format(data_point["instruction"])

        output = data_point["output"]
        return prompt, output

    def tokenize(self, prompt, output):
        """
        tokenize
        """
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
        b_ids = self.tokenizer.encode(text=output, add_special_tokens=False)

        if len(a_ids) > self.max_input_len - 1:
            a_ids = a_ids[: self.max_input_len - 1]

        if len(b_ids) > self.max_output_len - 2:
            b_ids = b_ids[: self.max_output_len - 2]

        input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

        context_length = input_ids.index(self.tokenizer.bos_token_id)
        mask_position = context_length - 1
        labels = [self.label_pad_token_id] * context_length + input_ids[mask_position + 1:]

        pad_len = (self.max_input_len + self.max_output_len) - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.label_pad_token_id] * pad_len

        return input_ids, labels

    def generate_and_tokenize_prompt(self, data_point):
        """
        generate and tokenize prompt
        """
        prompt, output = self.generate_prompt(data_point)
        input_ids, labels = self.tokenize(prompt, output)
        return input_ids, labels

    def __len__(self):
        """
        len
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        getitem
        """
        data_point = self.data[index]
        input_ids, labels = self.generate_and_tokenize_prompt(data_point)

        return (input_ids, labels)


class ChatGLMDataLoader:
    """
    数据加载类
    """

    def __init__(self, tokenizer, batch_size):
        """
        init
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def get_data_loader(self):
        """
        加载数据集
        """
        data_obj = DataHelper()
        train_data, valid_data = data_obj.gen_data()

        logger.info("train data size: {}".format(len(train_data)))
        logger.info("valid data size: {}".format(len(valid_data)))

        train_data_set = ChatGLMDataset(self.tokenizer, train_data)
        valid_data_set = ChatGLMDataset(self.tokenizer, valid_data)

        train_data_loader = DataLoader(
            train_data_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn)
        valid_data_loader = DataLoader(
            valid_data_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn)

        return train_data_loader, valid_data_loader