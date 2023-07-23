#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File    :   train_tool.py
brief   :   brief
Date    :   2023/07/22 13:28:02
Author  :   zhangce06
Contact :   zhangce06@baidu.com
"""


import torch
from transformers import Trainer
import os

class DataCollator:
    """
    数据处理函数
    """
    def __init__(self, pad_token_id):
        """
        init
        """
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        """
        call
        """
        lengths = [len(feature["input_ids"]) for feature in batch]
        longest = max(lengths)
        input_ids, labels = [], []
        for length, feature in sorted(zip(lengths, batch), key=lambda x: -x[0]):
            pad_len = longest - length
            ids = feature["input_ids"] + [self.pad_token_id] * pad_len
            label = feature["labels"] + [-100] * pad_len
            input_ids.append(torch.LongTensor(ids))
            labels.append(torch.LongTensor(label))

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        return {"input_ids": input_ids, "labels": labels}


class ModifiedTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        self.model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))