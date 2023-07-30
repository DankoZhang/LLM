#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File    :   model.py
brief   :   brief
Date    :   2023/07/26 08:26:46
Author  :   zhangce06
Contact :   zhangce06@baidu.com
"""

import torch
from torch import nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from args import args


class ChatGLMLoraModel(nn.Module):
    """
    定义模型
    """
    def __init__(self):
        """
        init
        """
        super(ChatGLMLoraModel, self).__init__()
        # 加载base模型
        model = AutoModel.from_pretrained(
            args.base_model,
            empty_init=False,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        # 构建lora配置
        target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["chatglm"]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        # 构建lora模型
        self.peft_model = get_peft_model(model, lora_config)
        # 设定lora相关配置
        self.peft_model.gradient_checkpointing_enable()
        self.peft_model.enable_input_require_grads()
        self.peft_model.config.use_cache = False

    def forward(self, input_ids, labels):
        """
        forward
        """
        output = self.peft_model(
            input_ids=input_ids,
            labels=labels,
            return_dict=True
        )

        loss = output["loss"]
        logits = output["logits"]
        predictions = logits.argmax(dim=-1)

        return loss, predictions

    def print_trainable_parameters(self):
        """
        打印参数
        """
        self.peft_model.print_trainable_parameters()

    def save_lora_model(self, lora_model):
        """
        保存模型
        """
        self.peft_model.save_pretrained(lora_model)
