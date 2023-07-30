#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File    :   args.py
brief   :   brief
Date    :   2023/07/26 12:30:06
Author  :   zhangce06
Contact :   zhangce06@baidu.com
"""


import argparse


parser = argparse.ArgumentParser(description='ChatGLM LoRA DP')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument("--eval_every", type=int, default=500)
parser.add_argument("--checkpoint_every", type=int, default=500)
parser.add_argument("--train_steps", type=int, default=1500)
parser.add_argument("--warmup_steps", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--accu_steps", type=int, default=64)
parser.add_argument("--max_input_len", type=int, default=96)
parser.add_argument("--max_output_len", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=float, default=64)
parser.add_argument("--lora_dropout", type=float, default=0.1)
parser.add_argument("--val_set_size", type=int, default=2000)
parser.add_argument("--data_path", type=str, default="./data/trans_chinese_alpaca_data.json")
parser.add_argument("--base_model", type=str, default="../../chatglm-6b-model/")
parser.add_argument("--lora_model", type=str, default="./output/lora_model")

args = parser.parse_args()