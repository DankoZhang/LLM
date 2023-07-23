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
Date    :   2023/07/22 13:31:01
Author  :   zhangce06
Contact :   zhangce06@baidu.com
"""


import argparse


parser = argparse.ArgumentParser(description='ChatGLM LoRA')
parser.add_argument("--train_args_file", type=str, required=True)
parser.add_argument("--model_name_or_path", type=str, default="THUDM/chatglm-6b")
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--eval_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_input_length", type=int, default=1024)
parser.add_argument("--max_output_length", type=int, default=1024)
parser.add_argument("--lora_rank", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--int8", action="store_true")
parser.add_argument("--model_parallel", action="store_true")
parser.add_argument("--no_gradient_checkpointing", action="store_true")
args = parser.parse_args()