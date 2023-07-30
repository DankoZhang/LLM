#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File    :   gpu_compute.py
brief   :   brief
Date    :   2023/07/29 13:26:06
Author  :   zhangce06
Contact :   zhangce06@baidu.com
"""



from transformers import AutoModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
 
# 估算训练需要使用的资源

# specify the model you want to train on your device
model = AutoModel.from_pretrained("../../chatglm-6b-model/", trust_remote_code=True) 

# estimate the memory cost (both CPU and GPU)
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)
estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)