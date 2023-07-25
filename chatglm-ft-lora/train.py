#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File    :   train.py
brief   :   brief
Date    :   2023/07/22 13:28:33
Author  :   zhangce06
Contact :   zhangce06@baidu.com
"""


import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    HfArgumentParser,
    set_seed,
    TrainingArguments,
    PreTrainedModel
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
    prepare_model_for_int8_training
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from args import args
from train_tool import DataCollator, ModifiedTrainer


        
def train(args):
    """
    训练主函数
    """
    # 加载训练参数
    parser = HfArgumentParser(TrainingArguments)
    training_args, = parser.parse_json_file(json_file=args.train_args_file)
    
    # 模型并行设置策略（流水线并行），auto表示自动分配模型层至不同卡。
    # 启动命令: CUDA_VISIBLE_DEVICES=1,2 python train_lora.py --int8: 模型并行+八位加载，device="auto", 自动分配模型层至1卡和2卡
    # 启动命令: CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train_lora.py: 数据并行, 不能使用八位加载 
    device_map = "auto"

    # 设置随机数种子
    set_seed(args.seed)
    training_args.seed = args.seed
    
    # 下载model和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if args.int8:
        # 使用八位加载模型时，需使用模型并行，trainer的DDP模式不支持八位模型(trainer.py line1457)。device_map将模型不同层映射至不同显卡
        model = AutoModel.from_pretrained(args.model_name_or_path, load_in_8bit=True,
            device_map=device_map, trust_remote_code=True)
        # 可打印模型并行时，模型各层所在的显卡信息
        # print(f"hf_device_map: {model.hf_device_map}")
    else:
        # 不使用八位加载模型时，可进行数据并行
        model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        # 不需要使用model.half().cuda(), trainer会将模型加载至显卡上(trainer.py line510), 推理时就需要了
        model = model.half()

    # 此处个人认为没什么用处, 因为上述line61如果启用了模型并行, 此处不启用也没关系(trainer.py line391)
    # 但是反过来不行，如果此处启用了模型并行, 那么line61也必须开启, 否则trainer会将模型加载至cpu
    if args.model_parallel:
        model.is_parallelizable = True
        model.model_parallel = True

    # 定义LoRA配置
    target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["chatglm"]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM
    )
    
    # 适配LoRA和八位模型
    if args.int8:
        # 作用参见源码
        model = prepare_model_for_int8_training(model)
    else:
        # 向前传播的过程中使用torch.no_grad()不去存储中间激活值，降低动态显存的占用。而只是保存输入和激活函数，当进行反向传播的时候，会重新获取输入和激活函数计算激活值用于梯度计算。因此向前传播会计算两遍，所以需要更多的训练时间。
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    # use_cache设置为False，是因为和gradient checkpoint存在冲突。因为use_cache是对解码速度的优化，在解码器解码时，存储每一步输出的hidden-state用于下一步的输入，而因为开启了gradient checkpoint，中间激活值不会存储，因此use_cahe=False
    model.config.use_cache = False
    
    # add LoRA adaptor
    model = get_peft_model(model, lora_config)

    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint is not None:
        # Full checkpoint
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = False  # So the trainer won't try loading its state
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")
    
    # 打印模型的可训练参数量
    model.print_trainable_parameters()
    
    # 加载数据集
    data = load_dataset(path="json", data_files=args.data_path)
    column_names = data["train"].column_names
        
    def tokenize_function(example):
        question = example["instruction"]
        if example.get("input"):
            if example["input"].strip():
                question += f"\n{example['input']}"
        answer = example["output"]
        
        q_ids = tokenizer.encode(text=question, add_special_tokens=False)
        a_ids = tokenizer.encode(text=answer, add_special_tokens=False)
        if len(q_ids) > args.max_input_length - 1:
            q_ids = q_ids[: args.max_input_length - 1]
        if len(a_ids) > args.max_output_length - 2:
            a_ids = a_ids[: args.max_output_length - 2]
        
        input_ids = tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)
        question_length = input_ids.index(tokenizer.bos_token_id)
        labels = [-100] * question_length + input_ids[question_length: ]
        return {"input_ids": input_ids, "labels": labels}
    
    train_dataset = data["train"].map(tokenize_function, remove_columns=column_names)
    eval_dataset = None
    if args.eval_path is not None:
        eval_data = load_dataset(path="json", data_files=args.eval_path)
        eval_dataset = eval_data["train"].map(tokenize_function, remove_columns=column_names)
    
    # trainer
    trainer = ModifiedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(pad_token_id=tokenizer.pad_token_id)
    )
    
    # 打印模型参数信息
    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print(f"{name},------------,{param.device}")
    
    # 训练模型
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save our LoRA model & tokenizer results
    trainer.model.save_pretrained(training_args.output_dir)
    #tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train(args)

# 启动训练: 
# 1、数据并行: CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train.py --train_args_file ./conf/chatglm_6b_lora.json --model_name_or_path ../../chatglm-6b-model/ --data_path ./data/AdvertiseGen/train.jsonl --max_input_length 128 --max_output_length 256
# 2、模型并行: CUDA_VISIBLE_DEVICES=1,2 python train.py --train_args_file ./conf/chatglm_6b_lora.json --model_name_or_path ../../chatglm-6b-model/ --data_path ./data/AdvertiseGen/train.jsonl --max_input_length 128 --max_output_length 256 --int8

# 启动推理: CUDA_VISIBLE_DEVICES=1 python inference.py --model_name_or_path ../../chatglm-6b-model/ --lora_checkpoint ./output/adgen-chatglm-6b-lora/
