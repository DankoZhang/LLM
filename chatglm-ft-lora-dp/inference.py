#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File    :   inference.py
brief   :   brief
Date    :   2023/07/30 10:35:34
Author  :   zhangce06
Contact :   zhangce06@baidu.com
"""

import torch
from transformers import AutoModel, AutoTokenizer
# from transformers.generation.utils import LogitsProcessorList, LogitsProcessor
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from peft import PeftModel

from args import args


class Generator:
    """
    生成文本
    """
    def __init__(self):
        """
        init
        """
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(
            model,
            args.lora_model
        )

        # 推断时直接转换成FP16
        # self.model.half()
        self.model.cuda()
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, title):
        """
        推理
        """
        title += "。净含量是："
        # 处理器
        logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        
        # 生成参数
        # max_length: 最大输入长度
        # num_beams=1, do_sample=True: 对候选词做一个采样，而不是选条件概率最高的词，增加多样性
        # top_p: 用于构建候选词表，已知生成各个词的总概率是1（即默认是1.0），如果top_p小于1，则从高到低累加直到top_p，取这前N个词作为候选词表
        # temperature: 值越低，生成内容越稳定，不能设定为0，会报错
        gen_kwargs = {"max_length": 2048, "num_beams": 1, "do_sample": True, "top_p": 0.7,
                      "temperature": 0.01, "logits_processor": logits_processor}

        inputs = self.tokenizer([title], return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        outputs = self.model.generate(input_ids=input_ids, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = self.tokenizer.decode(outputs)

        return response


class InvalidScoreLogitsProcessor(LogitsProcessor):
    """
    继承自LogitsProcessor。该类用于处理可能出现的NaN和inf值，通过将它们替换为零来确保计算的稳定性
    """
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        call
        """
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


if __name__ == "__main__":
    generator = Generator()

    pred = generator.evaluate("西红柿500g")
    print(pred)