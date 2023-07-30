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
Date    :   2023/07/26 08:06:27
Author  :   zhangce06
Contact :   zhangce06@baidu.com
"""


import os
import time

import torch
from torch.nn.utils import clip_grad_norm_

from accelerate import Accelerator, DistributedType
from accelerate.utils import DummyOptim, DummyScheduler

from transformers import AdamW, get_linear_schedule_with_warmup, get_scheduler, AutoTokenizer

from model import ChatGLMLoraModel
from train_tool import ChatGLMDataLoader
from metric import mean, accuracy
from utils import logger
from args import args

def set_seed(seed):
    """
    设定随机数种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Trainer:
    def __init__(self):
        """
        init
        """
        self.epochs = args.epochs
        self.log_every = args.log_every
        self.eval_every = args.eval_every
        self.checkpoint_every = args.checkpoint_every

        self.train_steps = args.train_steps
        self.warmup_steps = args.warmup_steps
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay

        self.batch_size = args.batch_size
        self.accu_steps = args.accu_steps

        self.lora_model = args.lora_model

        self.accelerator = Accelerator()
        print("dist type: ", self.accelerator.distributed_type)
        print("mix-precision: ", self.accelerator.mixed_precision)

        # 初始化tokenizer对象
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        print("tokenizer init done: ", len(self.tokenizer))

        self.data_loader_tool = ChatGLMDataLoader(self.tokenizer, self.batch_size)
        self.train_data_loader, self.valid_data_loader = self.data_loader_tool.get_data_loader()
        print("get data loader done")

        # 初始化模型对象
        self.model = ChatGLMLoraModel()
        print("model load done")
        # 构建优化器、学习率迭代器
        self.optimizer, self.scheduler = self.create_optimizer_and_scheduler()

        # accelerate初始化模型、优化器等对象
        self.model, self.optimizer, self.train_data_loader, self.valid_data_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_data_loader, self.valid_data_loader, self.scheduler
        )

        self.model.train()

    def create_optimizer_and_scheduler(self):
        """
        构建优化器和学习率
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # 如果deepspeed config中定义了optimizer和scheduler，则需要使用DummyOptim和DummyScheduler
        optimizer_cls = (
            AdamW
            if self.accelerator.state.deepspeed_plugin is None
               or "optimizer" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
            else DummyOptim
        )
        
        # 构建优化器
        optimizer = optimizer_cls(optimizer_grouped_parameters, lr=self.learning_rate)

        # 构建
        if (
                self.accelerator.state.deepspeed_plugin is None
                or "scheduler" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            scheduler = get_scheduler(
                name="linear",
                optimizer=optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.train_steps,
            )
        else:
            scheduler = DummyScheduler(
                optimizer, total_num_steps=self.train_steps, warmup_num_steps=self.warmup_steps
            )

        return optimizer, scheduler

    def eval(self):
        """
        评估模型
        """
        self.model.eval()
        with torch.no_grad():
            eval_losses = []
            eval_word_preds = []
            eval_word_labels = []
            for batch_data in self.valid_data_loader:
                input_ids = batch_data[0].to(self.accelerator.device)
                labels = batch_data[1].to(self.accelerator.device)

                loss, predictions = self.model(input_ids, labels)

                loss, predictions, labels = self.accelerator.gather_for_metrics((loss, predictions, labels))
                loss = loss.mean()
                eval_losses.append(float(loss))
                eval_word_preds.extend(predictions.tolist())
                eval_word_labels.extend(labels.tolist())

            if self.accelerator.is_main_process:
                acc = accuracy(pred_ys=eval_word_preds, true_ys=eval_word_labels)

                logger.info("\n")
                logger.info("eval: num: {},  loss: {}, acc: {}".format(
                    len(eval_word_preds), mean(eval_losses), acc))
                logger.info("\n")

    def train(self):
        """
        模型训练
        """
        current_step = 1
        start = time.time()

        train_losses = []
        train_word_preds = []
        train_word_labels = []

        for epoch in range(self.epochs):
            logger.info("----- Epoch {}/{} -----".format(epoch + 1, self.epochs))

            for batch_data in self.train_data_loader:

                input_ids = batch_data[0].to(self.accelerator.device)
                labels = batch_data[1].to(self.accelerator.device)

                loss, predictions = self.model(input_ids, labels)

                # 为了训练加速，减少gpu间的通信，只在主进程上看训练的日志
                if self.accelerator.is_main_process:
                    train_losses.append(float(loss))
                    train_word_preds.extend(predictions.tolist())
                    train_word_labels.extend(labels.tolist())

                # 梯度累积训练
                loss /= self.accu_steps
                self.accelerator.backward(loss)

                if current_step % self.accu_steps == 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                # 打印日志
                if current_step % (self.log_every * self.accu_steps) == 0:
                    if self.accelerator.is_main_process:
                        acc = accuracy(pred_ys=train_word_preds, true_ys=train_word_labels)
                        logger.info("train: step: {}, num: {}, loss: {}, acc: {}".format(
                            current_step // self.accu_steps, len(train_word_preds), mean(train_losses), acc))

                        train_losses = []
                        train_word_preds = []
                        train_word_labels = []
                # 评估模型
                if current_step % (self.eval_every * self.accu_steps) == 0:
                    self.eval()
                    self.accelerator.wait_for_everyone()
                    self.model.train()

                current_step += 1

                if (current_step // self.accu_steps) > self.train_steps:
                    break
            if (current_step // self.accu_steps) > self.train_steps:
                break

        end = time.time()
        print("total train time: ", end - start)

        # 保存模型
        self.accelerator.wait_for_everyone()
        self.model.module.peft_model.save_pretrained(
            self.lora_model,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(self.model)
        )
        print("model save done")


def main():
    """
    训练主函数
    """
    set_seed(args.seed)

    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()
