#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File    :   utils.py
brief   :   brief
Date    :   2023/07/26 13:24:14
Author  :   zhangce06
Contact :   zhangce06@baidu.com
"""


from loguru import logger
from pathlib import Path
import time

project_path = Path.cwd()
log_path = Path(project_path, "logs")
today = time.strftime("%Y_%m_%d")
logger.add(f"{log_path}/train_chatglm_lora_dp_{today}.log", rotation='100MB', encoding="utf-8", retention=3, level="INFO")