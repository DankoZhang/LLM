# LLM
本项目实现LLM微调，包括ChatGLM-6B+LoRA、ChatGLM2-6B+LoRA等方案。
## 一、ChatGLM-6B+LoRA
1、环境配置：
* transformers==4.30.2

* datasets==2.10.1

* cpm_kernels==1.0.11

* torch==1.13.0+cu116

* peft==0.4.0

如果启用load_in_8bit，则还需要安装accelerate、bitsandbytes

2、启动训练：
* 数据并行:
```python
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train.py --train_args_file ./conf/chatglm_6b_lora.json --model_name_or_path ../../chatglm-6b-model/ --data_path ./data/AdvertiseGen/train.jsonl --max_input_length 128 --max_output_length 256
```
* 模型并行:
```python
CUDA_VISIBLE_DEVICES=1,2 python train.py --train_args_file ./conf/chatglm_6b_lora.json --model_name_or_path ../../chatglm-6b-model/ --data_path ./data/AdvertiseGen/train.jsonl --max_input_length 128 --max_output_length 256
```
3、启动推理：
```python
CUDA_VISIBLE_DEVICES=1 python inference.py --model_name_or_path ../../chatglm-6b-model/ --lora_checkpoint ./output/adgen-chatglm-6b-lora/
```
## 二、ChatGLM2-6B+LoRA
1、环境配置：
* transformers==4.30.2

* datasets==2.10.1

* cpm_kernels==1.0.11

* torch==1.13.0+cu116

* peft==0.4.0

如果启用load_in_8bit，则还需要安装accelerate、bitsandbytes

2、启动训练：
* 数据并行:
```python
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train.py --train_args_file ./conf/chatglm2_6b_lora.json --model_name_or_path ../../chatglm2-6b-model/ --data_path ./data/AdvertiseGen/train.jsonl --max_input_length 128 --max_output_length 256
```
* 模型并行:
```python
CUDA_VISIBLE_DEVICES=1,2 python train.py --train_args_file ./conf/chatglm2_6b_lora.json --model_name_or_path ../../chatglm2-6b-model/ --data_path ./data/AdvertiseGen/train.jsonl --max_input_length 128 --max_output_length 256
```
3、启动推理：
```python
CUDA_VISIBLE_DEVICES=1 python inference.py --model_name_or_path ../../chatglm2-6b-model/ --lora_checkpoint ./output/adgen-chatglm2-6b-lora/
```
