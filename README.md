# LLM
本项目实现LLM微调，包括ChatGLM-6B+LoRA、ChatGLM2-6B+LoRA、ChatGLM-6B+LoRA+Accelerate+Deepspeed等方案。
## 一、ChatGLM-6B+LoRA
1、环境配置：以下所有安装包的版本都是推荐，可按实际情况自行调整
* transformers==4.30.2

* datasets==2.10.1（若报错，安装最新版本即可）

* cpm_kernels==1.0.11

* torch==1.13.0+cu116

* peft==0.4.0

如果启用load_in_8bit，则还需要安装accelerate、bitsandbytes、scipy、tensorboardX
* accelerate==0.21.0

* bitsandbytes==0.41.0

* scipy==1.11.1

* tensorboardX==2.6.1

2、启动训练：
* 数据并行:
```python
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train.py --train_args_file ./conf/chatglm_6b_lora.json --model_name_or_path ../../chatglm-6b-model/ --data_path ./data/AdvertiseGen/train.jsonl --max_input_length 128 --max_output_length 256
```
* 模型（流水线）并行:
```python
CUDA_VISIBLE_DEVICES=1,2 python train.py --train_args_file ./conf/chatglm_6b_lora.json --model_name_or_path ../../chatglm-6b-model/ --data_path ./data/AdvertiseGen/train.jsonl --max_input_length 128 --max_output_length 256 --int8
```
3、启动推理：
```python
CUDA_VISIBLE_DEVICES=1 python inference.py --model_name_or_path ../../chatglm-6b-model/ --lora_checkpoint ./output/adgen-chatglm-6b-lora/
```
## 二、ChatGLM2-6B+LoRA
1、环境配置：以下所有安装包的版本都是推荐，可按实际情况自行调整
* transformers==4.30.2

* datasets==2.10.1（若报错，安装最新版本即可）

* cpm_kernels==1.0.11

* torch==1.13.0+cu116

* peft==0.4.0

如果启用load_in_8bit，则还需要安装accelerate、bitsandbytes、scipy、tensorboardX
* accelerate==0.21.0

* bitsandbytes==0.41.0

* scipy==1.11.1

* tensorboardX==2.6.1

2、启动训练：
* 数据并行:
```python
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train.py --train_args_file ./conf/chatglm2_6b_lora.json --model_name_or_path ../../chatglm2-6b-model/ --data_path ./data/AdvertiseGen/train.jsonl --max_input_length 128 --max_output_length 256
```
* 模型（流水线）并行:
```python
CUDA_VISIBLE_DEVICES=1,2 python train.py --train_args_file ./conf/chatglm2_6b_lora.json --model_name_or_path ../../chatglm2-6b-model/ --data_path ./data/AdvertiseGen/train.jsonl --max_input_length 128 --max_output_length 256 --int8
```
3、启动推理：
```python
CUDA_VISIBLE_DEVICES=1 python inference.py --model_name_or_path ../../chatglm2-6b-model/ --lora_checkpoint ./output/adgen-chatglm2-6b-lora/
```
4、备注：
* 采用transformers.Trainer框架进行训练时，内部已经实现数据并行策略，因此不需要做类似DDP、Accelerate等框架的封装工作。另外，保存模型调用的save_pretrained方法，会自动保存主进程的模型，因此也不用进行是否是主进程的判断；
* model.chat()方法中，max_length指的是输入+输出的长度，详见[model.chat()方法中的max_length](https://github.com/THUDM/ChatGLM-6B/issues/812)
## 三、ChatGLM-6B+LoRA+Accelerate+Deepspeed
1、环境配置：
* 包括Docker环境构建+Python环境构建，参考[通俗易懂的LLM](https://blog.csdn.net/qq_39439006/article/details/130796416?spm=1001.2014.3001.5502)。

2、启动训练：
```python
accelerate launch --config_file ./conf/accelerate_config.yaml train.py
```
3、启动推理：
```python
CUDA_VISIBLE_DEVICES=1 python inference.py
```
4、备注：
* 加载模型及源码修改的注意事项，同样参考[通俗易懂的LLM](https://blog.csdn.net/qq_39439006/article/details/130796416?spm=1001.2014.3001.5502)；
* 若模型采用ChatGLM2-6B，则参考./chatglm2-ft-lora/train.py line148，对tokenize进行修改即可。

