# LLM
本项目实现LLM微调，包括ChatGLM+LoRA、ChatGLM+LoRA等方案。
## 一、ChatGLM+LoRA
1、环境配置：
* transformers==4.30.2

* datasets==2.10.1

* cpm_kernels==1.0.11

* torch==1.13.0+cu116

* peft==0.4.0

  如果启用load_in_8bit，则还需要安装accelerate、bitsandbytes
