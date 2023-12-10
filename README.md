<p align="right">
   <strong>中文</strong> | <a href="./README.en.md">English</a>
</p>

<div align="center">

# ComfyUI Easy Use

为了更加方便简单地使用ComfyUI，我对一部分常用的节点做了一些优化与整合。

[//]: # ([![Bilibili Badge]&#40;https://img.shields.io/badge/使用说明视频-00A1D6?style=for-the-badge&logo=bilibili&logoColor=white&link=https://space.bilibili.com/1840885116&#41;]&#40;https://space.bilibili.com/1840885116&#41;)
</div>

## 流程对比


<img src="./docs/workflow_node_compare.png">

EasyUse在[tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes)的基础上做了简化，建议搭配原版的[tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes)节点包进行使用。

### 主要的优化

- 增加了 **preSampling** 预采样参数节点，目的是为了把采样参数配置与采样时的实时预览图分离。
- 调整种子默认的生成时序，从**control_after_generate**修改为**control_before_generate**。 

## 示例

### 文生图

<img src="./docs/text_to_image.png">

### 图生图+controlnet

<img src="./docs/image_to_image_controlnet.png">

### SDTurbo+高清修复+SVD

<img src="./docs/sdturbo_hiresfix_svd.png">