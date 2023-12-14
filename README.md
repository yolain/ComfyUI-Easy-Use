<p align="right">
   <strong>中文</strong> | <a href="./README.en.md">English</a>
</p>

<div align="center">

# ComfyUI Easy Use

为了更加方便简单地使用ComfyUI，我对一部分常用的节点做了一些优化与整合。

[![Bilibili Badge](https://img.shields.io/badge/视频介绍-00A1D6?style=for-the-badge&logo=bilibili&logoColor=white&link=https://www.bilibili.com/video/BV1vQ4y1G7z7)](https://www.bilibili.com/video/BV1vQ4y1G7z7/)
</div>

## 流程对比

<img src="./docs/workflow_node_compare.png">

EasyUse 在 [tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes) 的基础上做了简化，在简化的节点中去除了过多的传入和传出参数，建议您配合 [tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes) 中的 **pipeIn**、**pipeOut**、**pipeEdit** 使用，可参考下方示例里 [工作流](https://github.com/yolain/ComfyUI-Easy-Use?tab=readme-ov-file#sdturbo%E9%AB%98%E6%B8%85%E4%BF%AE%E5%A4%8Dsvd)。

### 更新

**2023-12-14**

- `easy a1111Loader` 和 `easy comfyLoader` 新增 `batch_size` 传入参数
- 新增 `easy controlnetLoaderADV` 节点
- `easy controlnetLoaderADV` 和 `easy controlnetLoader` 新增 `control_net` 可选传入参数
- `easy preSampling` 和 `easy preSamplingAdvanced` 新增 `image_to_latent` 可选传入参数
- 新增 `easy imageSizeBySide` 节点，可选输出为长边或短边

<details>
<summary><b>2023-12-13</b></summary>

-  新增 `easy LLLiteLoader` 节点，如果您预先安装过 kohya-ss/ControlNet-LLLite-ComfyUI 包，请将 models 里的模型文件移动至 ComfyUI\models\controlnet\ (即comfy默认的controlnet路径里，请勿修改模型的文件名，不然会读取不到)。
-  修改 `easy controlnetLoader` 到 loader 分类底下。
-  新增 `easy imageSize` 和 `easy imageSizeByLongerSize` 输出的尺寸显示。
</details>

<details>
<summary><b>2023-12-11</b></summary>

-  新增 `easy showSpentTime` 节点用于展示图片推理花费时间与VAE解码花费时间。
</details>

### 主要的优化

- 增加了 **preSampling** 预采样参数节点，目的是为了把采样参数配置与采样时的实时预览图分离。
- 调整种子默认的生成时序，从**control_after_generate**修改为**control_before_generate**。 

### 涉及到的相关节点包

声明: 非常尊重这些原作者们的付出，开源不易，我仅仅只是做了一些整合与优化。

| 节点名                        | 相关的库                                                                        | 库相关的节点                  |
|:---------------------------|:----------------------------------------------------------------------------|:------------------------|
| easy SetNode               | [diffus3/ComfyUI-extensions](https://github.com/diffus3/ComfyUI-extensions) | diffus3.SetNode         |
| easy GetNode               | [diffus3/ComfyUI-extensions](https://github.com/diffus3/ComfyUI-extensions) | diffus3.GetNode         |
| easy LLLiteLoader          | [kohya-ss/ControlNet-LLLite-ComfyUI](https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI) | LLLiteLoader            |
| easy GlobalSeed            | [ltdrdata/ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) | Global Seed (Inspire)   | 
| easy PreSamplingDynamicCFG | [sd-dynamic-thresholding](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding) | DynamicThresholdingFull | 
| DynamicThresholdingFull    | [sd-dynamic-thresholding](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding) | DynamicThresholdingFull | 
| easy ImageInsetCrop        | [rgthree/rgthree-comfy](https://github.com/rgthree/rgthree-comfy) | ImageInsetCrop          | 

## 示例

导入后请自行更换您目录里的大模型

### 文生图

<img src="./docs/text_to_image.png">

### 图生图+controlnet

<img src="./docs/image_to_image_controlnet.png">

### SDTurbo+高清修复+SVD

<img src="./docs/sdturbo_hiresfix_svd.png">