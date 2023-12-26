<p align="right">
   <strong>中文</strong> | <a href="./README.en.md">English</a>
</p>

<div align="center">

# ComfyUI Easy Use

[![Bilibili Badge](https://img.shields.io/badge/基本介绍(较早版本)-00A1D6?style=for-the-badge&logo=bilibili&logoColor=white&link=https://www.bilibili.com/video/BV1vQ4y1G7z7)](https://www.bilibili.com/video/BV1vQ4y1G7z7/)
</div>

**ComfyUI-Easy-Use** 是一个化繁为简的节点整合包, 在 [tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes) 的基础上进行延展，并针对了诸多主流的节点包做了整合与优化，以达到更快更方便使用ComfyUI的目的，在保证自由度的同时还原了本属于Stable Diffusion的极致畅快出图体验。

## 介绍

### 更符合人性化的随机种
<img src="./docs/seed_generate_compare.jpg">

### 分离采样参数与采样预览

<img src="./docs/workflow_node_compare.png">

### 支持通配符与Lora的提示词节点

支持 `yaml` `txt` `json` 格式的通配符文件，將其放置到节点包的 `wildcards` 文件夹下即可，更新文件需重新运行ComfyUI。 <br>
如需使用Lora Block Weight用法，需先保证自定义节点包中安装了 [ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack)。

### UI界面美化

安装节点包后会自动切换UI界面，如需更换其他主题请在 Settings -> Color Palette 中自行切换并刷新页面即可。

### 简体中文节点

如果您下载了 [AIGODLIKE-COMFYUI-TRANSLATION](https://github.com/AIGODLIKE/AIGODLIKE-ComfyUI-Translation)
, 程序将在启动时拷贝中文对照文件至该节点包目录下，当您选择语言为中文时即可看到已更改后的简体中文节点。

## 更新日志

**v1.0.1（2023-12-26）**

- 修复 `easy zero123Loader` 和 `easy svdLoader` 读取模型时将模型加入到缓存中
- 新增 `easy seed` - 简易随机种
- 修复 `easy kSampler` `easy kSamplerTiled` `easy detailerFix` 的 `image_output` 默认值为 Preview
- `easy fullLoader` 和 `easy a1111Loader` 新增了 `a1111_prompt_style` 参数可以重现和webui生成相同的图像，当前您需要安装 [ComfyUI_smZNodes](https://github.com/shiimizu/ComfyUI_smZNodes) 才能使用此功能

**2023-12-25**

- 修复 `easy globalSeed` 不生效问题
- 修复所有的`seed_num` 因 [cg-use-everywhere](https://github.com/chrisgoringe/cg-use-everywhere) 实时更新图表导致值错乱的问题

**v1.0.0（2023-12-24）**

- 新增`easy positive` - 简易正面提示词文本
- 新增`easy negative`  - 简易负面提示词文本
- 新增`easy wildcards` - 支持通配符和Lora选择的提示词文本
- 新增`easy portraitMaster` - 肖像大师v2.2
- 新增`easy loraStack` - Lora堆
- 新增`easy fullLoader` - 完整版的加载器
- 新增`easy zero123Loader` - 简易zero123加载器
- 新增`easy svdLoader` - 简易svd加载器
- 新增`easy fullkSampler` - 完整版的采样器（无分离）
- 新增`easy hiresFix` - 支持Pipe的高清修复
- 新增`easy predetailerFix` `easy DetailerFix` - 支持Pipe的细节修复
- 新增`easy ultralyticsDetectorPipe` `easy samLoaderPipe` - 检测加载器（细节修复的输入项）
- 新增`easy pipein` `easy pipeout` - Pipe的输入与输出
- 新增`easy xyPlot` - 简易的xyplot (后续会更新更多可控参数)
- 新增`easy imageRemoveBG` - 图像去除背景
- 新增`easy imagePixelPerfect` - 图像完美像素
- 新增`easy poseEditor` - 姿势编辑器
- 新增UI主题（黑曜石）- 默认自动加载UI, 也可在设置中自行更替 


- 修复`easy imageSize` `easy imageSizeBySide` `easy imageSizeByLongerSide` 可作为终节点
- 修复 `seed_num` (随机种子值) 在历史记录中读取无法一致的Bug

<details>
<summary><b>2023-12-14</b></summary>

- `easy a1111Loader` 和 `easy comfyLoader` 新增 `batch_size` 传入参数
- 新增 `easy controlnetLoaderADV` 节点
- `easy controlnetLoaderADV` 和 `easy controlnetLoader` 新增 `control_net` 可选传入参数
- `easy preSampling` 和 `easy preSamplingAdvanced` 新增 `image_to_latent` 可选传入参数
- 新增 `easy imageSizeBySide` 节点，可选输出为长边或短边
</details>

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

## 涉及到的相关节点包

声明: 非常尊重这些原作者们的付出，开源不易，我仅仅只是做了一些整合与优化。

| 节点名 (搜索名)                  | 相关的库                                                                        | 库相关的节点                  |
|:---------------------------|:----------------------------------------------------------------------------|:------------------------|
| easy setNode               | [ComfyUI-extensions](https://github.com/diffus3/ComfyUI-extensions) | diffus3.SetNode         |
| easy getNode               | [ComfyUI-extensions](https://github.com/diffus3/ComfyUI-extensions) | diffus3.GetNode         |
| easy bookmark              | [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) | Bookmark 🔖             |
| easy portraitMarker        | [comfyui-portrait-master](https://github.com/florestefano1975/comfyui-portrait-master) | Portrait Master         |
| easy LLLiteLoader          | [ControlNet-LLLite-ComfyUI](https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI) | LLLiteLoader            |
| easy globalSeed            | [ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) | Global Seed (Inspire)   | 
| easy preSamplingDynamicCFG | [sd-dynamic-thresholding](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding) | DynamicThresholdingFull | 
| dynamicThresholdingFull    | [sd-dynamic-thresholding](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding) | DynamicThresholdingFull | 
| easy imageInsetCrop        | [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) | ImageInsetCrop          | 
| easy poseEditor            | [ComfyUI_Custom_Nodes_AlekPet](https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet) | poseNode                | 

## 示例

导入后请自行更换您目录里的大模型

### 文生图

<img src="./workflow/text_to_image.png">

### 图生图+controlnet

<img src="./workflow/image_to_image_controlnet.png">

### SDTurbo+高清修复+SVD

<img src="./workflow/sdturbo_hiresfix_svd.png">

## Credits

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 功能强大且模块化的Stable Diffusion GUI

[ComfyUI-ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) - ComfyUI管理器

[tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes) - 管道节点（节点束）让用户减少了不必要的连接

[ComfyUI-extensions](https://github.com/diffus3/ComfyUI-extensions) - diffus3的获取与设置点让用户可以分离工作流构成 


[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) - 常规整合包1

[ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) - 常规整合包2