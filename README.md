<p align="right">
   <strong>中文</strong> | <a href="./README.en.md">English</a>
</p>

<div align="center">

# ComfyUI Easy Use

[![Bilibili Badge](https://img.shields.io/badge/1.0版本-00A1D6?style=for-the-badge&logo=bilibili&logoColor=white&link=https://www.bilibili.com/video/BV1Wi4y1h76G)](https://www.bilibili.com/video/BV1Wi4y1h76G)
[![Bilibili Badge](https://img.shields.io/badge/基本介绍-00A1D6?style=for-the-badge&logo=bilibili&logoColor=white&link=https://www.bilibili.com/video/BV1vQ4y1G7z7)](https://www.bilibili.com/video/BV1vQ4y1G7z7/)
</div>

**ComfyUI-Easy-Use** 是一个化繁为简的节点整合包, 在 [tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes) 的基础上进行延展，并针对了诸多主流的节点包做了整合与优化，以达到更快更方便使用ComfyUI的目的，在保证自由度的同时还原了本属于Stable Diffusion的极致畅快出图体验。

## 介绍

### 更符合人性化的随机种
<img src="https://raw.githubusercontent.com/yolain/yolain-comfyui-workflow/main/Docs/seed_generate_compare.jpg">

### 分离采样参数与采样预览

<img src="https://raw.githubusercontent.com/yolain/yolain-comfyui-workflow/main/Docs/workflow_node_compare.png">

### 支持通配符与Lora的提示词节点

支持 `yaml` `txt` `json` 格式的通配符文件，將其放置到节点包的 `wildcards` 文件夹下即可，更新文件需重新运行ComfyUI。 <br>
如需使用Lora Block Weight用法，需先保证自定义节点包中安装了 [ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack)。

### UI界面美化

首次安装的用户，如需使用本节点包UI主题，请在安装节点包在 Settings -> Color Palette 中自行切换并**刷新页面**即可。

### 简体中文节点

如果您下载了 [AIGODLIKE-COMFYUI-TRANSLATION](https://github.com/AIGODLIKE/AIGODLIKE-ComfyUI-Translation)
, 程序将在启动时拷贝中文对照文件至该节点包目录下，当您选择语言为中文时即可看到已更改后的简体中文节点。

## 更新日志

**2024-01-31**

- 修复 `easy wildcards` 在使用PS扩展插件运行时报错
- 增加 `easy showLoaderSettingsNames` 可显示与输出加载器部件中的 模型与VAE名称

**2024-01-29**

- 更改首次安装节点包不再自动替换主题，需手动调整并刷新页面
- `easy imageSave` 增加 **only_preivew**
- 增加 `easy promptList` - 提示词列表
- 修改 `easy latentCompositeMaskedWithCond`

**v1.0.4（2024-01-28）**

- `easy kSamplerInpainting` 增加 **patch** 传入值，配合Fooocus内补节点使用
- 增加 `easy fooocusInpaintLoader` - Fooocus内补节点（仅支持XL模型的流程）
- 增加 **Logic** 逻辑类节点 - 包含类型、计算、判断和转换类型等
- 增加 `easy imageSave` - 带日期转换和宽高格式化的图像保存节点
- 增加 `easy joinImageBatch` - 合并图像批次
- 修复 `easy XYInputs: ControlNet` 报错
- 修复 `easy loraStack` **toggle** 为 disabled 时报错

**2024-01-22**

- 修复 `easy XYInputs: Sampler/Scheduler` 报错
- 修复 右侧菜单 点击按钮时老是跑位的问题 
- 调整UI主题，分为官方默认背景和深黑色背景两套样式，可在设置里调色板里切换（调整原因：深黑色背景对亮度较暗的用户和UP主不太友好）

<details>
<summary>2024-01-21</summary>>

- 修改 styles 路径以兼容其他环境
- 修复 `easy comfyLoader` 读取错误
- 修复 xyPlot 在连接 zero123 时报错
</details>

**v1.0.3（2024-01-19）**

- 修复加载器中提示词为组件时报错
- 增加 `easy stylesSelector` 风格化提示词选择器
- `easy controlnetLoader` 和 `easy controlnetLoaderADV` 增加参数 **scale_soft_weights** 
- 增加队列进度条设置项，默认为未启用状态
- 修复 `easy getNode` 和 `easy setNode` 加载时标题未更改
- 修复所有采样器中存储图片使用子目录前缀不生效的问题

<details>
<summary>2024-01-12</summary>

- 修复在Comfy新版本中UI加载失败
</details>

<details>
<summary>2024-01-11</summary>

- 替换了XY图生成时的字体文件
</details>

<details>
<summary><b>2024-01-09</b></summary>

- 修复 `easy imageInsetCrop` 测量值为百分比时步进为1
- 修复 开启 `a1111_prompt_style` 时XY图表无法使用的问题
- 增加了一个 **autocomplete** 文件夹，如果您安装了 [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts), 将在启动时合并该文件夹下的所有txt文件并覆盖到pyssss包里的autocomplete.txt文件。
- 右键菜单中增加了一个 `📜Groups Map(EasyUse)` 
</details>

<details>
<summary><b>2024-01-08</b></summary>

- 修改 `easy fullLoader` 和 `easy a1111Loader` 中的 **a1111_prompt_style** 默认值为 False
- `easy XYInputs ModelMergeBlocks` 支持csv文件导入数值
- 修复 `easy pipeToBasicPipe` 报错
- 移除了 `easy imageRemBg` (很多包中都有移除背景的节点，由于比较占用ComfyUI启动加载时间，这次更新中注释掉了其相关代码)
</details>

**v1.0.2（2024-01-05）**

- 增加 `easy XYPlotAdvanced` 和 `easy XYInputs` 等相关节点
- 增加 **Alt+1到9** 快捷键，可快速粘贴 Node templates 的节点预设 （对应 1到9 顺序）
- 移除包中的介绍图和工作流文件，减少包体积

<details>
<summary><b>2024-01-03</b></summary>

- 修复 `easy comfyLoader` 报错
- 修复所有包含输出图片尺寸的节点取值方式无法批处理的问题
- 增加 `easy kSamplerInpainting` 用于内补潜空间的采样器
- 增加 `easy pipeToBasicPipe` 用于转换到Impact的某些节点上
</details>

<details>
<summary><b>2024-01-02</b></summary>

- 修复 `width` 和 `height` 无法在 `easy svdLoader` 自定义的报错问题
- 修复所有采样器预览图片的地址链接 (解决在 MACOS 系统中图片无法在采样器中预览的问题）
</details>

<details>
<summary><b>2023-12-31</b></summary>

- 修复 `vae_name` 在 `easy fullLoader` 和 `easy a1111Loader` 和 `easy comfyLoader` 中选择但未替换原始vae问题
- 修复 `easy fullkSampler` 除pipe外其他输出值的报错
</details>

<details>
<summary><b>2023-12-29</b></summary>

- 修复 `easy hiresFix` 输入连接pipe和image、vae同时存在时报错
- 修复 `easy fullLoader` 中 `model_override` 连接后未执行 
</details>

<details>
<summary><b>2023-12-27</b></summary>

- 修复 因新增`easy seed` 导致action错误
- 修复 `easy xyplot` 的字体文件路径读取错误
- 修复 convert 到 `easy seed` 随机种无法固定的问题
- 修复 `easy pipeIn` 值传入的报错问题
- `easy preDetailerFix` 新增了 `optional_image` 传入图像可选，如未传默认取值为pipe里的图像
</details>

**v1.0.1（2023-12-26）**

- 修复 `easy zero123Loader` 和 `easy svdLoader` 读取模型时将模型加入到缓存中
- 新增 `easy seed` - 简易随机种
- 修复 `easy kSampler` `easy kSamplerTiled` `easy detailerFix` 的 `image_output` 默认值为 Preview
- `easy fullLoader` 和 `easy a1111Loader` 新增了 `a1111_prompt_style` 参数可以重现和webui生成相同的图像，当前您需要安装 [ComfyUI_smZNodes](https://github.com/shiimizu/ComfyUI_smZNodes) 才能使用此功能

<details>
<summary><b>2023-12-25</b></summary>

- 修复 `easy globalSeed` 不生效问题
- 修复所有的`seed_num` 因 [cg-use-everywhere](https://github.com/chrisgoringe/cg-use-everywhere) 实时更新图表导致值错乱的问题
</details>

<details>
<summary><b>v1.0.0（2023-12-24）</b></summary>

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
</details>

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
| easy if                    | [ComfyUI-Logic](https://github.com/theUpsider/ComfyUI-Logic) | IfExecute                    | 

## 示例

导入后请自行更换您目录里的大模型

### 文生图

<img src="https://raw.githubusercontent.com/yolain/yolain-comfyui-workflow/main/Workflows/Simple/text_to_image.png">

### 图生图+controlnet

<img src="https://raw.githubusercontent.com/yolain/yolain-comfyui-workflow/main/Workflows/Simple/image_to_image_controlnet.png">

### SDTurbo+高清修复+SVD

<img src="https://raw.githubusercontent.com/yolain/yolain-comfyui-workflow/main/Workflows/Simple/sdturbo_hiresfix_svd.png">

## Credits

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 功能强大且模块化的Stable Diffusion GUI

[ComfyUI-ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) - ComfyUI管理器

[tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes) - 管道节点（节点束）让用户减少了不必要的连接

[ComfyUI-extensions](https://github.com/diffus3/ComfyUI-extensions) - diffus3的获取与设置点让用户可以分离工作流构成

[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) - 常规整合包1

[ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) - 常规整合包2

[ComfyUI-Logic](https://github.com/theUpsider/ComfyUI-Logic) -  ComfyUI逻辑运算