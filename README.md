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


### Stable Cascade

[工作流示例](https://github.com/yolain/ComfyUI-Easy-Use?tab=readme-ov-file#StableCascade) <br><br>
目前支持文生图与图生图，还未支持Lora和Controlnet，敬请期待!<br>
stage_c 与 stage_b 可以使用[checkpoints](https://huggingface.co/stabilityai/stable-cascade/tree/main/comfyui_checkpoints)模型或原来的unet模型 <br><br>

使用方式：<br>
1.选择[checkpoints](https://huggingface.co/stabilityai/stable-cascade/tree/main/comfyui_checkpoints)模型无需额外加载其余的VAE及clip<br> 
2.选择Unet模型的话需要额外加载[stage_a](https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_a.safetensors)、[clip](https://huggingface.co/stabilityai/stable-cascade/resolve/main/text_encoder/model.safetensors)及图生图需要用到的[effnet_encoder](https://huggingface.co/stabilityai/stable-cascade/resolve/main/effnet_encoder.safetensors?download=true)和[previewer](https://huggingface.co/stabilityai/stable-cascade/resolve/main/previewer.safetensors)。<br>

## 更新日志


**v1.0.9 [2024-3-2]**

- 新增 `easy instantIDApply` - 需要先安装 [ComfyUI_InstantID](https://github.com/cubiq/ComfyUI_InstantID), 工作流参考[示例](https://github.com/yolain/ComfyUI-Easy-Use?tab=readme-ov-file#InstantID)
- 修复 `easy detailerFix` 未添加到保存图片格式化扩展名可用节点列表
- 修复 `easy XYInputs: PromptSR` 在替换负面提示词时报错

**v1.0.8 (f28cbf7)**

- `easy cascadeLoader` stage_c 与 stage_b 支持checkpoint模型 (需要下载[checkpoints](https://huggingface.co/stabilityai/stable-cascade/tree/main/comfyui_checkpoints)) 
- `easy styleSelector` 搜索框修改为不区分大小写匹配
- `easy fullLoader` 增加 **positive**、**negative**、**latent** 输出项
- 修复 SDXLClipModel 在 ComfyUI 修订版本号 2016[c2cb8e88] 及以上的报错（判断了版本号可兼容老版本）
- 修复 `easy detailerFix` 批次大小大于1时生成出错
- 修复`easy preSampling`等 latent传入后无法根据批次索引生成的问题
- 修复 `easy svdLoader` 报错
- 优化代码，减少了诸多冗余，提升运行速度
- 去除中文翻译对照文本

（翻译对照已由 [AIGODLIKE-COMFYUI-TRANSLATION](https://github.com/AIGODLIKE/AIGODLIKE-ComfyUI-Translation) 统一维护啦！
首次下载或者版本较早的朋友请更新 AIGODLIKE-COMFYUI-TRANSLATION 和本节点包至最新版本。）

**v1.0.7**

- 增加 `easy cascadeLoader` - stable cascade 加载器
- 增加 `easy preSamplingCascade` - stabled cascade stage_c 预采样参数
- 增加 `easy fullCascadeKSampler` - stable cascade stage_c 完整版采样器
- 增加 `easy cascadeKSampler` - stable cascade stage-c ksampler simple

**v1.0.6 (2024-02-16)**

- 增加 `easy XYInputs: Checkpoint`
- 增加 `easy XYInputs: Lora`
- `easy seed` 增加固定种子值时可手动切换随机种
- 修复 `easy fullLoader`等加载器切换lora时自动调整节点大小的问题
- 去除原有ttn的图片保存逻辑并适配ComfyUI默认的图片保存格式化扩展

**v1.0.5**

- 增加 `easy isSDXL` 
- `easy svdLoader` 增加提示词控制, 可配合open_clip模型进行使用
- `easy wildcards` 增加 **populated_text** 可输出通配填充后文本

**v1.0.4**

- 增加 `easy showLoaderSettingsNames` 可显示与输出加载器部件中的 模型与VAE名称
- 增加 `easy promptList` - 提示词列表
- 增加 `easy fooocusInpaintLoader` - Fooocus内补节点（仅支持XL模型的流程）
- 增加 **Logic** 逻辑类节点 - 包含类型、计算、判断和转换类型等
- 增加 `easy imageSave` - 带日期转换和宽高格式化的图像保存节点
- 增加 `easy joinImageBatch` - 合并图像批次
- `easy showAnything` 增加支持转换其他类型（如：tensor类型的条件、图像等）
- `easy kSamplerInpainting` 增加 **patch** 传入值，配合Fooocus内补节点使用
- `easy imageSave` 增加 **only_preivew**

- 修复 xyplot在pillow>9.5中报错
- 修复 `easy wildcards` 在使用PS扩展插件运行时报错
- 修复 `easy latentCompositeMaskedWithCond`
- 修复 `easy XYInputs: ControlNet` 报错
- 修复 `easy loraStack` **toggle** 为 disabled 时报错

- 修改首次安装节点包不再自动替换主题，需手动调整并刷新页面

<details>
<summary><b>v1.0.3</b></summary>

- 增加 `easy stylesSelector` 风格化提示词选择器
- 增加队列进度条设置项，默认为未启用状态
- `easy controlnetLoader` 和 `easy controlnetLoaderADV` 增加参数 **scale_soft_weights**


- 修复 `easy XYInputs: Sampler/Scheduler` 报错
- 修复 右侧菜单 点击按钮时老是跑位的问题
- 修复 styles 路径在其他环境报错
- 修复 `easy comfyLoader` 读取错误
- 修复 xyPlot 在连接 zero123 时报错
- 修复加载器中提示词为组件时报错
- 修复 `easy getNode` 和 `easy setNode` 加载时标题未更改
- 修复所有采样器中存储图片使用子目录前缀不生效的问题


- 调整UI主题
</details>

<details>
<summary><b>v1.0.2</b></summary>

- 增加 **autocomplete** 文件夹，如果您安装了 [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts), 将在启动时合并该文件夹下的所有txt文件并覆盖到pyssss包里的autocomplete.txt文件。
- 增加 `easy XYPlotAdvanced` 和 `easy XYInputs` 等相关节点
- 增加 **Alt+1到9** 快捷键，可快速粘贴 Node templates 的节点预设 （对应 1到9 顺序）

- 修复 `easy imageInsetCrop` 测量值为百分比时步进为1
- 修复 开启 `a1111_prompt_style` 时XY图表无法使用的问题
- 右键菜单中增加了一个 `📜Groups Map(EasyUse)` 

- 修复在Comfy新版本中UI加载失败
- 修复 `easy pipeToBasicPipe` 报错
- 修改 `easy fullLoader` 和 `easy a1111Loader` 中的 **a1111_prompt_style** 默认值为 False
- `easy XYInputs ModelMergeBlocks` 支持csv文件导入数值

- 替换了XY图生成时的字体文件

- 移除 `easy imageRemBg`
- 移除包中的介绍图和工作流文件，减少包体积

</details>

<details>
<summary><b>v1.0.1</b></summary>

- 新增 `easy seed` - 简易随机种
- `easy preDetailerFix` 新增了 `optional_image` 传入图像可选，如未传默认取值为pipe里的图像
- 新增 `easy kSamplerInpainting` 用于内补潜空间的采样器
- 新增 `easy pipeToBasicPipe` 用于转换到Impact的某些节点上

- 修复 `easy comfyLoader` 报错
- 修复所有包含输出图片尺寸的节点取值方式无法批处理的问题
- 修复 `width` 和 `height` 无法在 `easy svdLoader` 自定义的报错问题
- 修复所有采样器预览图片的地址链接 (解决在 MACOS 系统中图片无法在采样器中预览的问题）
- 修复 `vae_name` 在 `easy fullLoader` 和 `easy a1111Loader` 和 `easy comfyLoader` 中选择但未替换原始vae问题
- 修复 `easy fullkSampler` 除pipe外其他输出值的报错
- 修复 `easy hiresFix` 输入连接pipe和image、vae同时存在时报错
- 修复 `easy fullLoader` 中 `model_override` 连接后未执行 
- 修复 因新增`easy seed` 导致action错误
- 修复 `easy xyplot` 的字体文件路径读取错误
- 修复 convert 到 `easy seed` 随机种无法固定的问题
- 修复 `easy pipeIn` 值传入的报错问题
- 修复 `easy zero123Loader` 和 `easy svdLoader` 读取模型时将模型加入到缓存中
- 修复 `easy kSampler` `easy kSamplerTiled` `easy detailerFix` 的 `image_output` 默认值为 Preview
- `easy fullLoader` 和 `easy a1111Loader` 新增了 `a1111_prompt_style` 参数可以重现和webui生成相同的图像，当前您需要安装 [ComfyUI_smZNodes](https://github.com/shiimizu/ComfyUI_smZNodes) 才能使用此功能
</details>

<details>
<summary><b>v1.0.0</b></summary>

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

- 修复 `easy globalSeed` 不生效问题
- 修复所有的`seed_num` 因 [cg-use-everywhere](https://github.com/chrisgoringe/cg-use-everywhere) 实时更新图表导致值错乱的问题
- 修复`easy imageSize` `easy imageSizeBySide` `easy imageSizeByLongerSide` 可作为终节点
- 修复 `seed_num` (随机种子值) 在历史记录中读取无法一致的Bug
</details>


<details>
<summary><b>v0.5</b></summary>

- 新增 `easy controlnetLoaderADV` 节点
-  新增 `easy imageSizeBySide` 节点，可选输出为长边或短边
-  新增 `easy LLLiteLoader` 节点，如果您预先安装过 kohya-ss/ControlNet-LLLite-ComfyUI 包，请将 models 里的模型文件移动至 ComfyUI\models\controlnet\ (即comfy默认的controlnet路径里，请勿修改模型的文件名，不然会读取不到)。
-  新增 `easy imageSize` 和 `easy imageSizeByLongerSize` 输出的尺寸显示。
-  新增 `easy showSpentTime` 节点用于展示图片推理花费时间与VAE解码花费时间。
- `easy controlnetLoaderADV` 和 `easy controlnetLoader` 新增 `control_net` 可选传入参数
- `easy preSampling` 和 `easy preSamplingAdvanced` 新增 `image_to_latent` 可选传入参数
- `easy a1111Loader` 和 `easy comfyLoader` 新增 `batch_size` 传入参数

-  修改 `easy controlnetLoader` 到 loader 分类底下。
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

### StableDiffusion
#### 文生图

<img src="https://raw.githubusercontent.com/yolain/yolain-comfyui-workflow/main/Workflows/Simple/text_to_image.png">

#### 图生图+controlnet

<img src="https://raw.githubusercontent.com/yolain/yolain-comfyui-workflow/main/Workflows/Simple/image_to_image_controlnet.png">

#### InstantID

<img src="https://raw.githubusercontent.com/yolain/yolain-comfyui-workflow/main/Workflows/Simple/instantID.png">

### StableCascade
#### 文生图
<img src="https://raw.githubusercontent.com/yolain/yolain-comfyui-workflow/main/Workflows/StableCascade/text_to_image.png">

#### 图生图
<img src="https://raw.githubusercontent.com/yolain/yolain-comfyui-workflow/main/Workflows/StableCascade/image_to_image.png">


## Credits

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 功能强大且模块化的Stable Diffusion GUI

[ComfyUI-ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) - ComfyUI管理器

[tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes) - 管道节点（节点束）让用户减少了不必要的连接

[ComfyUI-extensions](https://github.com/diffus3/ComfyUI-extensions) - diffus3的获取与设置点让用户可以分离工作流构成

[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) - 常规整合包1

[ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) - 常规整合包2

[ComfyUI-Logic](https://github.com/theUpsider/ComfyUI-Logic) -  ComfyUI逻辑运算