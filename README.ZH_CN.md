![comfyui-easy-use](https://github.com/user-attachments/assets/9b7a5e44-f5e2-4c27-aed2-d0e6b50c46bb)

<div align="center">
<a href="https://space.bilibili.com/1840885116">视频介绍</a> |
<a href="https://docs.easyuse.yolain.com">文档</a> | 
<a href="https://github.com/yolain/ComfyUI-Yolain-Workflows">工作流合集</a> |
<a href="#%EF%B8%8F-donation">捐助</a> 
<br><br>
<a href="./README.md"><img src="https://img.shields.io/badge/🇬🇧English-e9e9e9"></a>
<a href="./README.ZH_CN.md"><img src="https://img.shields.io/badge/🇨🇳中文简体-0b8cf5"></a>
</div>

**ComfyUI-Easy-Use** 是一个化繁为简的节点整合包, 在 [tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes) 的基础上进行延展，并针对了诸多主流的节点包做了整合与优化，以达到更快更方便使用ComfyUI的目的，在保证自由度的同时还原了本属于Stable Diffusion的极致畅快出图体验。

## 👨🏻‍🎨 特色介绍

- 沿用了 [tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes) 的思路，大大减少了折腾工作流的时间成本。
- UI界面美化，首次安装的用户，如需使用UI主题，请在 Settings -> Color Palette 中自行切换主题并**刷新页面**即可
- 增加了预采样参数配置的节点，可与采样节点分离，更方便预览。
- 支持通配符与Lora的提示词节点，如需使用Lora Block Weight用法，需先保证自定义节点包中安装了 [ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack)
- 可多选的风格化提示词选择器，默认是Fooocus的样式json，可自定义json放在styles底下，samples文件夹里可放预览图(名称和name一致,图片文件名如有空格需转为下划线'_')
- 加载器可开启A1111提示词风格模式，可重现与webui生成近乎相同的图像，需先安装 [ComfyUI_smZNodes](https://github.com/shiimizu/ComfyUI_smZNodes)
- 可使用`easy latentNoisy`或`easy preSamplingNoiseIn`节点实现对潜空间的噪声注入
- 简化 SD1.x、SD2.x、SDXL、SVD、Zero123等流程 
- 简化 Stable Cascade [示例参考](https://github.com/yolain/ComfyUI-Yolain-Workflows?tab=readme-ov-file#1-13-stable-cascade)
- 简化 Layer Diffuse [示例参考](https://github.com/yolain/ComfyUI-Yolain-Workflows?tab=readme-ov-file#2-3-layerdiffusion)
- 简化 InstantID [示例参考](https://github.com/yolain/ComfyUI-Yolain-Workflows?tab=readme-ov-file#2-2-instantid), 需先保证自定义节点包中安装了 [ComfyUI_InstantID](https://github.com/cubiq/ComfyUI_InstantID)
- 简化 IPAdapter, 需先保证自定义节点包中安装最新版v2的 [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)
- 扩展 XYplot 的可用性
- 整合了Fooocus Inpaint功能
- 整合了常用的逻辑计算、转换类型、展示所有类型等
- 支持节点上checkpoint、lora模型子目录分类及预览图 (请在设置中开启上下文菜单嵌套子目录)
- 支持BriaAI的RMBG-1.4模型的背景去除节点，[技术参考](https://huggingface.co/briaai/RMBG-1.4)
- 支持 强制清理comfyUI模型显存占用
- 支持Stable Diffusion 3 多账号API节点
- 支持IC-Light的应用 [示例参考](https://github.com/yolain/ComfyUI-Yolain-Workflows?tab=readme-ov-file#2-5-ic-light) | [代码整合来源](https://github.com/huchenlei/ComfyUI-IC-Light) | [技术参考](https://github.com/lllyasviel/IC-Light)
- 中文提示词自动识别，使用[opus-mt-zh-en模型](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)
- 支持 sd3 模型
- 支持 kolors 模型
- 支持 flux 模型
- 支持 惰性条件判断（ifElse）和 for循环

## 👨🏻‍🔧 安装

1. 将存储库克隆到 **custom_nodes** 目录并安装依赖
```shell
#1. git下载
git clone https://github.com/yolain/ComfyUI-Easy-Use
#2. 安装依赖
双击install.bat安装依赖
```

## 📜 更新日志

**v1.2.8**

- 修复前端提示框丢失
- 增加在循环中当RAM超出80%时自动清理 [使用方案](https://github.com/LAOGOU-666/Comfyui-Memory_Cleanup)
- 修复了一些BUG (😹)
- 增加了多语言目录

**v1.2.7**

- 优化管理节点组显示
- 在 `easy imageRemBg` 上添加 `ben2`
- 添加 joyCaption2 API版节点（ https://github.com/siliconflow/BizyAir ）
- 使用一种新的方式在 loader 中显示模型缩略图（支持 diffusion_models、lors、checkpoints）

**v1.2.6**

- 修复了在缺少自定义节点时缺少 “红色框框” 样式的问题。
- 在一些简单的加载器中，将 `clip_skip` 的默认值从 `-1` 调整为 `-2`。
- 修复因设置节点中缺少相连接的自定义节点而导致弄乱画布的问题
- 修复 'easy imageChooser' 不能循环使用的问题。

**v1.2.5**

- 在 `easy preSamplingCustom` 和 `easy preSamplingAdvanced` 上增加 `enable (GPU=A1111)` 噪波生成模式选择项
- 增加 `easy makeImageForICLora`
- 在 `easy ipadapterApply` 添加 `REGULAR - FLUX and SD3.5 only (high strength)` 预置项以支持 InstantX Flux ipadapter 
- 修复brushnet 无法在 `--fast` 模式下使用 
- 支持briaai RMBG-2.0
- 支持mochi模型
- 实现在循环主体中重复使用终端节点输出（例如预览图像和显示任何内容等输出节点...）

**v1.2.4**

- 增加 `easy imageSplitTiles` and `easy imageTilesFromBatch` - 图像分块
- 支持 `model_override`,`vae_override`,`clip_override` 可以在 `easy fullLoader` 中单独输入
- 增加 `easy saveImageLazy`
- 增加 `easy loadImageForLoop`
- 增加 `easy isFileExist`
- 增加 `easy saveText`

**v1.2.3**

- `easy showAnything` 和 `easy cleanGPUUsed` 增加输出插槽
- 添加新的人体分割在 `easy humanSegmentation` 节点上 - 代码从 [ComfyUI_Human_Parts](https://github.com/metal3d/ComfyUI_Human_Parts) 整合
- 当你在 `easy preSamplingCustom` 节点上选择basicGuider，CFG>0 且当前模型为Flux时，将使用FluxGuidance
- 增加 `easy loraStackApply` and `easy controlnetStackApply`

**v1.2.2**

- 增加 `easy batchAny`
- 增加 `easy anythingIndexSwitch`
- 增加 `easy forLoopStart` 和 `easy forLoopEnd`  
- 增加 `easy ifElse`
- 增加 v2 版本新前端代码
- 增加 `easy fluxLoader`
- 增加 `controlnetApply` 相关节点对sd3和hunyuanDiT的支持
- 修复 当使用fooocus inpaint后，再使用Lora模型无法生效的问题

**v1.2.1**

- 增加 `easy ipadapterApplyFaceIDKolors`
- `easy ipadapterApply` 和 `easy ipadapterApplyADV` 增加 **PLUS (kolors genernal)** 和 **FACEID PLUS KOLORS** 预置项
- `easy imageRemBg` 增加 **inspyrenet** 选项
- 增加 `easy controlnetLoader++`
- 去除 `easy positive` `easy negative` 等prompt节点的自动将中文翻译功能，自动翻译仅在 `easy a1111Loader` 等不支持中文TE的加载器中生效
- 增加 `easy kolorsLoader` - 可灵加载器，参考了 [MinusZoneAI](https://github.com/MinusZoneAI/ComfyUI-Kolors-MZ) 和 [kijai](https://github.com/kijai/ComfyUI-KwaiKolorsWrapper) 的代码。

**v1.2.0**

- 增加 `easy pulIDApply` 和 `easy pulIDApplyADV`
- 增加 `easy hunyuanDiTLoader` 和 `easy pixArtLoader`
- 当新菜单的位置在上或者下时增加上 crystools 的显示，推荐开两个就好（如果后续crystools有更新UI适配我可能会删除掉）
- 增加 **easy sliderControl** - 滑块控制节点，当前可用于控制ipadapterMS的参数 (双击滑块可重置为默认值)
- 增加 **layer_weights** 属性在 `easy ipadapterApplyADV` 节点

**v1.1.9**

- 增加 新的调度器 **gitsScheduler**
- 增加 `easy imageBatchToImageList` 和 `easy imageListToImageBatch` (修复Impact版的一点小问题)
- 递归模型子目录嵌套
- 支持 sd3 模型 
- 增加 `easy applyInpaint` - 局部重绘全模式节点 (相比与之前的kSamplerInpating节点逻辑会更合理些)

**v1.1.8**

- 增加中文提示词自动翻译，使用[opus-mt-zh-en模型](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en), 默认已对wildcard、lora正则处理, 其他需要保留的中文，可使用`@你的提示词@`包裹 (若依赖安装完成后报错, 请重启)，测算大约会占0.3GB显存
- 增加 `easy controlnetStack` - controlnet堆
- 增加 `easy applyBrushNet` - [示例参考](https://github.com/yolain/ComfyUI-Yolain-Workflows/blob/main/workflows/2_advanced/2-4inpainting/2-4brushnet_1.1.8.json)
- 增加 `easy applyPowerPaint` - [示例参考](https://github.com/yolain/ComfyUI-Yolain-Workflows/blob/main/workflows/2_advanced/2-4inpainting/2-4powerpaint_outpaint_1.1.8.json)

**v1.1.7**

- 修复 一些模型(如controlnet模型等)未成功写入缓存，导致修改前置节点束参数（如提示词）需要二次载入模型的问题
- 增加 `easy prompt` - 主体和光影预置项，后期可能会调整
- 增加 `easy icLightApply` - 重绘光影, 从[ComfyUI-IC-Light](https://github.com/huchenlei/ComfyUI-IC-Light)优化
- 增加 `easy imageSplitGrid` - 图像网格拆分
- `easy kSamplerInpainting` 的 **additional** 属性增加差异扩散和brushnet等相关选项 
- 增加 brushnet模型加载的支持 - [ComfyUI-BrushNet](https://github.com/nullquant/ComfyUI-BrushNet)
- 增加 `easy applyFooocusInpaint` - Fooocus内补节点 替代原有的 FooocusInpaintLoader
- 移除 `easy fooocusInpaintLoader` - 容易bug，不再使用
- 修改 easy kSampler等采样器中并联的model 不再替换输出中pipe里的model

**v1.1.6**

- 增加步调齐整适配 - 在所有的预采样和全采样器节点中的 调度器(schedulder) 增加了 **alignYourSteps** 选项
- `easy kSampler` 和 `easy fullkSampler` 的 **image_output** 增加 **Preview&Choose**选项
- 增加 `easy styleAlignedBatchAlign` - 风格对齐 [style_aligned_comfy](https://github.com/brianfitzgerald/style_aligned_comfy)
- 增加 `easy ckptNames`
- 增加 `easy controlnetNames`
- 增加 `easy imagesSplitimage` - 批次图像拆分单张
- 增加 `easy imageCount` - 图像数量
- 增加 `easy textSwitch` - 文字切换

<details>
<summary><b>v1.1.5</b></summary>

- 重写 `easy cleanGPUUsed` - 可强制清理comfyUI的模型显存占用
- 增加 `easy humanSegmentation` - 多类分割、人像分割
- 增加 `easy imageColorMatch`
- 增加 `easy ipadapterApplyRegional`
- 增加 `easy ipadapterApplyFromParams`
- 增加 `easy imageInterrogator` - 图像反推
- 增加 `easy stableDiffusion3API` - 简易的Stable Diffusion 3 多账号API节点
</details>

<details>
<summary><b>v1.1.4</b></summary>

- 增加 `easy imageChooser` - 从[cg-image-picker](https://github.com/chrisgoringe/cg-image-picker)简化的图片选择器
- 增加 `easy preSamplingCustom` - 自定义预采样，可支持cosXL-edit
- 增加 `easy ipadapterStyleComposition`
- 增加 在Loaders上右键菜单可查看 checkpoints、lora 信息
- 修复 `easy preSamplingNoiseIn`、`easy latentNoisy`、`east Unsampler` 以兼容ComfyUI Revision>=2098 [0542088e] 以上版本
- 修复 FooocusInpaint修改ModelPatcher计算权重引发的问题，理应在生成model后重置ModelPatcher为默认值
</details>

<details>
<summary><b>v1.1.3</b></summary>

- `easy ipadapterApply` 增加 **COMPOSITION** 预置项
- 增加 对[ResAdapter](https://huggingface.co/jiaxiangc/res-adapter) lora模型 的加载支持
- 增加 `easy promptLine`
- 增加 `easy promptReplace`
- 增加 `easy promptConcat`
- `easy wildcards` 增加 **multiline_mode**属性 
- 增加 当节点需要下载模型时，若huggingface连接超时，会切换至镜像地址下载模型
</details>

<details>
<summary><b>v1.1.2</b></summary>

- 改写 EasyUse 相关节点的部分插槽推荐节点
- 增加 **启用上下文菜单自动嵌套子目录** 设置项，默认为启用状态，可分类子目录及checkpoints、loras预览图
- 增加 `easy sv3dLoader` 
- 增加 `easy dynamiCrafterLoader` 
- 增加 `easy ipadapterApply`
- 增加 `easy ipadapterApplyADV`
- 增加 `easy ipadapterApplyEncoder`
- 增加 `easy ipadapterApplyEmbeds`
- 增加 `easy preMaskDetailerFix`
- `easy kSamplerInpainting` 增加 **additional** 属性，可设置成 Differential Diffusion 或 Only InpaintModelConditioning
- 修复 `easy stylesSelector` 当未选择样式时，原有提示词发生了变化
- 修复 `easy pipeEdit` 提示词输入lora时报错
- 修复 layerDiffuse xyplot相关bug
</details>

<details>
<summary><b>v1.1.1</b></summary>

- 修复首次添加含seed的节点且当前模式为control_before_generate时，seed为0的问题
- `easy preSamplingAdvanced` 增加 **return_with_leftover_noise**
- 修复 `easy stylesSelector` 当选择自定义样式文件时运行队列报错
- `easy preSamplingLayerDiffusion` 增加 mask 可选传入参数
- 将所有 **seed_num** 调整回 **seed**
- 修补官方BUG: 当control_mode为before 在首次加载页面时未修改节点中widget名称为 control_before_generate
- 去除强制**control_before_generate**设定
- 增加 `easy imageRemBg` - 默认为BriaAI的RMBG-1.4模型, 移除背景效果更加，速度更快
</details>

<details>
<summary><b>v1.1.0</b></summary>

- 增加 `easy imageSplitList` - 拆分每 N 张图像
- 增加 `easy preSamplingDiffusionADDTL` - 可配置前景、背景、blended的additional_prompt等   
- 增加 `easy preSamplingNoiseIn` 可替代需要前置的`easy latentNoisy`节点 实现效果更好的噪声注入
- `easy pipeEdit` 增加 条件拼接模式选择，可选择替换、合并、联结、平均、设置条件时间
- 增加 `easy pipeEdit` - 可编辑Pipe的节点（包含可重新输入提示词）
- 增加 `easy preSamplingLayerDiffusion` 与 `easy kSamplerLayerDiffusion` （连接 `easy kSampler` 也能通）
- 增加 在 加载器、预采样、采样器、Controlnet等节点上右键可快速替换同类型节点的便捷菜单
- 增加 `easy instantIDApplyADV` 可连入 positive 与 negative
- 修复 `easy wildcards` 读取lora未填写完整路径时未自动检索导致加载lora失败的问题
- 修复 `easy instantIDApply` mask 未传入正确值
- 修复 在 非a1111提示词风格下 BREAK 不生效的问题
</details>

<details>
<summary><b>v1.0.9</b></summary>

- 修复未安装 ComfyUI-Impack-Pack 和 ComfyUI_InstantID 时报错
- 修复 `easy pipeIn` - pipe设为可不必选
- 增加 `easy instantIDApply` - 需要先安装 [ComfyUI_InstantID](https://github.com/cubiq/ComfyUI_InstantID), 工作流参考[示例](https://github.com/yolain/ComfyUI-Yolain-Workflows?tab=readme-ov-file#2-2-instantid)
- 修复 `easy detailerFix` 未添加到保存图片格式化扩展名可用节点列表
- 修复 `easy XYInputs: PromptSR` 在替换负面提示词时报错
</details>

<details>
<summary><b>v1.0.8</b></summary>

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
</details>

<details>
<summary><b>v1.0.7</b></summary>

- 增加 `easy cascadeLoader` - stable cascade 加载器
- 增加 `easy preSamplingCascade` - stabled cascade stage_c 预采样参数
- 增加 `easy fullCascadeKSampler` - stable cascade stage_c 完整版采样器
- 增加 `easy cascadeKSampler` - stable cascade stage-c ksampler simple
</details>

<details>
<summary><b>v1.0.6</b></summary>

- 增加 `easy XYInputs: Checkpoint`
- 增加 `easy XYInputs: Lora`
- `easy seed` 增加固定种子值时可手动切换随机种
- 修复 `easy fullLoader`等加载器切换lora时自动调整节点大小的问题
- 去除原有ttn的图片保存逻辑并适配ComfyUI默认的图片保存格式化扩展
</details>

<details>
<summary><b>v1.0.5</b></summary>

- 增加 `easy isSDXL` 
- `easy svdLoader` 增加提示词控制, 可配合open_clip模型进行使用
- `easy wildcards` 增加 **populated_text** 可输出通配填充后文本
</details>

<details>
<summary><b>v1.0.4</b></summary>

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
</details>

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

## 整合参考到的相关节点包

声明: 非常尊重这些原作者们的付出，开源不易，我仅仅只是做了一些整合与优化。

| 节点名 (搜索名)                      | 相关的库                                                                        | 库相关的节点                  |
|:-------------------------------|:----------------------------------------------------------------------------|:------------------------|
| easy setNode                   | [ComfyUI-extensions](https://github.com/diffus3/ComfyUI-extensions) | diffus3.SetNode         |
| easy getNode                   | [ComfyUI-extensions](https://github.com/diffus3/ComfyUI-extensions) | diffus3.GetNode         |
| easy bookmark                  | [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) | Bookmark 🔖             |
| easy portraitMarker            | [comfyui-portrait-master](https://github.com/florestefano1975/comfyui-portrait-master) | Portrait Master         |
| easy LLLiteLoader              | [ControlNet-LLLite-ComfyUI](https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI) | LLLiteLoader            |
| easy globalSeed                | [ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) | Global Seed (Inspire)   | 
| easy preSamplingDynamicCFG     | [sd-dynamic-thresholding](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding) | DynamicThresholdingFull | 
| dynamicThresholdingFull        | [sd-dynamic-thresholding](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding) | DynamicThresholdingFull | 
| easy imageInsetCrop            | [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) | ImageInsetCrop          | 
| easy poseEditor                | [ComfyUI_Custom_Nodes_AlekPet](https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet) | poseNode                | 
| easy if                        | [ComfyUI-Logic](https://github.com/theUpsider/ComfyUI-Logic) | IfExecute               | 
| easy preSamplingLayerDiffusion | [ComfyUI-layerdiffusion](https://github.com/huchenlei/ComfyUI-layerdiffusion) | LayeredDiffusionApply等  | 
| easy dynamiCrafterLoader       | [ComfyUI-layerdiffusion](https://github.com/ExponentialML/ComfyUI_Native_DynamiCrafter) | Apply Dynamicrafter     | 
| easy imageChooser              | [cg-image-picker](https://github.com/chrisgoringe/cg-image-picker) | Preview Chooser         | 
| easy styleAlignedBatchAlign    | [style_aligned_comfy](https://github.com/chrisgoringe/cg-image-picker) | styleAlignedBatchAlign  | 
| easy icLightApply              | [ComfyUI-IC-Light](https://github.com/huchenlei/ComfyUI-IC-Light) | ICLightApply等           |
| easy kolorsLoader              | [ComfyUI-Kolors-MZ](https://github.com/MinusZoneAI/ComfyUI-Kolors-MZ) | kolorsLoader            |

## Credits

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 功能强大且模块化的Stable Diffusion GUI

[ComfyUI-ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) - ComfyUI管理器

[tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes) - 管道节点（节点束）让用户减少了不必要的连接

[ComfyUI-extensions](https://github.com/diffus3/ComfyUI-extensions) - diffus3的获取与设置点让用户可以分离工作流构成

[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) - 常规整合包1

[ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) - 常规整合包2

[ComfyUI-Logic](https://github.com/theUpsider/ComfyUI-Logic) -  ComfyUI逻辑运算

[ComfyUI-ResAdapter](https://github.com/jiaxiangc/ComfyUI-ResAdapter) - 让模型生成不受训练分辨率限制

[ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) - 风格迁移

[ComfyUI_InstantID](https://github.com/cubiq/ComfyUI_InstantID) - 人脸迁移

[ComfyUI_PuLID](https://github.com/cubiq/PuLID_ComfyUI) - 人脸迁移

[ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) - pyssss 小蛇🐍脚本

[cg-image-picker](https://github.com/chrisgoringe/cg-image-picker) - 图片选择器

[ComfyUI-BrushNet](https://github.com/nullquant/ComfyUI-BrushNet) - BrushNet 内补节点

[ComfyUI_ExtraModels](https://github.com/city96/ComfyUI_ExtraModels) - DiT架构相关节点（Pixart、混元DiT等）

## 免责声明

本开源项目及其内容按 “原样 ”提供，不作任何明示或暗示的保证，包括但不限于适销性、特定用途适用性和非侵权保证。在任何情况下，作者或其他版权所有者均不对因本软件或本软件的使用或其他交易而产生、引起或与之相关的任何索赔、损害或其他责任承担责任，无论是合同诉讼、侵权诉讼还是其他诉讼。

用户应自行负责确保在使用本软件或发布由本软件生成的内容时，遵守所在司法管辖区的所有适用法律和法规。作者和版权所有者不对用户在其各自所在地违反法律或法规的行为负责。

## ☕️ 投喂

**Comfyui-Easy-Use** 是一个 GPL 许可的开源项目。为了项目取得更好、可持续的发展，我希望能够获得更多的支持。 如果我的自定义节点为您的一天增添了价值，请考虑喝杯咖啡来进一步补充能量！ 💖感谢您的支持，每一杯咖啡都是我创作的动力！

- [BiliBili充电](https://space.bilibili.com/1840885116)
- [爱发电](https://afdian.com/a/yolain)
- [Wechat/Alipay](https://github.com/user-attachments/assets/803469bd-ed6a-4fab-932d-50e5088a2d03)

感谢您的捐助，我将用这些费用来租用 GPU 或购买其他 GPT 服务，以便更好地调试和完善 ComfyUI-Easy-Use 功能

##  🌟大富大贵的人儿

我对那些慷慨的赐予一颗星的人表示感谢。非常感谢您的支持！

[![Stargazers repo roster for @yolain/ComfyUI-Easy-Use](https://reporoster.com/stars/yolain/ComfyUI-Easy-Use)](https://github.com/yolain/ComfyUI-Easy-Use/stargazers)