import {getLocale} from './utils.js'
const locale = getLocale()

const zhCN = {
    // ExtraMenu
    "💎 View Checkpoint Info...": "💎 查看 Checkpoint 信息...",
    "💎 View Lora Info...": "💎 查看 Lora 信息...",
    "🔃 Reload Node": "🔃 刷新节点",
    // ModelInfo
    "Updated At:": "最近更新：",
    "Created At:": "首次发布：",
    "Details": "详情",
    "Download": "下载量",
    "Source": "来源",
    // GroupMap
    "Groups Map (EasyUse)": "管理组 (EasyUse)",
    "Always": "启用中",
    "Bypass": "已忽略",
    "Never": "已停用",
    "Auto Sorting": "自动排序",
    "Toggle `Show/Hide` can set mode of group, LongPress can set group nodes to never": "点击`启用中/已忽略`可设置组模式, 长按可停用该组节点",
    // Quick
    "Enable ALT+1~9 to paste nodes from nodes template (ComfyUI-Easy-Use)": "启用ALT1~9从节点模板粘贴到工作流 (ComfyUI-Easy-Use)",
    "Enable process bar in queue button (ComfyUI-Easy-Use)": "启用提示词队列进度显示条 (ComfyUI-Easy-Use）",
    "Enable ContextMenu Auto Nest Subdirectories (ComfyUI-Easy-Use)": "启用上下文菜单自动嵌套子目录 (ComfyUI-Easy-Use)"
}

export const $t = (key) => {
    const cn = zhCN[key]
    return locale === 'zh-CN' && cn ? cn : key
}