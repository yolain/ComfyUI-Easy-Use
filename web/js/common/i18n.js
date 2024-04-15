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
    "✏️ Edit": "✏️ 编辑",
    "💾 Save": "💾 保存",
    "No notes": "当前还没有备注内容",
    "Saving Notes...": "正在保存备注...",
    "Type your notes here":"在这里输入备注内容",
    "Notes": "备注",
    "Type": "类型",
    "Trained Words": "训练词",
    "BaseModel": "基础算法",
    "Details": "详情",
    "Download": "下载量",
    "Source": "来源",
    "Saving Preview...": "正在保存预览图...",
    "Saving Succeed":"保存成功",
    "Saving Failed":"保存失败",
    "No COMBO link": "沒有找到COMBO连接",
    "Reboot ComfyUI":"重启ComfyUI",
    "Are you sure you'd like to reboot the server?": "是否要重启ComfyUI？",
    // GroupMap
    "Groups Map (EasyUse)": "管理组 (EasyUse)",
    "Reboot ComfyUI (EasyUse)": "重启服务 (EasyUse)",
    "Always": "启用中",
    "Bypass": "已忽略",
    "Never": "已停用",
    "Auto Sorting": "自动排序",
    "Toggle `Show/Hide` can set mode of group, LongPress can set group nodes to never": "点击`启用中/已忽略`可设置组模式, 长按可停用该组节点",
    // Quick
    "Enable ALT+1~9 to paste nodes from nodes template (ComfyUI-Easy-Use)": "启用ALT1~9从节点模板粘贴到工作流 (ComfyUI-Easy-Use)",
    "Enable process bar in queue button (ComfyUI-Easy-Use)": "启用提示词队列进度显示条 (ComfyUI-Easy-Use）",
    "Enable ContextMenu Auto Nest Subdirectories (ComfyUI-Easy-Use)": "启用上下文菜单自动嵌套子目录 (ComfyUI-Easy-Use)",
    "Too many thumbnails, have closed the display": "模型缩略图太多啦，为您关闭了显示"
}

export const $t = (key) => {
    const cn = zhCN[key]
    return locale === 'zh-CN' && cn ? cn : key
}