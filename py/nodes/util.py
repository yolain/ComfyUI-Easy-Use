import os
import re
import folder_paths
import json
from ..libs.utils import AlwaysEqualProxy
from ..libs.text import draw_text, getbbox, has_arabic, reads_right_to_left

class showLoaderSettingsNames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("ckpt_name", "vae_name", "lora_name")

    FUNCTION = "notify"
    OUTPUT_NODE = True

    CATEGORY = "EasyUse/Util"

    def notify(self, pipe, names=None, unique_id=None, extra_pnginfo=None):
        if unique_id and extra_pnginfo and "workflow" in extra_pnginfo:
            workflow = extra_pnginfo["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
            if node:
                ckpt_name = pipe['loader_settings']['ckpt_name'] if 'ckpt_name' in pipe['loader_settings'] else ''
                vae_name = pipe['loader_settings']['vae_name'] if 'vae_name' in pipe['loader_settings'] else ''
                lora_name = pipe['loader_settings']['lora_name'] if 'lora_name' in pipe['loader_settings'] else ''

                if ckpt_name:
                    ckpt_name = os.path.basename(os.path.splitext(ckpt_name)[0])
                if vae_name:
                    vae_name = os.path.basename(os.path.splitext(vae_name)[0])
                if lora_name:
                    lora_name = os.path.basename(os.path.splitext(lora_name)[0])

                names = "ckpt_name: " + ckpt_name + '\n' + "vae_name: " + vae_name + '\n' + "lora_name: " + lora_name
                node["widgets_values"] = names

        return {"ui": {"text": [names]}, "result": (ckpt_name, vae_name, lora_name)}

class sliderControl:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (['ipadapter layer weights'],),
                "model_type": (['sdxl', 'sd1'],),
            },
            "hidden": {
                "prompt": "PROMPT",
                "my_unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("layer_weights",)

    FUNCTION = "control"

    CATEGORY = "EasyUse/Util"

    def control(self, mode, model_type, prompt=None, my_unique_id=None, extra_pnginfo=None):
        values = ''
        if my_unique_id in prompt:
            if 'values' in prompt[my_unique_id]["inputs"]:
                values = prompt[my_unique_id]["inputs"]['values']

        return (values,)

class setCkptName:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = (AlwaysEqualProxy('*'),)
    RETURN_NAMES = ("ckpt_name",)
    FUNCTION = "set_name"
    CATEGORY = "EasyUse/Util"

    def set_name(self, ckpt_name):
        return (ckpt_name,)

class setControlName:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "controlnet_name": (folder_paths.get_filename_list("controlnet"),),
            }
        }

    RETURN_TYPES = (AlwaysEqualProxy('*'),)
    RETURN_NAMES = ("controlnet_name",)
    FUNCTION = "set_name"
    CATEGORY = "EasyUse/Util"

    def set_name(self, controlnet_name):
        return (controlnet_name,)
    
class setLoraName:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
            }
        }

    RETURN_TYPES = (AlwaysEqualProxy('*'),)
    RETURN_NAMES = ("lora_name",)
    FUNCTION = "set_name"
    CATEGORY = "EasyUse/Util"

    def set_name(self, lora_name):
        return (lora_name,)


def _markdown_table_to_image(markdown: str, font_path: str):
    """将 Markdown 表格字符串渲染为 PIL.Image（RGB），支持单元格自动换行。"""
    from PIL import Image, ImageDraw, ImageFont

    # 解析行，过滤分隔行（如 |---|---|）
    lines = [l for l in (markdown or '').strip().splitlines() if l.strip()]
    table_rows = []
    for line in lines:
        if re.match(r'^\|[\s\-:|]+\|$', line.strip()):
            continue
        cells = [re.sub(r'\*\*(.+?)\*\*', lambda m: '\x01' + m.group(1) + '\x02',
                 re.sub(r'<br\s*/?>', '\n', c.strip(), flags=re.IGNORECASE))
                 for c in line.strip().strip('|').split('|')]
        table_rows.append(cells)

    if not table_rows:
        return Image.new("RGB", (400, 80), (255, 255, 255))

    num_cols = max(len(r) for r in table_rows)
    table_rows = [r + [''] * (num_cols - len(r)) for r in table_rows]

    # 表头为阿拉伯文时，表格从右到左排列（第一列在最右侧）
    if reads_right_to_left(' '.join(table_rows[0])):
        table_rows = [r[::-1] for r in table_rows]
    table_has_arabic = any(has_arabic(c) for r in table_rows for c in r)

    # 加载字体
    font_size = 16
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    pad_x, pad_y = 14, 10
    border = 1
    max_cell_text_width = 200  # 单元格文字区域最大宽度（像素）

    def get_text_width(text):
        clean = re.sub('[\x01\x02]', '', text)
        try:
            bbox = getbbox(font, clean)
            return bbox[2] - bbox[0]
        except Exception:
            return len(clean) * 9

    def get_line_height():
        try:
            bbox = font.getbbox('Ag\u4e2d')
            height = bbox[3] - bbox[1]
            if table_has_arabic:
                # 阿拉伯文字母更高（如 أ、ل）并有下伸部分（如 ج）
                arabic = getbbox(font, '\u0623\u0644\u062c')
                height = max(height, arabic[3] - arabic[1])
            return height
        except Exception:
            return font_size + 2

    def wrap_text(text, max_width):
        """换行：先按 \\n 切段，每段再按英文单词边界 / CJK 字符换行。"""
        if not text:
            return ['']
        # 先按显式换行符切段，再对每段分别软换行
        hard_lines = text.split('\n')
        if len(hard_lines) > 1:
            result = []
            for hl in hard_lines:
                result.extend(wrap_text(hl, max_width))
            return result if result else ['']

        # 将文本拆分为：CJK 单字符 / 空白序列 / 非CJK非空白序列（英文单词/标点等）
        tokens = re.findall(
            r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]'
            r'|[ \t]+'
            r'|[^ \t\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+',
            text
        )

        result, current = [], ''
        for token in tokens:
            is_space = token.strip() == ''
            test = current + token
            if get_text_width(test) <= max_width:
                if is_space and not current:
                    continue  # 跳过行首空格
                current = test
            else:
                if is_space:
                    # 空白处换行，丢弃该空白
                    if current:
                        result.append(current)
                    current = ''
                elif get_text_width(token) <= max_width:
                    # 整个 token 能放一行，整体移到下一行
                    if current:
                        result.append(current)
                    current = token
                else:
                    # token 本身超宽（极长单词），逐字符强拆
                    for char in token:
                        if get_text_width(current + char) <= max_width:
                            current += char
                        else:
                            if current:
                                result.append(current)
                            current = char
        if current:
            result.append(current)
        return result if result else ['']

    line_h = get_line_height()

    def parse_line_segments(line):
        """将含 \\x01..\\x02 粗体标记的行拆分为 (text, is_bold) 片段列表。"""
        result, bold = [], False
        for part in re.split('([\x01\x02])', line):
            if part == '\x01':
                bold = True
            elif part == '\x02':
                bold = False
            elif part:
                result.append((part, bold))
        return result or [('', False)]

    def balance_bold_markers(lines):
        """确保每行粗体标记自成一对：跨行时在行首补开、行尾补关标记。"""
        result, in_bold = [], False
        for line in lines:
            if in_bold:
                line = '\x01' + line
            for ch in line:
                if ch == '\x01':   in_bold = True
                elif ch == '\x02': in_bold = False
            if in_bold:
                line = line + '\x02'
            result.append(line)
        return result

    # 第一遍：计算各列宽度（不超过 max_cell_text_width，按子行分别测量）
    col_text_widths = []
    for col_idx in range(num_cols):
        max_w = 0
        for row in table_rows:
            cell = row[col_idx] if col_idx < len(row) else ''
            for seg_line in cell.split('\n'):
                max_w = max(max_w, min(get_text_width(seg_line), max_cell_text_width))
        col_text_widths.append(max_w)
    col_widths = [w + pad_x * 2 for w in col_text_widths]

    # 第二遍：对每行每格换行，计算各行高度
    wrapped_rows = []
    row_heights  = []
    for row in table_rows:
        wrapped_cells = []
        max_lines = 1
        for col_idx in range(num_cols):
            cell = row[col_idx] if col_idx < len(row) else ''
            wrapped = balance_bold_markers(wrap_text(cell, col_text_widths[col_idx]))
            wrapped_cells.append(wrapped)
            max_lines = max(max_lines, len(wrapped))
        wrapped_rows.append(wrapped_cells)
        row_heights.append(max_lines * line_h + pad_y * 2)

    # 每列左边缘 x 坐标（每列前留 1px 边框）
    col_x = [border]
    for cw in col_widths:
        col_x.append(col_x[-1] + cw + border)

    total_width  = col_x[-1]
    total_height = border + sum(rh + border for rh in row_heights)

    # 配色
    header_bg    = (52,  73,  94)
    header_fg    = (255, 255, 255)
    even_bg      = (248, 249, 252)
    odd_bg       = (255, 255, 255)
    border_color = (180, 185, 195)
    text_color   = (50,  54,  62)

    # 以边框色填充整张图，格线自然显现
    img  = Image.new("RGB", (total_width, total_height), border_color)
    draw = ImageDraw.Draw(img)

    # 阿拉伯文行共用一条基线（以 ألج 的顶部对齐），短词不会贴在格子顶部
    try:
        arabic_top = getbbox(font, '\u0623\u0644\u062c')[1] if table_has_arabic else 0
    except Exception:
        arabic_top = 0

    def render_line(x, y, line, fg, width):
        """逐片段渲染一行文字；粗体通过向右偏移 1px 再描一遍来模拟加粗。"""
        if reads_right_to_left(re.sub('[\x01\x02]', '', line)):
            render_line_right_to_left(x + width, y - arabic_top, line, fg)
            return
        try:
            text_offset = -getbbox(font, re.sub('[\x01\x02]', '', line) or 'A')[1]
        except Exception:
            text_offset = 0
        sx = x
        for seg, is_bold in parse_line_segments(line):
            draw_text(draw, (sx, y + text_offset), seg, font=font, fill=fg)
            if is_bold:
                draw_text(draw, (sx + 1, y + text_offset), seg, font=font, fill=fg)
            try:
                bbox = getbbox(font, seg)
                w = bbox[2] - bbox[0]
            except Exception:
                w = len(seg) * 9
            sx += w + (1 if is_bold else 0)

    def render_line_right_to_left(right, y, line, fg):
        """阿拉伯文行：右对齐，片段从右向左排列；片段首尾的空格单独计宽。"""
        try:
            space = font.getlength(' ')
        except Exception:
            space = 4
        sx = right
        for seg, is_bold in parse_line_segments(line):
            core = seg.strip(' ')
            if not core:
                sx -= len(seg) * space
                continue
            sx -= (len(seg) - len(seg.lstrip(' '))) * space
            left, _, ink_right, _ = getbbox(font, core)
            sx -= ink_right - left + (1 if is_bold else 0)
            draw_text(draw, (sx - left, y), core, font=font, fill=fg)
            if is_bold:
                draw_text(draw, (sx - left + 1, y), core, font=font, fill=fg)
            sx -= (len(seg) - len(seg.rstrip(' '))) * space

    row_y = border
    for row_idx, (wrapped_cells, rh) in enumerate(zip(wrapped_rows, row_heights)):
        is_header = row_idx == 0
        bg = header_bg if is_header else (odd_bg if row_idx % 2 == 1 else even_bg)
        fg = header_fg if is_header else text_color

        for col_idx in range(num_cols):
            cx, cw = col_x[col_idx], col_widths[col_idx]
            # 填充单元格背景
            draw.rectangle([cx, row_y, cx + cw - 1, row_y + rh - 1], fill=bg)

            cell_lines     = wrapped_cells[col_idx] if col_idx < len(wrapped_cells) else ['']
            total_text_h   = len(cell_lines) * line_h
            ty             = row_y + (rh - total_text_h) // 2  # 垂直居中起点
            for line_text in cell_lines:
                render_line(cx + pad_x, ty, line_text, fg, cw - pad_x * 2)
                ty += line_h

        row_y += rh + border

    # 最长边不小于 1280，等比放大
    min_long_side = 1280
    long_side = max(img.width, img.height)
    if long_side < min_long_side:
        scale = min_long_side / long_side
        new_w = round(img.width  * scale)
        new_h = round(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img


class tableEditor:
    """表格编辑器节点 —— 通过可视化表格或 Markdown 语法编辑数据，输出 Markdown 字符串。"""

    CATEGORY = "EasyUse/Util"

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("markdown", "image")
    FUNCTION = "execute"

    DESCRIPTION = "通过可视化表格或 Markdown 语法编辑数据，输出 Markdown 格式的表格字符串。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "table_data": ("EASY_TABLE_EDITOR",),
            },
        }

    def execute(self, table_data):
        # 表格数据可能是纯 Markdown 字符串，也可能是序列化后的 JSON
        if isinstance(table_data, str) and table_data.strip().startswith('{'):
            try:
                obj = json.loads(table_data)
                markdown = obj.get('markdown', '')
                if not markdown:
                    # 重新从 headers/rows 生成
                    headers = obj.get('headers', [])
                    rows = obj.get('rows', [])
                    col_widths = [max(len(str(h)), 3) for h in headers]
                    for row in rows:
                        for i, cell in enumerate(row):
                            if i < len(col_widths):
                                col_widths[i] = max(col_widths[i], len(str(cell)))
                    header_line = '| ' + ' | '.join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers)) + ' |'
                    sep_line    = '| ' + ' | '.join('-' * w for w in col_widths) + ' |'
                    row_lines   = [
                        '| ' + ' | '.join(str(row[i] if i < len(row) else '').ljust(col_widths[i]) for i in range(len(headers))) + ' |'
                        for row in rows
                    ]
                    markdown = '\n'.join([header_line, sep_line] + row_lines)
            except Exception:
                markdown = table_data
        else:
            markdown = table_data

        # 将 Markdown 表格渲染为图像
        font_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'resources', 'wenquan.ttf'
        )
        from ..libs.image import pil2tensor
        img_tensor = pil2tensor(_markdown_table_to_image(markdown, font_path).convert("RGB"))

        return (markdown, img_tensor)


NODE_CLASS_MAPPINGS = {
    "easy showLoaderSettingsNames": showLoaderSettingsNames,
    "easy sliderControl": sliderControl,
    "easy ckptNames": setCkptName,
    "easy controlnetNames": setControlName,
    "easy loraNames": setLoraName,
    "easy tableEditor": tableEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy showLoaderSettingsNames": "Show Loader Settings Names",
    "easy sliderControl": "Easy Slider Control",
    "easy ckptNames": "Ckpt Names",
    "easy controlnetNames": "ControlNet Names",
    "easy loraNames": "Lora Names",
    "easy tableEditor": "Table Editor",
}
