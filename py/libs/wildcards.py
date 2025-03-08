import re
import random
import os
import folder_paths
import yaml
import json
from .log import log_node_info

easy_wildcard_dict = {}

def get_wildcard_list():
    return [f"__{x}__" for x in easy_wildcard_dict.keys()]

def wildcard_normalize(x):
    return x.replace("\\", "/").lower()

def read_wildcard(k, v):
    if isinstance(v, list):
        k = wildcard_normalize(k)
        easy_wildcard_dict[k] = v
    elif isinstance(v, dict):
        for k2, v2 in v.items():
            new_key = f"{k}/{k2}"
            new_key = wildcard_normalize(new_key)
            read_wildcard(new_key, v2)

def read_wildcard_dict(wildcard_path):
    global easy_wildcard_dict
    for root, directories, files in os.walk(wildcard_path, followlinks=True):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, wildcard_path)
                key = os.path.splitext(rel_path)[0].replace('\\', '/').lower()

                try:
                    with open(file_path, 'r', encoding="UTF-8", errors="ignore") as f:
                        lines = f.read().splitlines()
                        easy_wildcard_dict[key] = lines
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding="ISO-8859-1") as f:
                        lines = f.read().splitlines()
                        easy_wildcard_dict[key] = lines
            elif file.endswith('.yaml'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    yaml_data = yaml.load(f, Loader=yaml.FullLoader)

                    for k, v in yaml_data.items():
                        read_wildcard(k, v)
            elif file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        json_data = json.load(f)
                        for key, value in json_data.items():
                            key = wildcard_normalize(key)
                            easy_wildcard_dict[key] = value
                except ValueError:
                    print('json files load error')
    return easy_wildcard_dict


def process(text, seed=None):

    if seed is not None:
        random.seed(seed)

    def replace_options(string):
        replacements_found = False

        def replace_option(match):
            nonlocal replacements_found
            options = match.group(1).split('|')

            multi_select_pattern = options[0].split('$$')
            select_range = None
            select_sep = ' '
            range_pattern = r'(\d+)(-(\d+))?'
            range_pattern2 = r'-(\d+)'

            if len(multi_select_pattern) > 1:
                r = re.match(range_pattern, options[0])

                if r is None:
                    r = re.match(range_pattern2, options[0])
                    a = '1'
                    b = r.group(1).strip()
                else:
                    a = r.group(1).strip()
                    b = r.group(3).strip()

                if r is not None:
                    if b is not None and is_numeric_string(a) and is_numeric_string(b):
                        # PATTERN: num1-num2
                        select_range = int(a), int(b)
                    elif is_numeric_string(a):
                        # PATTERN: num
                        x = int(a)
                        select_range = (x, x)

                    if select_range is not None and len(multi_select_pattern) == 2:
                        # PATTERN: count$$
                        options[0] = multi_select_pattern[1]
                    elif select_range is not None and len(multi_select_pattern) == 3:
                        # PATTERN: count$$ sep $$
                        select_sep = multi_select_pattern[1]
                        options[0] = multi_select_pattern[2]

            adjusted_probabilities = []

            total_prob = 0

            for option in options:
                parts = option.split('::', 1)
                if len(parts) == 2 and is_numeric_string(parts[0].strip()):
                    config_value = float(parts[0].strip())
                else:
                    config_value = 1  # Default value if no configuration is provided

                adjusted_probabilities.append(config_value)
                total_prob += config_value

            normalized_probabilities = [prob / total_prob for prob in adjusted_probabilities]

            if select_range is None:
                select_count = 1
            else:
                select_count = random.randint(select_range[0], select_range[1])

            if select_count > len(options):
                selected_items = options
            else:
                selected_items = random.choices(options, weights=normalized_probabilities, k=select_count)
                selected_items = set(selected_items)

                try_count = 0
                while len(selected_items) < select_count and try_count < 10:
                    remaining_count = select_count - len(selected_items)
                    additional_items = random.choices(options, weights=normalized_probabilities, k=remaining_count)
                    selected_items |= set(additional_items)
                    try_count += 1

            selected_items2 = [re.sub(r'^\s*[0-9.]+::', '', x, 1) for x in selected_items]
            replacement = select_sep.join(selected_items2)
            if '::' in replacement:
                pass

            replacements_found = True
            return replacement

        pattern = r'{([^{}]*?)}'
        replaced_string = re.sub(pattern, replace_option, string)

        return replaced_string, replacements_found

    def replace_wildcard(string):
        global easy_wildcard_dict
        pattern = r"__([\w\s.\-+/*\\]+?)__"
        matches = re.findall(pattern, string)
        replacements_found = False

        for match in matches:
            keyword = match.lower()
            keyword = wildcard_normalize(keyword)
            if keyword in easy_wildcard_dict:
                replacement = random.choice(easy_wildcard_dict[keyword])
                replacements_found = True
                string = string.replace(f"__{match}__", replacement, 1)
            elif '*' in keyword:
                subpattern = keyword.replace('*', '.*').replace('+', r'\+')
                total_patterns = []
                found = False
                for k, v in easy_wildcard_dict.items():
                    if re.match(subpattern, k) is not None:
                        total_patterns += v
                        found = True

                if found:
                    replacement = random.choice(total_patterns)
                    replacements_found = True
                    string = string.replace(f"__{match}__", replacement, 1)
            elif '/' not in keyword:
                string_fallback = string.replace(f"__{match}__", f"__*/{match}__", 1)
                string, replacements_found = replace_wildcard(string_fallback)

        return string, replacements_found

    replace_depth = 100
    stop_unwrap = False
    while not stop_unwrap and replace_depth > 1:
        replace_depth -= 1  # prevent infinite loop

        # pass1: replace options
        pass1, is_replaced1 = replace_options(text)

        while is_replaced1:
            pass1, is_replaced1 = replace_options(pass1)

        # pass2: replace wildcards
        text, is_replaced2 = replace_wildcard(pass1)
        stop_unwrap = not is_replaced1 and not is_replaced2

    return text


def is_numeric_string(input_str):
    return re.match(r'^-?\d+(\.\d+)?$', input_str) is not None


def safe_float(x):
    if is_numeric_string(x):
        return float(x)
    else:
        return 1.0


def extract_lora_values(string):
    pattern = r'<lora:([^>]+)>'
    matches = re.findall(pattern, string)

    def touch_lbw(text):
        return re.sub(r'LBW=[A-Za-z][A-Za-z0-9_-]*:', r'LBW=', text)

    items = [touch_lbw(match.strip(':')) for match in matches]

    added = set()
    result = []
    for item in items:
        item = item.split(':')

        lora = None
        a = None
        b = None
        lbw = None
        lbw_a = None
        lbw_b = None

        if len(item) > 0:
            lora = item[0]

            for sub_item in item[1:]:
                if is_numeric_string(sub_item):
                    if a is None:
                        a = float(sub_item)
                    elif b is None:
                        b = float(sub_item)
                elif sub_item.startswith("LBW="):
                    for lbw_item in sub_item[4:].split(';'):
                        if lbw_item.startswith("A="):
                            lbw_a = safe_float(lbw_item[2:].strip())
                        elif lbw_item.startswith("B="):
                            lbw_b = safe_float(lbw_item[2:].strip())
                        elif lbw_item.strip() != '':
                            lbw = lbw_item

        if a is None:
            a = 1.0
        if b is None:
            b = 1.0

        if lora is not None and lora not in added:
            result.append((lora, a, b, lbw, lbw_a, lbw_b))
            added.add(lora)

    return result


def remove_lora_tags(string):
    pattern = r'<lora:[^>]+>'
    result = re.sub(pattern, '', string)

    return result

def process_with_loras(wildcard_opt, model, clip, title="Positive", seed=None, can_load_lora=True, pipe_lora_stack=[], easyCache=None):
    pass1 = process(wildcard_opt, seed)
    loras = extract_lora_values(pass1)
    pass2 = remove_lora_tags(pass1)

    has_noodle_key = True if "__" in wildcard_opt else False
    has_loras = True if loras != [] else False
    show_wildcard_prompt = True if has_noodle_key or has_loras else False

    if can_load_lora and has_loras:
        for lora_name, model_weight, clip_weight, lbw, lbw_a, lbw_b in loras:
            if (lora_name.split('.')[-1]) not in folder_paths.supported_pt_extensions:
                lora_name = lora_name+".safetensors"
            lora = {
                "lora_name": lora_name, "model": model, "clip": clip, "model_strength": model_weight,
                "clip_strength": clip_weight,
                "lbw_a": lbw_a,
                "lbw_b": lbw_b,
                "lbw": lbw
            }
            model, clip = easyCache.load_lora(lora)
            lora["model"] = model
            lora["clip"] = clip
            pipe_lora_stack.append(lora)

    log_node_info("easy wildcards",f"{title}: {pass2}")
    if pass1 != pass2:
        log_node_info("easy wildcards",f'{title}_decode: {pass1}')

    return model, clip, pass2, pass1, show_wildcard_prompt, pipe_lora_stack
