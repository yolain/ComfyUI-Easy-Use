
def find_nearest_steps(clip_id, prompt):
    """Find the nearest KSampler or preSampling node that references the given id."""
    def check_link_to_clip(node_id, clip_id, visited=None, node=None):
        """Check if a given node links directly or indirectly to a loader node."""
        if visited is None:
            visited = set()

        if node_id in visited:
            return False
        visited.add(node_id)
        if "pipe" in node["inputs"]:
            link_ids = node["inputs"]["pipe"]
            for id in link_ids:
                if id != 0 and id == str(clip_id):
                    return True
        return False

    for id in prompt:
        node = prompt[id]
        if "Sampler" in node["class_type"] or "sampler" in node["class_type"] or "Sampling" in node["class_type"]:
            # Check if this KSampler node directly or indirectly references the given CLIPTextEncode node
            if check_link_to_clip(id, clip_id, None, node):
                steps = node["inputs"]["steps"] if "steps" in node["inputs"] else 1
                return steps
    return 1

def find_wildcards_seed(clip_id, text, prompt):
    """ Find easy wildcards seed value"""
    def find_link_clip_id(id, seed, wildcard_id):
        node = prompt[id]
        if "positive" in node['inputs']:
            link_ids = node["inputs"]["positive"]
            if type(link_ids) == list:
                for id in link_ids:
                    if id != 0:
                        if id == wildcard_id:
                            wildcard_node = prompt[wildcard_id]
                            seed = wildcard_node["inputs"]["seed_num"] if "seed_num" in wildcard_node["inputs"] else None
                            return seed
                        else:
                            return find_link_clip_id(id, seed, wildcard_id)
            else:
                return None
        else:
            return None
    if "__" in text:
        seed = None
        for id in prompt:
            node = prompt[id]
            if "wildcards" in node["class_type"]:
                wildcard_id = id
                return find_link_clip_id(str(clip_id), seed, wildcard_id)
        return seed
    else:
        return None

def is_linked_styles_selector(prompt, my_unique_id, prompt_type='positive'):
    inputs_values = prompt[my_unique_id]['inputs'][prompt_type] if prompt_type in prompt[my_unique_id][
        'inputs'] else None
    if type(inputs_values) == list and inputs_values != 'undefined' and inputs_values[0]:
        return True if prompt[inputs_values[0]] and prompt[inputs_values[0]]['class_type'] == 'easy stylesSelector' else False
    else:
        return False

def get_local_filepath(url, dirname, local_file_name=None):
    """Get local file path when is already downloaded or download it"""
    import os
    from urllib.parse import urlparse
    from torch.hub import download_url_to_file
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)
    destination = os.path.join(dirname, local_file_name)
    if not os.path.exists(destination):
        print(f'downloading {url} to {destination}')
        download_url_to_file(url, destination)
    return destination

def easySave(images, filename_prefix, output_type, prompt=None, extra_pnginfo=None):
    """Save or Preview Image"""
    from nodes import PreviewImage, SaveImage
    if output_type == "Hide":
        return list()
    if output_type == "Preview":
        filename_prefix = 'easyPreview'
        results = PreviewImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return results['ui']['images']
    else:
        results = SaveImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return results['ui']['images']
