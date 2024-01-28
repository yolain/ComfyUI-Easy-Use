import re
import os
import sys
import json
from server import PromptServer
from .config import RESOURCES_DIR, FOOOCUS_STYLES_DIR, FOOOCUS_STYLES_SAMPLES

try:
    import aiohttp
    from aiohttp import web
except ImportError:
    print("Module 'aiohttp' not installed. Please install it via:")
    print("pip install aiohttp")
    sys.exit()

# parse csv
@PromptServer.instance.routes.post("/easyuse/upload/csv")
async def parse_csv(request):
    post = await request.post()
    csv = post.get("csv")
    if csv and csv.file:
        file = csv.file
        text = ''
        for line in file.readlines():
            line = str(line.strip())
            line = line.replace("'", "").replace("b",'')
            text += line + '; \n'
        return web.json_response(text)


#get style list
@PromptServer.instance.routes.get("/easyuse/prompt/styles")
async def getStylesList(request):
    if "name" in request.rel_url.query:
        name = request.rel_url.query["name"]
        if name == 'fooocus_styles':
            file = os.path.join(RESOURCES_DIR, name+'.json')
            cn_file = os.path.join(RESOURCES_DIR, name + '_cn.json')
        else:
            file = os.path.join(FOOOCUS_STYLES_DIR, name+'.json')
            cn_file = os.path.join(FOOOCUS_STYLES_DIR, name + '_cn.json')
        cn_data = None
        if os.path.isfile(cn_file):
            f = open(cn_file, 'r', encoding='utf-8')
            cn_data = json.load(f)
            f.close()
        if os.path.isfile(file):
            f = open(file, 'r', encoding='utf-8')
            data = json.load(f)
            f.close()
            if data:
                ndata = []
                for d in data:
                    nd = {}
                    name = d['name'].replace('-', ' ')
                    words = name.split(' ')
                    key = ' '.join(
                        word.upper() if word.lower() in ['mre', 'sai', '3d'] else word.capitalize() for word in
                        words)
                    img_name = '_'.join(words).lower()
                    if "name_cn" in d:
                        nd['name_cn'] = d['name_cn']
                    elif cn_data:
                        nd['name_cn'] = cn_data[key] if key in cn_data else key
                    nd["name"] = d['name']
                    nd['imgName'] = img_name
                    ndata.append(nd)
                return web.json_response(ndata)
    return web.Response(status=400)

# get style preview image
@PromptServer.instance.routes.get("/easyuse/prompt/styles/image")
async def getStylesImage(request):
    styles_name = request.rel_url.query["styles_name"] if "styles_name" in request.rel_url.query else None
    if "name" in request.rel_url.query:
        name = request.rel_url.query["name"]
        if os.path.exists(os.path.join(FOOOCUS_STYLES_DIR, 'samples')):
            file = os.path.join(FOOOCUS_STYLES_DIR, 'samples', name + '.jpg')
            if os.path.isfile(file):
                return web.FileResponse(file)
            elif styles_name == 'fooocus_styles':
                return web.Response(text=FOOOCUS_STYLES_SAMPLES + name + '.jpg')
        elif styles_name == 'fooocus_styles':
            return web.Response(text=FOOOCUS_STYLES_SAMPLES + name + '.jpg')
    return web.Response(status=400)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}