import re
import os
import sys
from server import PromptServer

try:
    import aiohttp
    from aiohttp import web
except ImportError:
    print("Module 'aiohttp' not installed. Please install it via:")
    print("pip install aiohttp")
    sys.exit()

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

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}