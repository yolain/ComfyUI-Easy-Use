import os
import hashlib
import hmac
import sys
import json
import shutil
import secrets
import tempfile
from functools import lru_cache
from urllib.parse import urlsplit
import folder_paths
from aiohttp import web
from PIL import Image, UnidentifiedImageError
from server import PromptServer
from .config import RESOURCES_DIR, FOOOCUS_STYLES_DIR, FOOOCUS_STYLES_SAMPLES
from .libs.model import easyModelManager
from .libs.utils import getMetadata, cleanGPUUsedForce, get_local_filepath
from .libs.cache import remove_cache
from .libs.translate import has_chinese, zh_to_en

@PromptServer.instance.routes.get('/easyuse/version')
def get_version(request):
    try:
        from .. import __version__
        return web.json_response({"version": __version__})
    except Exception as e:
        print(e)
        return web.Response(status=500)

@PromptServer.instance.routes.post("/easyuse/cleangpu")
def cleanGPU(request):
    try:
        cleanGPUUsedForce()
        return web.Response(status=200)
    except Exception as e:
        return web.Response(status=500)
        pass

@PromptServer.instance.routes.post("/easyuse/removecache")
async def removecache(request):
    post = await request.post()
    key = post.get("key")
    try:
        remove_cache(key)
        return web.Response(status=200)
    except Exception as e:
        return web.Response(status=500)
        pass

@PromptServer.instance.routes.post("/easyuse/translate")
async def translate(request):
    post = await request.post()
    text = post.get("text")
    if has_chinese(text):
        return web.json_response({"text": zh_to_en([text])[0]})
    else:
        return web.json_response({"text": text})

_reboot_token = secrets.token_urlsafe(32)


def _same_origin_request(request):
    fetch_site = request.headers.get("Sec-Fetch-Site")
    if fetch_site and fetch_site not in ("same-origin", "none"):
        return False
    origin = request.headers.get("Origin")
    return not origin or urlsplit(origin).netloc == request.host


@PromptServer.instance.routes.get("/easyuse/reboot-token")
async def get_reboot_token(request):
    if not _same_origin_request(request):
        return web.Response(status=403)
    return web.json_response({"token": _reboot_token}, headers={"Cache-Control": "no-store"})


@PromptServer.instance.routes.post("/easyuse/reboot")
async def reboot(request):
    token = request.headers.get("X-EasyUse-Reboot-Token", "")
    if not _same_origin_request(request) or not hmac.compare_digest(token, _reboot_token):
        return web.Response(status=403)

    try:
        sys.stdout.close_log()
    except Exception as e:
        pass

    return os.execv(sys.executable, [sys.executable] + sys.argv)

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
        style_name = request.rel_url.query["name"]
        fooocus_custom_dir = os.path.join(FOOOCUS_STYLES_DIR, 'fooocus_styles.json')
        if style_name == 'fooocus_styles' and not os.path.exists(fooocus_custom_dir):
            file = os.path.join(RESOURCES_DIR, style_name+'.json')
            cn_file = os.path.join(RESOURCES_DIR, style_name + '_cn.json')
        else:
            file = os.path.join(FOOOCUS_STYLES_DIR, style_name+'.json')
            cn_file = os.path.join(FOOOCUS_STYLES_DIR, style_name + '_cn.json')
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
                    if "name_cn" in d:
                        nd['name_cn'] = d['name_cn']
                    elif cn_data:
                        nd['name_cn'] = cn_data[key] if key in cn_data else key
                    nd["name"] = d['name']
                    if "thumbnail" in d:
                        thumbnail = d['thumbnail']
                        if isinstance(d['thumbnail'], str):
                            nd['thumbnail'] = thumbnail if "http" in thumbnail else f'/easyuse/prompt/styles/image?path={thumbnail}'
                        elif isinstance(d['thumbnail'], list):
                            nd['thumbnail'] = [thumb if "http" in thumb else f'/easyuse/prompt/styles/image?path={thumb}' for thumb in thumbnail]
                    else:
                        nd['thumbnail'] = f'/easyuse/prompt/styles/image?name={name}&styles_name={style_name}'
                    if "thumbnail_variant" in d:
                        nd['thumbnailVariant'] = d['thumbnail_variant']
                    if "media_type" in d:
                        nd['mediaType'] = d['media_type']
                    if "media_subtype" in d:
                        nd['mediaSubtype'] = d['media_subtype']
                    if "prompt" in d:
                        nd['prompt'] = d['prompt']
                    if "negative_prompt" in d:
                        nd['negative_prompt'] = d['negative_prompt']
                    ndata.append(nd)
                return web.json_response(ndata)
    return web.Response(status=400)

# get style preview image
@PromptServer.instance.routes.get("/easyuse/prompt/styles/image")
async def getStylesImage(request):
    styles_name = request.rel_url.query["styles_name"] if "styles_name" in request.rel_url.query else None
    if "path" in request.rel_url.query:
        path = request.rel_url.query["path"]
        file = os.path.join(FOOOCUS_STYLES_DIR, 'samples', path)
        parent_file = os.path.join(FOOOCUS_STYLES_DIR, path)
        if os.path.isfile(file):
            return web.FileResponse(file)
        elif os.path.isfile(parent_file):
            return web.FileResponse(parent_file)
    elif "name" in request.rel_url.query:
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

# get models lists
@PromptServer.instance.routes.get("/easyuse/models/list")
async def getModelsList(request):
    if "type" in request.rel_url.query:
        type = request.rel_url.query["type"]
        if type not in ['checkpoints', 'loras']:
            return web.Response(status=400)
        manager = easyModelManager()
        return web.json_response(manager.get_model_lists(type))
    else:
        return web.Response(status=400)

@PromptServer.instance.routes.post("/easyuse/metadata/notes/{name}")
async def save_notes(request):
    name = request.match_info["name"]
    type, separator, name = name.partition("/")
    if not separator or type not in ("checkpoints", "loras", "embeddings"):
        return web.Response(status=400)

    file_path = None
    if type == "embeddings" or type == "loras":
        name = name.lower()
        files = folder_paths.get_filename_list(type)
        for f in files:
            lower_f = f.lower()
            if lower_f == name:
                file_path = folder_paths.get_full_path(type, f)
            else:
                n = os.path.splitext(f)[0].lower()
                if n == name:
                    file_path = folder_paths.get_full_path(type, f)

            if file_path is not None:
                break
    else:
        if name in folder_paths.get_filename_list(type):
            file_path = folder_paths.get_full_path(type, name)
    if not file_path:
        return web.Response(status=404)

    file_no_ext = os.path.splitext(file_path)[0]
    info_file = file_no_ext + ".txt"
    staged_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=os.path.dirname(info_file),
                prefix=".easyuse-notes-", delete=False
        ) as staged:
            staged_path = staged.name
            staged.write(await request.text())
        os.replace(staged_path, info_file)
    finally:
        if staged_path and os.path.exists(staged_path):
            os.unlink(staged_path)

    return web.Response(status=200)

@PromptServer.instance.routes.get("/easyuse/metadata/{name}")
async def load_metadata(request):
    name = request.match_info["name"]
    type, separator, name = name.partition("/")
    if not separator or type not in ("checkpoints", "loras", "embeddings"):
        return web.Response(status=400)

    file_path = None
    if type == "embeddings":
        name = name.lower()
        files = folder_paths.get_filename_list(type)
        for f in files:
            lower_f = f.lower()
            if lower_f == name:
                file_path = folder_paths.get_full_path(type, f)
            else:
                n = os.path.splitext(f)[0].lower()
                if n == name:
                    file_path = folder_paths.get_full_path(type, f)

            if file_path is not None:
                break
    else:
        if name in folder_paths.get_filename_list(type):
            file_path = folder_paths.get_full_path(type, name)
    if not file_path:
        return web.Response(status=404)

    try:
        header = getMetadata(file_path)
        header_json = json.loads(header)
        meta = header_json["__metadata__"] if "__metadata__" in header_json else None
    except:
        meta = None

    if meta is None:
        meta = {}

    file_no_ext = os.path.splitext(file_path)[0]

    info_file = file_no_ext + ".txt"
    if os.path.isfile(info_file) and not os.path.islink(info_file):
        with open(info_file, "r") as f:
            meta["easyuse.notes"] = f.read()

    # Sidecar hashes are user-controlled; never use them as proof of the model's hash.
    stat = os.stat(file_path)
    meta["easyuse.sha256"] = _model_sha256(
        file_path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    )

    return web.json_response(meta)


@lru_cache(maxsize=128)
def _model_sha256(path, size, mtime_ns, ctime_ns):
    digest = hashlib.sha256()
    with open(path, "rb") as model:
        for chunk in iter(lambda: model.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_PREVIEW_FORMATS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".gif": "GIF",
}

@PromptServer.instance.routes.post("/easyuse/save/{name}")
async def save_preview(request):
    name = request.match_info["name"]
    model_type, separator, model_name = name.partition("/")
    if not separator or model_type not in ("checkpoints", "loras"):
        return web.Response(status=400)
    if model_name not in folder_paths.get_filename_list(model_type):
        return web.Response(status=404)

    model_path = folder_paths.get_full_path(model_type, model_name)
    if not model_path:
        return web.Response(status=404)

    body = await request.json()
    filename = body.get("filename")
    if (body.get("type") != "temp" or body.get("subfolder", "") != ""
            or not isinstance(filename, str) or not filename
            or os.path.basename(filename) != filename or filename in (".", "..")):
        return web.Response(status=400)

    extension = os.path.splitext(filename)[1].lower()
    if extension not in _PREVIEW_FORMATS:
        return web.Response(status=400)

    temp_dir = folder_paths.get_directory_by_type("temp")
    filepath = os.path.join(temp_dir, filename)
    if (os.path.commonpath((os.path.realpath(temp_dir), os.path.realpath(filepath)))
            != os.path.realpath(temp_dir) or not os.path.isfile(filepath)):
        return web.Response(status=400)

    image_path = os.path.splitext(model_path)[0] + extension
    if (os.path.islink(image_path)
            or os.path.realpath(os.path.dirname(image_path))
            != os.path.realpath(os.path.dirname(model_path))):
        return web.Response(status=400)

    staged_path = None
    try:
        with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(image_path), prefix=".easyuse-preview-", delete=False
        ) as staged:
            staged_path = staged.name
            with open(filepath, "rb") as source:
                shutil.copyfileobj(source, staged)
        with Image.open(staged_path) as image:
            if image.format != _PREVIEW_FORMATS[extension]:
                return web.Response(status=400)
            image.verify()
        os.replace(staged_path, image_path)
    except (OSError, ValueError, UnidentifiedImageError):
        return web.Response(status=400)
    finally:
        if staged_path and os.path.exists(staged_path):
            os.unlink(staged_path)

    return web.json_response({
        "image":  model_type + "/" + os.path.basename(image_path)
    })

@PromptServer.instance.routes.post("/easyuse/model/download")
async def download_model(request):
    post = await request.post()
    url = post.get("url")
    local_dir = post.get("local_dir")
    if local_dir not in ['checkpoints', 'loras', 'controlnet', 'onnx', 'instantid', 'ipadapter', 'dynamicrafter_models', 'mediapipe', 'rembg', 'layer_model']:
        return web.Response(status=400)
    local_path = os.path.join(folder_paths.models_dir, local_dir)
    try:
        get_local_filepath(url, local_path)
        return web.Response(status=200)
    except:
        return web.Response(status=500)
