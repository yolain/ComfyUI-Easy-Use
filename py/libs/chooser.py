from server import PromptServer
from aiohttp import web
import time

class ChooserCancelled(Exception):
    pass

class ChooserMessage:
    stash = {}
    messages = {}
    cancelled = False

    @classmethod
    def addMessage(cls, id, message):
        if message == '__cancel__':
            cls.messages = {}
            cls.cancelled = True
        elif message == '__start__':
            cls.messages = {}
            cls.stash = {}
            cls.cancelled = False
        else:
            cls.messages[str(id)] = message

    @classmethod
    def waitForMessage(cls, id, period=0.1, asList=False):
        sid = str(id)
        while not (sid in cls.messages) and not ("-1" in cls.messages):
            if cls.cancelled:
                cls.cancelled = False
                raise ChooserCancelled()
            time.sleep(period)
        if cls.cancelled:
            cls.cancelled = False
            raise ChooserCancelled()
        message = cls.messages.pop(str(id), None) or cls.messages.pop("-1")
        try:
            if asList:
                return [int(x.strip()) for x in message.split(",")]
            else:
                return int(message.strip())
        except ValueError:
            print(
                f"ERROR IN IMAGE_CHOOSER - failed to parse '${message}' as ${'comma separated list of ints' if asList else 'int'}")
            return [1] if asList else 1


@PromptServer.instance.routes.post('/easyuse/image_chooser_message')
async def make_image_selection(request):
    post = await request.post()
    ChooserMessage.addMessage(post.get("id"), post.get("message"))
    return web.json_response({})