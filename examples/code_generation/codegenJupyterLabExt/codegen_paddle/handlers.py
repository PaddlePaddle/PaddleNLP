import os
import json
import paddle
from paddlenlp.transformers import CodeGenTokenizer, CodeGenForCausalLM

from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join

import tornado
from tornado.web import StaticFileHandler

from .codegen import gen_code
from .config import ModifiedConfig, DefaultConfig


generate_config = None
tokenizer = None
model = None
init = True


class HelloRouteHandler(APIHandler):
    @tornado.web.authenticated
    def get(self):
        self.finish(json.dumps({"data": "This is /codegen-paddle/hello endpoint!"}))


class InitModelRouteHandler(APIHandler):
    @tornado.web.authenticated
    def post(self):
        global init, generate_config, model, tokenizer
        if init:
            init = False
            config = ModifiedConfig()
            try:
                input_data = self.get_json_body()
                max_length = input_data["max_length"]
                min_length = input_data["min_length"]
                repetition_penalty = input_data["repetition_penalty"]
                top_p = input_data["top_p"]
                top_k = input_data["top_k"]
                temperature = input_data["temperature"]
                device = input_data["device"]
                model_ = input_data["model"]

                config.max_length = max_length
                config.min_length = min_length
                config.repetition_penalty = repetition_penalty
                config.top_p = top_p
                config.top_k = top_k
                config.temperature = temperature
                config.device = device
                config.model_name_or_path = model_
            except:
                config = DefaultConfig()
            generate_config = config
            paddle.set_device(generate_config.device)
            paddle.set_default_dtype(generate_config.default_dtype)

            try:
                tokenizer = CodeGenTokenizer.from_pretrained(generate_config.model_name_or_path)
                model = CodeGenForCausalLM.from_pretrained(
                    generate_config.model_name_or_path,
                    load_state_as_np=generate_config.load_state_as_np)
                self.finish(json.dumps({"res": "{}".format('succ')}))
            except:
                init = True
                self.finish(json.dumps({"res": "{}".format('fail')}))


class CodegenRouteHandler(APIHandler):
    @tornado.web.authenticated
    def post(self):
        input_data = self.get_json_body()
        prompt = input_data["prompt"]
        res = gen_code(prompt, model, tokenizer, generate_config)
        data = {"res": "{}".format(res)}
        self.finish(json.dumps(data))


def setup_handlers(web_app, url_path):
    host_pattern = ".*$"
    base_url = web_app.settings["base_url"]

    # Prepend the base_url so that it works in a JupyterHub setting
    route_pattern = url_path_join(base_url, url_path, "hello")
    handlers = [(route_pattern, HelloRouteHandler)]
    web_app.add_handlers(host_pattern, handlers)

    # Prepend the base_url so that it works in a JupyterHub setting
    route_pattern = url_path_join(base_url, url_path, "init-model")
    handlers = [(route_pattern, InitModelRouteHandler)]
    web_app.add_handlers(host_pattern, handlers)

    # Prepend the base_url so that it works in a JupyterHub setting
    route_pattern = url_path_join(base_url, url_path, "codegen")
    handlers = [(route_pattern, CodegenRouteHandler)]
    web_app.add_handlers(host_pattern, handlers)

    # Prepend the base_url so that it works in a JupyterHub setting
    doc_url = url_path_join(base_url, url_path, "public")
    doc_dir = os.getenv(
        "JLAB_SERVER_EXAMPLE_STATIC_DIR",
        os.path.join(os.path.dirname(__file__), "public"),
    )
    handlers = [("{}/(.*)".format(doc_url), StaticFileHandler, {"path": doc_dir})]
    web_app.add_handlers(".*$", handlers)
