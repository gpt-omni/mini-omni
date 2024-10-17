import flask
import base64
import tempfile
import traceback

import torch
from flask import Flask, Response, stream_with_context
from inference import OmniInference


class OmniChatServer(object):
    def __init__(self, ip='0.0.0.0', port=60808, run_app=True,
                 ckpt_dir='./checkpoint', device=None) -> None:
        server = Flask(__name__)
        # CORS(server, resources=r"/*")
        # server.config["JSON_AS_ASCII"] = False
        self.device = self.get_device(device)
        self.client = OmniInference(ckpt_dir, self.device)
        self.client.warm_up()

        server.route("/chat", methods=["POST"])(self.chat)

        if run_app:
            server.run(host=ip, port=port, threaded=False)
        else:
            self.server = server

    def get_device(self, device):
        if device is None:
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        else:
            if device == 'cuda' and torch.cuda.is_available():
                return 'cuda'
            elif device == 'mps' and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'

    def chat(self) -> Response:

        req_data = flask.request.get_json()
        try:
            print("Req Data: ", req_data)
            data_buf = req_data["audio"].encode("utf-8")
            print("Data buffer: ", data_buf)
            data_buf = base64.b64decode(data_buf)
            stream_stride = req_data.get("stream_stride", 4)
            print(stream_stride)
            max_tokens = req_data.get("max_tokens", 2048)
            print(max_tokens)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(data_buf)
                audio_generator = self.client.run_AT_batch_stream(f.name, stream_stride, max_tokens)
                return Response(stream_with_context(audio_generator), mimetype="audio/wav")
        except Exception as e:
            print(traceback.format_exc())


# CUDA_VISIBLE_DEVICES=1 gunicorn -w 2 -b 0.0.0.0:60808 'server:create_app()'
def create_app():
    server = OmniChatServer(run_app=False)
    return server.server


def serve(ip='0.0.0.0', port=60808, device=None):
    OmniChatServer(ip, port=port, run_app=True, device=device)



if __name__ == "__main__":
    import fire

    fire.Fire(serve)