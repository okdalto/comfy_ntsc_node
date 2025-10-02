# import cv2

# class Example:
#     """
#     A example node

#     Class methods
#     -------------
#     INPUT_TYPES (dict):
#         Tell the main program input parameters of nodes.
#     IS_CHANGED:
#         optional method to control when the node is re executed.

#     Attributes
#     ----------
#     RETURN_TYPES (`tuple`):
#         The type of each element in the output tuple.
#     RETURN_NAMES (`tuple`):
#         Optional: The name of each output in the output tuple.
#     FUNCTION (`str`):
#         The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
#     OUTPUT_NODE ([`bool`]):
#         If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
#         The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
#         Assumed to be False if not present.
#     CATEGORY (`str`):
#         The category the node should appear in the UI.
#     DEPRECATED (`bool`):
#         Indicates whether the node is deprecated. Deprecated nodes are hidden by default in the UI, but remain
#         functional in existing workflows that use them.
#     EXPERIMENTAL (`bool`):
#         Indicates whether the node is experimental. Experimental nodes are marked as such in the UI and may be subject to
#         significant changes or removal in future versions. Use with caution in production workflows.
#     execute(s) -> tuple || None:
#         The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
#         For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
#     """
#     def __init__(self):
#         pass

#     @classmethod
#     def INPUT_TYPES(s):
#         """
#             Return a dictionary which contains config for all input fields.
#             Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
#             Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
#             The type can be a list for selection.

#             Returns: `dict`:
#                 - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
#                 - Value input_fields (`dict`): Contains input fields config:
#                     * Key field_name (`string`): Name of a entry-point method's argument
#                     * Value field_config (`tuple`):
#                         + First value is a string indicate the type of field or a list for selection.
#                         + Second value is a config for type "INT", "STRING" or "FLOAT".
#         """
#         return {
#             "required": {
#                 "image": ("IMAGE",),
#                 "int_field": ("INT", {
#                     "default": 0, 
#                     "min": 0, #Minimum value
#                     "max": 4096, #Maximum value
#                     "step": 64, #Slider's step
#                     "display": "number", # Cosmetic only: display as "number" or "slider"
#                     "lazy": True # Will only be evaluated if check_lazy_status requires it
#                 }),
#                 "float_field": ("FLOAT", {
#                     "default": 1.0,
#                     "min": 0.0,
#                     "max": 10.0,
#                     "step": 0.01,
#                     "round": 0.001, #The value representing the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
#                     "display": "number",
#                     "lazy": True
#                 }),
#                 "print_to_screen": (["enable", "disable"],),
#                 "string_field": ("STRING", {
#                     "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
#                     "default": "Hello World!",
#                     "lazy": True
#                 }),
#             },
#         }

#     RETURN_TYPES = ("IMAGE",)
#     #RETURN_NAMES = ("image_output_name",)

#     FUNCTION = "test"

#     #OUTPUT_NODE = False

#     CATEGORY = "Example"

#     def check_lazy_status(self, image, string_field, int_field, float_field, print_to_screen):
#         """
#             Return a list of input names that need to be evaluated.

#             This function will be called if there are any lazy inputs which have not yet been
#             evaluated. As long as you return at least one field which has not yet been evaluated
#             (and more exist), this function will be called again once the value of the requested
#             field is available.

#             Any evaluated inputs will be passed as arguments to this function. Any unevaluated
#             inputs will have the value None.
#         """
#         if print_to_screen == "enable":
#             return ["int_field", "float_field", "string_field"]
#         else:
#             return []

#     def test(self, image, string_field, int_field, float_field, print_to_screen):
#         if print_to_screen == "enable":
#             print(f"""Your input contains:
#                 string_field aka input text: {string_field}
#                 int_field: {int_field}
#                 float_field: {float_field}
#             """)
#         #do some processing on the image, in this example I just invert it
#         image = 1.0 - image
#         print(image.shape)
#         return (image,)

#     """
#         The node will always be re executed if any of the inputs change but
#         this method can be used to force the node to execute again even when the inputs don't change.
#         You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
#         executed, if it is different the node will be executed again.
#         This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
#         changes between executions the LoadImage node is executed again.
#     """
#     #@classmethod
#     #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
#     #    return ""

# # Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# # WEB_DIRECTORY = "./somejs"


# # Add custom API routes, using router
# from aiohttp import web
# from server import PromptServer

# @PromptServer.instance.routes.get("/hello")
# async def get_hello(request):
#     return web.json_response("hello")


# # A dictionary that contains all nodes you want to export with their names
# # NOTE: names should be globally unique
# NODE_CLASS_MAPPINGS = {
#     "Example": Example
# }

# # A dictionary that contains the friendly/humanly readable titles for the nodes
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Example": "Example Node"
# }

import base64
import json
import time
import numpy as np
import cv2
import torch

# HTTP 클라이언트: requests 선호, 없으면 urllib로 폴백
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request, urllib.error

def tensor_to_cv2(image: torch.Tensor) -> np.ndarray:
    if not isinstance(image, torch.Tensor):
        raise TypeError("IMAGE must be torch.Tensor")
    b, h, w, c = image.shape
    arr = image[0].detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    if c == 4:
        rgb = arr[..., :3]
    else:
        rgb = arr
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def cv2_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    if img_bgr is None:
        raise ValueError("Invalid image")
    if img_bgr.ndim == 2:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    elif img_bgr.shape[2] == 4:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).unsqueeze(0).contiguous()
    return t.clamp_(0.0, 1.0)

def encode_png_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def decode_png_b64_to_cv2(b64s: str) -> np.ndarray:
    raw = base64.b64decode(b64s)
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

# --- 교체: http_post_json ---
def http_post_json(url: str, data: dict, timeout: float) -> (int, dict, str):
    """
    returns: (status_code, parsed_json_or_{}, raw_text)
    """
    if HAS_REQUESTS:
        r = requests.post(url, json=data, timeout=timeout)
        raw = r.text
        try:
            body = r.json()
        except Exception:
            body = {}
        return r.status_code, body, raw
    else:
        payload = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                code = resp.getcode()
                raw = resp.read().decode(resp.headers.get_content_charset() or "utf-8")
                try:
                    body = json.loads(raw)
                except Exception:
                    body = {}
                return code, body, raw
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"POST failed: {e.code} {e.reason}")

def http_get_json(url: str, timeout: float) -> (int, dict):
    if HAS_REQUESTS:
        r = requests.get(url, timeout=timeout)
        # 202(대기중)와 200(완료/에러) 모두 허용
        try:
            body = r.json()
        except Exception:
            body = {}
        return r.status_code, body
    else:
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                code = resp.getcode()
                text = resp.read().decode(resp.headers.get_content_charset() or "utf-8")
                try:
                    body = json.loads(text)
                except Exception:
                    body = {}
                return code, body
        except urllib.error.HTTPError as e:
            # 202를 urllib가 예외로 던질 수 있음 → 빈 바디로 반환 처리
            if e.code == 202:
                return 202, {}
            raise RuntimeError(f"GET failed: {e.code} {e.reason}")

class RemoteProcessNode:
    """
    IMAGE → (로컬 서버에 제출 → 폴링) → IMAGE
    - POST /process로 job 생성, job_id 수신
    - GET /result/{job_id}를 주기적으로 조회(202=대기, 200=완료/에러)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "server_base": ("STRING", {"default": "http://127.0.0.1:9000"}),
                "timeout_sec": ("INT", {"default": 10, "min": 1, "max": 300, "step": 1}),
                "poll_interval_sec": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 5.0, "step": 0.05}),
                "on_fail": (["error", "passthrough"], {"default": "error"}),
                "server_delay_sec": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 60.0, "step": 0.5}),  # 데모용: 서버에 전달
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "I/O/Remote"

    def run(self, image, server_base, timeout_sec, poll_interval_sec, on_fail, server_delay_sec):
        try:
            # (1) 텐서 → base64
            bgr = tensor_to_cv2(image)
            b64 = encode_png_b64(bgr)

            # (2) 작업 제출
            submit_url = f"{server_base.rstrip('/')}/process"
            code, resp, raw = http_post_json(submit_url, {"image_b64": b64, "delay_sec": float(server_delay_sec)}, timeout=float(timeout_sec))

            # 우선순위: job_id → id → job → data.job_id
            job_id = None
            if isinstance(resp, dict):
                job_id = resp.get("job_id") or resp.get("id") or resp.get("job")
                if not job_id:
                    data_field = resp.get("data")
                    if isinstance(data_field, dict):
                        job_id = data_field.get("job_id")
            
            if code != 200 or not job_id:
                # 디버깅 도움: 원시 응답까지 출력
                raise RuntimeError(f"submit failed: code={code}, resp={resp}, raw={raw[:500]}")

            if code >= 300 or "job_id" not in resp:
                raise RuntimeError(f"submit failed: {code} {resp}")
            job_id = resp["job_id"]

            # (3) 폴링
            result_url = f"{server_base.rstrip('/')}/result/{job_id}"
            deadline = time.time() + float(timeout_sec)
            last_status = None
            while True:
                if time.time() > deadline:
                    raise TimeoutError("poll timeout")

                code, body = http_get_json(result_url, timeout=float(timeout_sec))
                if code == 202 or (body.get("status") == "pending"):
                    # 대기중
                    last_status = "pending"
                    time.sleep(float(poll_interval_sec))
                    continue

                status = body.get("status")
                if status == "done" and "image_b64" in body:
                    out_cv = decode_png_b64_to_cv2(body["image_b64"])
                    if out_cv is None:
                        raise RuntimeError("decode returned image failed")
                    out_tensor = cv2_to_tensor(out_cv)
                    return (out_tensor,)

                if status == "error":
                    raise RuntimeError(f"remote error: {body.get('error')}")

                # 예외적인 응답
                raise RuntimeError(f"unexpected poll response: code={code} body={body}")

        except Exception as e:
            if on_fail == "passthrough":
                return (image,)
            raise e

# 노드 등록
NODE_CLASS_MAPPINGS = {
    "RemoteProcessNode": RemoteProcessNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteProcessNode": "Remote Process (Polling)",
}
