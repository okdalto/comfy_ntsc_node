import cv2
import numpy as np
import torch
from custom_nodes.sh.app import NTSCProcessor

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


class ImageProcessor:
    def __init__(self):
        self.proc = NTSCProcessor()

    @classmethod
    def INPUT_TYPES(s):
        """
        Return a dictionary which contains config for all input fields.
        Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
        The type can be a list for selection.
        """
        return {
            "required": {
                "image": ("IMAGE",),
    
                # --- Image / base fields ---
                "_image_size": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 2048,
                    "step": 64,
                    "display": "number",
                    "lazy": True
                }),
    
                # --- VHS / Composite sliders (INT / FLOAT 범위 반영) ---
                "_composite_preemphasis": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number",
                    "lazy": True
                }),
                "_vhs_out_sharpen": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "display": "number",
                    "lazy": True
                }),
                "_vhs_edge_wave": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "lazy": True
                }),
    
                "_ringing": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number",
                    "lazy": True
                }),
                "_ringing_power": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "lazy": True
                }),
                "_ringing_shift": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number",
                    "lazy": True
                }),
    
                "_freq_noise_size": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number",
                    "lazy": True
                }),
                "_freq_noise_amplitude": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "display": "number",
                    "lazy": True
                }),
    
                "_color_bleed_horiz": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "lazy": True
                }),
                "_color_bleed_vert": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "lazy": True
                }),
    
                "_video_chroma_noise": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16384,
                    "step": 1,
                    "display": "number",
                    "lazy": True
                }),
                "_video_chroma_phase_noise": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "number",
                    "lazy": True
                }),
                "_video_chroma_loss": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 800,
                    "step": 1,
                    "display": "number",
                    "lazy": True
                }),
                "_video_noise": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4200,
                    "step": 100,
                    "display": "number",
                    "lazy": True
                }),
    
                "_video_scanline_phase_shift": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 270,
                    "step": 1,
                    "display": "number",
                    "lazy": True
                }),
                "_video_scanline_phase_shift_offset": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 3,
                    "step": 1,
                    "display": "number",
                    "lazy": True
                }),
    
                "_head_switching_speed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number",
                    "lazy": True
                }),
    
                # --- Checkboxes (BOOLEAN) ---
                "_vhs_head_switching": ("BOOLEAN", {"default": False}),
                "_color_bleed_before": ("BOOLEAN", {"default": False}),
                "_enable_ringing2": ("BOOLEAN", {"default": False}),
                "_composite_in_chroma_lowpass": ("BOOLEAN", {"default": False}),
                "_composite_out_chroma_lowpass": ("BOOLEAN", {"default": False}),
                "_composite_out_chroma_lowpass_lite": ("BOOLEAN", {"default": False}),
                "_emulating_vhs": ("BOOLEAN", {"default": False}),
                "_nocolor_subcarrier": ("BOOLEAN", {"default": False}),
                "_vhs_chroma_vert_blend": ("BOOLEAN", {"default": False}),
                "_vhs_svideo_out": ("BOOLEAN", {"default": False}),
                "_output_ntsc": ("BOOLEAN", {"default": False}),
                "_black_line_cut": ("BOOLEAN", {"default": False}),
            },
        }


    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "process"

    #OUTPUT_NODE = False

    CATEGORY = "image/utils"

    def process(
        self,
        image,
        _image_size,
        _composite_preemphasis,
        _vhs_out_sharpen,
        _vhs_edge_wave,
        _ringing,
        _ringing_power,
        _ringing_shift,
        _freq_noise_size,
        _freq_noise_amplitude,
        _color_bleed_horiz,
        _color_bleed_vert,
        _video_chroma_noise,
        _video_chroma_phase_noise,
        _video_chroma_loss,
        _video_noise,
        _video_scanline_phase_shift,
        _video_scanline_phase_shift_offset,
        _head_switching_speed,
        _vhs_head_switching,
        _color_bleed_before,
        _enable_ringing2,
        _composite_in_chroma_lowpass,
        _composite_out_chroma_lowpass,
        _composite_out_chroma_lowpass_lite,
        _emulating_vhs,
        _nocolor_subcarrier,
        _vhs_chroma_vert_blend,
        _vhs_svideo_out,
        _output_ntsc,
        _black_line_cut,
    ):    
        # torch tensor -> cv2 image
        image = tensor_to_cv2(image)
    
        # process_image에 모든 오버라이드 전달
        processed_image = self.proc.process_image(
            image,
            height=_image_size,
            seed=24,
            ntsc_overrides={
                # FLOAT/INT 슬라이더
                "_composite_preemphasis": _composite_preemphasis,
                "_vhs_out_sharpen": _vhs_out_sharpen,
                "_vhs_edge_wave": _vhs_edge_wave,
                "_ringing": _ringing,
                "_ringing_power": _ringing_power,
                "_ringing_shift": _ringing_shift,
                "_freq_noise_size": _freq_noise_size,
                "_freq_noise_amplitude": _freq_noise_amplitude,
                "_color_bleed_horiz": _color_bleed_horiz,
                "_color_bleed_vert": _color_bleed_vert,
                "_video_chroma_noise": _video_chroma_noise,
                "_video_chroma_phase_noise": _video_chroma_phase_noise,
                "_video_chroma_loss": _video_chroma_loss,
                "_video_noise": _video_noise,
                "_video_scanline_phase_shift": _video_scanline_phase_shift,
                "_video_scanline_phase_shift_offset": _video_scanline_phase_shift_offset,
                "_head_switching_speed": _head_switching_speed,
    
                # 체크박스(BOOLEAN)
                "_vhs_head_switching": _vhs_head_switching,
                "_color_bleed_before": _color_bleed_before,
                "_enable_ringing2": _enable_ringing2,
                "_composite_in_chroma_lowpass": _composite_in_chroma_lowpass,
                "_composite_out_chroma_lowpass": _composite_out_chroma_lowpass,
                "_composite_out_chroma_lowpass_lite": _composite_out_chroma_lowpass_lite,
                "_emulating_vhs": _emulating_vhs,
                "_nocolor_subcarrier": _nocolor_subcarrier,
                "_vhs_chroma_vert_blend": _vhs_chroma_vert_blend,
                "_vhs_svideo_out": _vhs_svideo_out,
                "_output_ntsc": _output_ntsc,
                "_black_line_cut": _black_line_cut,
            },
        )
    
        # cv2 image -> torch tensor
        image = cv2_to_tensor(processed_image)
        return (image,)


    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, image, string_field, _video_noise, _color_bleed_horiz, _ringing, print_to_screen):
    #    return ""

# 노드 등록
NODE_CLASS_MAPPINGS = {
    "ImageProcessor": ImageProcessor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageProcessor": "NTSC Image Processor",
}
