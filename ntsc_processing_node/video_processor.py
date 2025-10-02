from fractions import Fraction
from typing import Tuple
from comfy_api.input.video_types import VideoInput
import numpy as np
import cv2
import torch
from custom_nodes.sh.app import NTSCProcessor
from comfy_api.util import VideoComponents
from comfy_api.input_impl import VideoFromComponents

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

class VideoProcessor:
    def __init__(self):
        self.proc = NTSCProcessor()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),  # ← 바로 이거

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
            }
        }


    RETURN_TYPES = ("VIDEO",)   # 처리 후 다시 비디오를 내보내는 경우
    FUNCTION = "process"
    CATEGORY = "video/utils"

    def process(
        self,
        video,
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
        comps = video.get_components()
        frames = comps.images  
        fps = comps.frame_rate  # fractions.Fraction
        audio = comps.audio     # Optional[TypedDict], 없을 수도 

        cv2_frames = []
        for i in range(frames.shape[0]):
            img_bgr = tensor_to_cv2(frames[i:i+1])
            cv2_frames.append(img_bgr)
        frames = self.proc.process_video(
            cv2_frames, 
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
        frames = [cv2_to_tensor(f) for f in frames]
        frames = torch.cat(frames, dim=0)  # [T, H, W, C] torch.Tensor
        
        new_comps = VideoComponents(
            images=frames,              # [T, H, W, 3] torch.Tensor
            frame_rate=fps if isinstance(fps, Fraction) else Fraction(float(fps)),
            audio=audio,              # 원래 오디오 유지(있다면)
            metadata=comps.metadata,  # 메타데이터 보존
        )

        return (VideoFromComponents(new_comps),)


NODE_CLASS_MAPPINGS = {
    "VideoProcessor": VideoProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoProcessor": "NTSC Video Processor",
}
