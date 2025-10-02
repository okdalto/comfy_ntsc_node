from typing import Optional, Dict, Any, List
import numpy as np
import cv2
from .ntsc import Ntsc, random_ntsc  # 프로젝트 경로에 맞게 조정


class NTSCProcessor:
    """
    Headless NTSC processor.
    - process_image: 단일 cv2 이미지(ndarray) 입력 -> NTSC 효과 적용 후 ndarray 반환
    - process_video: 프레임 리스트(list[ndarray]) 입력 -> 변환된 프레임 리스트 반환
    """

    # -------------------- 공개 메서드 --------------------

    def process_image(
        self,
        img: np.ndarray,
        *,
        height: int = 600,
        seed: Optional[int] = None,
        ntsc_overrides: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if not isinstance(img, np.ndarray):
            raise TypeError("img must be a numpy.ndarray")

        h, w = img.shape[:2]
        new_w, new_h = self._resize_to_height((w, h), height)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = self._trim_to_4width(img)

        nt = self._make_nt(seed)
        if ntsc_overrides:
            self._apply_ntsc_overrides(nt, ntsc_overrides)

        out = self._ntsc_apply(nt, img)
        return out

    def process_video(
        self,
        frames: List[np.ndarray],
        *,
        height: int = 600,
        seed: Optional[int] = None,
        upscale2x: bool = False,
        ntsc_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[np.ndarray]:
        if not isinstance(frames, list) or not all(isinstance(f, np.ndarray) for f in frames):
            raise TypeError("frames must be a list of numpy.ndarray")

        if len(frames) == 0:
            return []

        h, w = frames[0].shape[:2]
        new_w, new_h = self._resize_to_height((w, h), height)
        if upscale2x:
            new_w *= 2
            new_h *= 2
        out_size = (new_w, new_h)

        nt = self._make_nt(seed)
        if ntsc_overrides:
            self._apply_ntsc_overrides(nt, ntsc_overrides)

        processed_frames: List[np.ndarray] = []
        prev = None
        for idx, frame in enumerate(frames):
            resized = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
            resized = self._trim_to_4width(resized)

            out = self._ntsc_apply(nt, resized, prev)
            processed_frames.append(out)

            prev = resized
            if (idx + 1) % 50 == 0:
                print(f"  - processed {idx+1}/{len(frames)} frames")

        return processed_frames

    # -------------------- 내부 유틸 --------------------

    @staticmethod
    def _resize_to_height(wh, target_h):
        w, h = wh
        if h <= 0 or w <= 0:
            raise ValueError("Invalid original size")
        scale = target_h / float(h)
        return int(round(w * scale)), int(round(h * scale))

    @staticmethod
    def _trim_to_4width(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        new_w = (w // 4) * 4
        return img if new_w == w else img[:, :new_w]

    @staticmethod
    def _make_nt(seed: Optional[int]) -> Ntsc:
        if seed is None:
            import random
            seed = random.choice([18, 31, 38, 44])
        nt = random_ntsc(int(seed))
        nt._enable_ringing2 = True
        return nt

    @staticmethod
    def _apply_ntsc_overrides(nt: Ntsc, overrides: Dict[str, Any]):
        for k, v in overrides.items():
            if hasattr(nt, k):
                setattr(nt, k, v)
            else:
                print(f"[WARN] Unknown NTSC parameter ignored: {k}")

    @staticmethod
    def _ntsc_apply(nt: Ntsc, frame1: np.ndarray, frame2: Optional[np.ndarray] = None) -> np.ndarray:
        if frame2 is None:
            frame2 = frame1
        out = nt.composite_layer(frame1, frame2, field=2, fieldno=2)
        ntsc_out = cv2.convertScaleAbs(out)
        ntsc_out[1:-1:2] = ntsc_out[0:-2:2] / 2 + ntsc_out[2::2] / 2
        return ntsc_out
