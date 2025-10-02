# comfy_ntsc_node

A custom **ComfyUI** node that applies an **NTSC video effect** to images or videos.  
The NTSC processing logic is adapted from [JargeZ/ntscqt](https://github.com/JargeZ/ntscqt/tree/master).

---

## ✨ Features
- Simulates NTSC encoding/decoding artifacts (color bleeding, scanline blending, etc.)
- Works as a plug-and-play node in **ComfyUI**.
- Supports both **images** and **videos**.

---

## 📦 Installation

1. Clone or download this repository.
2. Copy the `ntsc_processing_node` folder into your **ComfyUI custom nodes directory**:
   ```
   ComfyUI/custom_nodes/ntsc_processing_node
   ```
3. Install the required dependency:
   ```bash
   pip install opencv-python
   ```
4. Restart ComfyUI.

---

## 🖼️ Usage

1. Open **ComfyUI**.
2. Add the **NTSC Processing Node** to your workflow.
3. Connect an input image/video.
4. Adjust parameters (if available) to control NTSC effect strength.
5. Preview or export the processed result.

---

## 🔗 Reference
- NTSC processing code adapted from: [JargeZ/ntscqt](https://github.com/JargeZ/ntscqt/tree/master)

---

## 📜 License
This project follows the license of the original [ntscqt](https://github.com/JargeZ/ntscqt) repository.  
Check the upstream repo for details.
