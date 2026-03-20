"""
PersonaLive Gradio Demo
支持摄像头实时输入和参考图片，生成实时数字人
"""

# ── 兼容性补丁：huggingface_hub >= 0.28 移除了 cached_download ──────────────
import huggingface_hub as _hf_hub
if not hasattr(_hf_hub, "cached_download"):
    _hf_hub.cached_download = _hf_hub.hf_hub_download
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import threading
import queue
import numpy as np
import torch
import cv2
from PIL import Image
from types import SimpleNamespace

import gradio as gr

# ── 路径配置 ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

CONFIG_PATH = os.path.join(ROOT, "configs/prompts/personalive_online.yaml")
CHUNK_SIZE = 4          # PersonaLive 每次处理 4 帧
TARGET_W, TARGET_H = 512, 512

# ── 懒加载 Pipeline ───────────────────────────────────────────────────────────
_pipeline_lock = threading.Lock()
_pipeline = None          # PersonaLive 实例
_ref_loaded = False       # 是否已加载参考图
_input_q: queue.Queue = queue.Queue(maxsize=32)
_output_q: queue.Queue = queue.Queue(maxsize=64)
_stop_evt = threading.Event()
_infer_thread = None


def _build_args():
    return SimpleNamespace(
        config_path=CONFIG_PATH,
        host="0.0.0.0",
        port=7860,
        reload=False,
        mode="default",
        max_queue_size=0,
        timeout=0.0,
        safety_checker=False,
        taesd=False,
        ssl_certfile=None,
        ssl_keyfile=None,
        debug=False,
        acceleration="xformers",
        engine_dir="engines",
    )


def _load_pipeline():
    """首次调用时加载模型（约占 10 GB 显存）"""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            from src.wrapper import PersonaLive
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[PersonaLive] Loading models on {device} ...")
            _pipeline = PersonaLive(_build_args(), device)
            print("[PersonaLive] Models loaded.")
    return _pipeline


def _infer_worker():
    """后台推理线程：不断从 _input_q 取 4 帧，推理后写入 _output_q"""
    global _ref_loaded
    pipe = _load_pipeline()

    while not _stop_evt.is_set():
        # 等待凑齐 CHUNK_SIZE 帧
        frames = []
        try:
            while len(frames) < CHUNK_SIZE:
                try:
                    f = _input_q.get(timeout=0.05)
                    frames.append(f)
                except queue.Empty:
                    if _stop_evt.is_set():
                        return
        except Exception:
            return

        if not _ref_loaded:
            continue

        # 拼成 (N, C, H, W)  [-1,1]
        batch = torch.cat(frames, dim=0)
        try:
            output = pipe.process_input(batch)   # (N, H, W, 3)  [0,1]
            for frame_np in output:
                img = Image.fromarray((frame_np * 255).astype(np.uint8))
                try:
                    _output_q.put_nowait(img)
                except queue.Full:
                    try:
                        _output_q.get_nowait()
                    except queue.Empty:
                        pass
                    _output_q.put_nowait(img)
        except Exception as e:
            print(f"[Infer Error] {e}")


def _ensure_thread():
    global _infer_thread
    if _infer_thread is None or not _infer_thread.is_alive():
        _stop_evt.clear()
        _infer_thread = threading.Thread(target=_infer_worker, daemon=True)
        _infer_thread.start()


# ── Gradio 回调 ───────────────────────────────────────────────────────────────

def load_reference(ref_image):
    """
    加载参考图片，调用 fuse_reference
    ref_image: numpy (H,W,3) uint8 from Gradio Image component
    """
    global _ref_loaded

    if ref_image is None:
        return "❌ 请先上传参考图片"

    pipe = _load_pipeline()
    _ensure_thread()

    ref_pil = Image.fromarray(ref_image).convert("RGB").resize((TARGET_W, TARGET_H))

    # 重置状态，加载新参考图
    pipe.reset()
    _ref_loaded = False

    # 清空队列
    for q in [_input_q, _output_q]:
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    pipe.fuse_reference(ref_pil)
    _ref_loaded = True

    return "✅ 参考图加载成功，请开启摄像头开始生成"


def process_webcam_frame(webcam_frame, ref_status):
    """
    Gradio webcam 回调（stream=True 模式每帧调用一次）
    webcam_frame: numpy (H,W,3) uint8
    返回：生成的数字人帧（numpy (H,W,3) uint8）
    """
    if webcam_frame is None:
        return None

    if not _ref_loaded:
        # 参考图未加载，直接返回原始帧
        return webcam_frame

    _ensure_thread()

    # 将摄像头帧转为 tensor  [-1, 1]
    frame_rgb = cv2.resize(webcam_frame, (TARGET_W, TARGET_H))
    frame_tensor = torch.from_numpy(frame_rgb).float() / 127.5 - 1.0   # [0,255]->[−1,1]
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)           # (1,C,H,W)

    try:
        _input_q.put_nowait(frame_tensor)
    except queue.Full:
        pass   # 队列满则丢帧

    # 取最新生成帧
    latest = None
    while not _output_q.empty():
        try:
            latest = _output_q.get_nowait()
        except queue.Empty:
            break

    if latest is not None:
        return np.array(latest)
    return webcam_frame   # 还未产出时回显原帧


def process_video_file(video_path, ref_image):
    """
    离线视频模式：上传驱动视频 + 参考图，生成结果视频并返回文件路径
    """
    if ref_image is None:
        return None, "❌ 请先上传参考图片"
    if video_path is None:
        return None, "❌ 请上传驱动视频"

    pipe = _load_pipeline()

    ref_pil = Image.fromarray(ref_image).convert("RGB").resize((TARGET_W, TARGET_H))
    pipe.reset()
    pipe.fuse_reference(ref_pil)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame_rgb)
    cap.release()

    if len(all_frames) == 0:
        return None, "❌ 无法读取视频"

    # 截断到 4 的倍数
    num_frames = (len(all_frames) // CHUNK_SIZE) * CHUNK_SIZE
    all_frames = all_frames[:num_frames]

    output_frames = []
    for i in range(0, num_frames, CHUNK_SIZE):
        chunk = all_frames[i : i + CHUNK_SIZE]
        tensors = []
        for f in chunk:
            f_resized = cv2.resize(f, (TARGET_W, TARGET_H))
            t = torch.from_numpy(f_resized).float() / 127.5 - 1.0
            t = t.permute(2, 0, 1).unsqueeze(0)
            tensors.append(t)
        batch = torch.cat(tensors, dim=0)
        result = pipe.process_input(batch)   # (4, H, W, 3)  [0,1]
        for r in result:
            output_frames.append((r * 255).astype(np.uint8))

    # 写输出视频
    os.makedirs("results/gradio", exist_ok=True)
    out_path = f"results/gradio/output_{int(time.time())}.mp4"
    h, w = output_frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in output_frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    return out_path, f"✅ 生成完成，共 {len(output_frames)} 帧"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

DESCRIPTION = """
# 🎭 PersonaLive — 实时数字人生成 Demo

**模式说明：**
- **实时摄像头**：上传参考图 → 点击「加载参考图」→ 开启摄像头，实时生成数字人
- **离线视频**：上传参考图 + 驱动视频，点击「生成视频」，下载结果

> **注意**：首次运行会加载模型（~30 秒），请耐心等待。
"""

with gr.Blocks(title="PersonaLive Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():

        # ── Tab 1: 实时摄像头 ──────────────────────────────────────────────
        with gr.TabItem("📷 实时摄像头"):
            with gr.Row():
                with gr.Column(scale=1):
                    ref_img_live = gr.Image(
                        label="参考图片（数字人外观）",
                        type="numpy",
                        height=300,
                        sources=["upload"],
                    )
                    load_btn = gr.Button("🔄 加载参考图", variant="primary")
                    ref_status = gr.Textbox(
                        label="状态",
                        value="请上传参考图并点击「加载参考图」",
                        interactive=False,
                    )

                with gr.Column(scale=1):
                    webcam_in = gr.Image(
                        label="摄像头输入",
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        height=300,
                    )

                with gr.Column(scale=1):
                    webcam_out = gr.Image(
                        label="数字人输出",
                        type="numpy",
                        height=300,
                        streaming=True,
                    )

            load_btn.click(
                fn=load_reference,
                inputs=[ref_img_live],
                outputs=[ref_status],
            )

            webcam_in.stream(
                fn=process_webcam_frame,
                inputs=[webcam_in, ref_status],
                outputs=[webcam_out],
                stream_every=0.04,   # ~25 fps poll
                time_limit=3600,
            )

        # ── Tab 2: 离线视频 ────────────────────────────────────────────────
        with gr.TabItem("🎬 离线视频"):
            with gr.Row():
                with gr.Column(scale=1):
                    ref_img_offline = gr.Image(
                        label="参考图片（数字人外观）",
                        type="numpy",
                        sources=["upload"],
                        height=280,
                    )
                    drv_video = gr.Video(
                        label="驱动视频（提供动作/表情）",
                        height=280,
                    )
                    gen_btn = gr.Button("🚀 生成视频", variant="primary")
                    offline_status = gr.Textbox(
                        label="状态", value="", interactive=False
                    )

                with gr.Column(scale=1):
                    output_video = gr.Video(
                        label="生成结果",
                        height=400,
                    )

            gen_btn.click(
                fn=process_video_file,
                inputs=[drv_video, ref_img_offline],
                outputs=[output_video, offline_status],
            )

        # ── Tab 3: 使用说明 ────────────────────────────────────────────────
        with gr.TabItem("📖 使用说明"):
            gr.Markdown("""
## 使用说明

### 实时摄像头模式

1. 在「参考图片」区域上传目标人物图片（正面、清晰）
2. 点击「🔄 加载参考图」，等待状态显示「✅ 参考图加载成功」
3. 点击「摄像头输入」区域的摄像头按钮，授权浏览器使用摄像头
4. 「数字人输出」区域将实时显示生成结果

### 离线视频模式

1. 上传参考图片
2. 上传驱动视频（包含人脸动作/表情的视频）
3. 点击「🚀 生成视频」
4. 等待处理完成后，下载输出视频

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 输出分辨率 | 512×512 | 固定 |
| 推理步数 | 4 | DDIM 4步采样 |
| 时间窗口 | 4帧 | 每批处理帧数 |
| 数据类型 | FP16 | 需要支持FP16的GPU |

### 环境要求

- GPU: NVIDIA 显卡，显存 ≥ 16 GB（推荐 A100 / RTX 4090）
- CUDA ≥ 11.8
- Python 3.10+

### 常见问题

**Q: 生成延迟较大？**
A: 首次加载模型需要时间，之后每批4帧约0.5-1秒。实时模式下存在约1-2秒延迟属正常。

**Q: 人脸未被检测到？**
A: 请确保参考图片中人脸清晰、正面、光线充足。

**Q: CUDA out of memory？**
A: 关闭其他占用显存的程序后重试，或降低批处理大小。
            """)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--share", action="store_true", help="创建公网链接")
    parser.add_argument("--preload", action="store_true", help="启动时立即加载模型")
    cli_args = parser.parse_args()

    if cli_args.preload:
        print("预加载模型中...")
        _load_pipeline()
        _ensure_thread()

    demo.launch(
        server_name=cli_args.host,
        server_port=cli_args.port,
        share=cli_args.share,
        show_error=True,
    )
