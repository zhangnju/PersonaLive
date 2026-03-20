"""
PersonaLive Gradio Demo
摄像头输入基于服务端 OpenCV/FFmpeg 采集，generator 流式输出数字人帧
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

# ── 路径 & 常量 ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

CONFIG_PATH = os.path.join(ROOT, "configs/prompts/personalive_online.yaml")
CHUNK_SIZE = 4
TARGET_W, TARGET_H = 512, 512

# ── 全局状态 ──────────────────────────────────────────────────────────────────
_pipeline_lock = threading.Lock()
_pipeline = None
_ref_loaded = False

_input_q: queue.Queue = queue.Queue(maxsize=32)
_output_q: queue.Queue = queue.Queue(maxsize=64)

_infer_stop = threading.Event()
_infer_thread = None

_cap_stop = threading.Event()
_cap_thread = None


# ── 模型加载 ──────────────────────────────────────────────────────────────────
def _build_args():
    return SimpleNamespace(
        config_path=CONFIG_PATH,
        host="0.0.0.0", port=7860, reload=False, mode="default",
        max_queue_size=0, timeout=0.0, safety_checker=False, taesd=False,
        ssl_certfile=None, ssl_keyfile=None, debug=False,
        acceleration="xformers", engine_dir="engines",
    )


def _load_pipeline():
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            from src.wrapper import PersonaLive
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[PersonaLive] Loading models on {device} ...")
            _pipeline = PersonaLive(_build_args(), device)
            print("[PersonaLive] Models loaded.")
    return _pipeline


# ── 推理线程 ──────────────────────────────────────────────────────────────────
def _infer_worker():
    pipe = _load_pipeline()
    while not _infer_stop.is_set():
        frames = []
        while len(frames) < CHUNK_SIZE:
            try:
                f = _input_q.get(timeout=0.05)
                frames.append(f)
            except queue.Empty:
                if _infer_stop.is_set():
                    return
        if not _ref_loaded:
            continue
        batch = torch.cat(frames, dim=0)
        try:
            output = pipe.process_input(batch)          # (4, H, W, 3) [0,1]
            for frame_np in output:
                out_img = (frame_np * 255).astype(np.uint8)
                try:
                    _output_q.put_nowait(out_img)
                except queue.Full:
                    try:
                        _output_q.get_nowait()
                    except queue.Empty:
                        pass
                    _output_q.put_nowait(out_img)
        except Exception as e:
            print(f"[Infer Error] {e}")


def _ensure_infer_thread():
    global _infer_thread
    if _infer_thread is None or not _infer_thread.is_alive():
        _infer_stop.clear()
        _infer_thread = threading.Thread(target=_infer_worker, daemon=True)
        _infer_thread.start()


# ── OpenCV/FFmpeg 采集线程 ────────────────────────────────────────────────────
def _capture_worker(source):
    """
    source: int (设备号) 或 str (文件路径 / RTSP / HTTP 流地址)
    持续读帧，resize 后转 tensor 推入 _input_q
    """
    # FFmpeg backend 对 RTSP/HTTP 更稳定
    backend = cv2.CAP_FFMPEG if isinstance(source, str) and source.startswith(("rtsp", "http")) \
              else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened():
        print(f"[Capture] 无法打开视频源: {source}")
        return

    print(f"[Capture] 已打开: {source}")
    while not _cap_stop.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[Capture] 读帧失败，尝试重连...")
            time.sleep(0.5)
            cap.release()
            cap = cv2.VideoCapture(source, backend)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (TARGET_W, TARGET_H))

        t = torch.from_numpy(frame_resized).float() / 127.5 - 1.0   # [-1,1]
        t = t.permute(2, 0, 1).unsqueeze(0)                          # (1,C,H,W)

        try:
            _input_q.put_nowait(t)
        except queue.Full:
            try:
                _input_q.get_nowait()
            except queue.Empty:
                pass
            _input_q.put_nowait(t)

    cap.release()
    print("[Capture] 采集线程已停止")


def _clear_queue(q: queue.Queue):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


# ── Gradio 回调 ───────────────────────────────────────────────────────────────

def load_reference(ref_image):
    """加载参考图，触发 fuse_reference"""
    global _ref_loaded
    if ref_image is None:
        return "❌ 请先上传参考图片"

    pipe = _load_pipeline()
    _ensure_infer_thread()

    ref_pil = Image.fromarray(ref_image).convert("RGB").resize((TARGET_W, TARGET_H))
    pipe.reset()
    _ref_loaded = False
    _clear_queue(_input_q)
    _clear_queue(_output_q)
    pipe.fuse_reference(ref_pil)
    _ref_loaded = True
    return "✅ 参考图加载成功，可开始采集"


def start_capture(cam_source_str):
    """
    启动 OpenCV 采集 + 推理，以 generator 方式逐帧 yield 输出图像
    cam_source_str: "0" / "1" / 文件路径 / rtsp:// / http://
    """
    global _cap_thread

    if not _ref_loaded:
        yield None, "❌ 请先加载参考图片"
        return

    # 停止旧采集线程
    _cap_stop.set()
    if _cap_thread and _cap_thread.is_alive():
        _cap_thread.join(timeout=2.0)
    _cap_stop.clear()
    _clear_queue(_input_q)
    _clear_queue(_output_q)

    # 解析输入源
    source = cam_source_str.strip()
    if source.isdigit():
        source = int(source)

    _cap_thread = threading.Thread(target=_capture_worker, args=(source,), daemon=True)
    _cap_thread.start()
    _ensure_infer_thread()

    last_frame = None
    while not _cap_stop.is_set():
        try:
            frame_np = _output_q.get(timeout=0.1)   # numpy (H,W,3) uint8
            last_frame = frame_np
            yield frame_np, "🟢 生成中..."
        except queue.Empty:
            if last_frame is not None:
                yield last_frame, "🟡 等待帧..."
            else:
                yield None, "🟡 等待推理..."


def stop_capture():
    """停止采集"""
    _cap_stop.set()
    _clear_queue(_input_q)
    _clear_queue(_output_q)
    return None, "⏹ 已停止"


def process_video_file(video_path, ref_image):
    """离线视频：逐批推理，返回结果视频路径"""
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
        all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not all_frames:
        return None, "❌ 无法读取视频"

    num_frames = (len(all_frames) // CHUNK_SIZE) * CHUNK_SIZE
    all_frames = all_frames[:num_frames]

    output_frames = []
    for i in range(0, num_frames, CHUNK_SIZE):
        chunk = all_frames[i : i + CHUNK_SIZE]
        tensors = []
        for f in chunk:
            f_r = cv2.resize(f, (TARGET_W, TARGET_H))
            t = torch.from_numpy(f_r).float() / 127.5 - 1.0
            t = t.permute(2, 0, 1).unsqueeze(0)
            tensors.append(t)
        batch = torch.cat(tensors, dim=0)
        result = pipe.process_input(batch)
        for r in result:
            output_frames.append((r * 255).astype(np.uint8))

    os.makedirs("results/gradio", exist_ok=True)
    out_path = f"results/gradio/output_{int(time.time())}.mp4"
    h, w = output_frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in output_frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    return out_path, f"✅ 完成，共 {len(output_frames)} 帧，已保存至 {out_path}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────
DESCRIPTION = """
# 🎭 基于AMD ROCm的实时数字人生成Demo
摄像头采集基于服务端 **OpenCV / FFmpeg**，支持本地摄像头设备号、视频文件、RTSP/HTTP 流。
"""

with gr.Blocks(title="基于AMD ROCm的实时数字人生成Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():

        # ── Tab 1: 实时摄像头（服务端采集）────────────────────────────────────
        with gr.TabItem("📷 实时摄像头"):
            with gr.Row():
                # 左列：参考图 + 摄像头配置
                with gr.Column(scale=1):
                    ref_img_live = gr.Image(
                        label="参考图片（数字人外观）",
                        type="numpy",
                        height=280,
                        sources=["upload"],
                    )
                    load_btn = gr.Button("🔄 加载参考图", variant="primary")
                    ref_status = gr.Textbox(
                        label="状态",
                        value="请上传参考图并点击「加载参考图」",
                        interactive=False,
                    )
                    gr.Markdown("---")
                    cam_source = gr.Textbox(
                        label="摄像头 / 视频源",
                        value="0",
                        placeholder="设备号: 0 / 1  |  文件: /path/to/video.mp4  |  流: rtsp://...",
                    )
                    with gr.Row():
                        start_btn = gr.Button("▶ 开始采集", variant="primary")
                        stop_btn  = gr.Button("⏹ 停止", variant="stop")

                # 右列：输出
                with gr.Column(scale=2):
                    live_out = gr.Image(
                        label="数字人输出",
                        type="numpy",
                        height=500,
                    )
                    live_status = gr.Textbox(
                        label="采集状态", value="", interactive=False
                    )

            load_btn.click(
                fn=load_reference,
                inputs=[ref_img_live],
                outputs=[ref_status],
            )
            start_btn.click(
                fn=start_capture,
                inputs=[cam_source],
                outputs=[live_out, live_status],
            )
            stop_btn.click(
                fn=stop_capture,
                outputs=[live_out, live_status],
            )

        # ── Tab 2: 离线视频 ────────────────────────────────────────────────────
        with gr.TabItem("🎬 离线视频"):
            with gr.Row():
                with gr.Column(scale=1):
                    ref_img_offline = gr.Image(
                        label="参考图片", type="numpy",
                        sources=["upload"], height=260,
                    )
                    drv_video = gr.Video(
                        label="驱动视频（提供动作/表情）", height=260,
                    )
                    gen_btn = gr.Button("🚀 生成视频", variant="primary")
                    offline_status = gr.Textbox(
                        label="状态", value="", interactive=False,
                    )
                with gr.Column(scale=1):
                    output_video = gr.Video(label="生成结果", height=400)

            gen_btn.click(
                fn=process_video_file,
                inputs=[drv_video, ref_img_offline],
                outputs=[output_video, offline_status],
            )

        # ── Tab 3: 使用说明 ────────────────────────────────────────────────────
        with gr.TabItem("📖 使用说明"):
            gr.Markdown("""
## 使用说明

### 实时摄像头模式

1. 上传参考图片，点击「🔄 加载参考图」
2. 填写视频源：
   - 本地摄像头：填 `0`（第一个摄像头）或 `1`、`2`…
   - 本地视频文件：填完整路径，如 `/home/user/video.mp4`
   - RTSP 流：`rtsp://192.168.1.100:554/stream`
   - HTTP 流：`http://192.168.1.100:8080/video`
3. 点击「▶ 开始采集」，右侧实时显示数字人输出
4. 点击「⏹ 停止」结束采集

### 离线视频模式

1. 上传参考图片 + 驱动视频
2. 点击「🚀 生成视频」，等待完成后可下载
3. 结果保存在 `results/gradio/` 目录

### 视频源格式参考

| 类型 | 填写示例 |
|------|---------|
| 本地摄像头 | `0` |
| 视频文件 | `/home/user/driving.mp4` |
| RTSP 摄像头 | `rtsp://admin:pass@192.168.1.1:554/stream1` |
| HTTP MJPEG | `http://192.168.1.100:8080/video` |

### 支持设备

| GPU | 显存 | 备注 |
|-----|------|------|
| AMD Radeon RX 9070 XT | 16 GB | 推荐，ROCm 6.x |
| AMD Radeon RX 7900 XTX | 24 GB | 推荐，ROCm 5.7+ |

> 运行前请确认已安装对应版本的 ROCm 驱动，并使用支持 ROCm 的 PyTorch 版本。

### 常见问题

**Q: 点击「开始采集」后无画面？**
A: 检查视频源填写是否正确；本地摄像头先用 `0`，不行换 `1`；RTSP 流检查网络连通性。

**Q: 生成延迟？**
A: 模型每批处理4帧，延迟约1-2秒属正常。

**Q: ROCm OOM？**
A: 确保显存 ≥ 16GB，关闭其他 GPU 程序后重试。
            """)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--preload", action="store_true", help="启动时立即加载模型")
    cli_args = parser.parse_args()

    if cli_args.preload:
        print("预加载模型...")
        _load_pipeline()
        _ensure_infer_thread()

    demo.launch(
        server_name=cli_args.host,
        server_port=cli_args.port,
        share=cli_args.share,
        show_error=True,
    )
