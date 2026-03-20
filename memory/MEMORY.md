# PersonaLive 项目记忆

## 项目概述
- 路径: /home/nzhang/source_codes/PersonaLive
- 功能: 扩散模型驱动的实时肖像动画（数字人）
- 核心: 4步DDIM采样，时间窗口4帧，输入512x512

## 关键文件
- `src/wrapper.py` - PersonaLive类（在线推理，核心接口）
- `inference_offline.py` - 离线视频推理入口
- `inference_online.py` - FastAPI + WebSocket在线服务
- `webcam/vid2vid.py` - 多进程Pipeline封装
- `configs/prompts/personalive_online.yaml` - 在线推理配置
- `gradio_demo.py` - Gradio Demo（已创建）

## PersonaLive 接口
```python
pipe = PersonaLive(args, device)  # args需要.config_path
pipe.fuse_reference(ref_pil)      # 加载参考图（PIL），每次换参考图前先reset()
output = pipe.process_input(batch) # batch: (4,C,H,W) [-1,1], 返回 (4,H,W,3) [0,1]
pipe.reset()                       # 重置状态
```

## 依赖/环境
- torch 2.1.0, diffusers 0.27.0, transformers 4.36.2
- gradio (需额外安装: pip install gradio)
- 显存需求: ≥16GB
- 配置文件: configs/prompts/personalive_online.yaml
