# `tools/` 工具链协作说明

工具脚本服务于可复现的标定、资源准备、诊断和 benchmark，不是核心业务模块。运行前查看脚本参数和所需外部资源，输出中记录设备、模型、视频、帧数和配置。

## 重要入口

- `calibrate_camera.py`：从棋盘格图片生成相机 `.npz`。
- `render_diagnostic_video.py`：用共享推理管线生成诊断视频。
- `benchmark_phase1.py`：真实 YOLO/球场模型配置对比。
- `benchmark_phase2.py`：不依赖模型权重的实体语义调度开销。
- `benchmark_phase4.py`：共享 JPEG 编码缓存开销。
- `benchmark_tracking_stability.py`、`benchmark_state.py`：轨迹稳定性和状态数据路径 benchmark。
benchmark 结果是条件化证据，不是普遍性能承诺；缺权重的 case 应保留 missing/error 状态，不能静默换模型。
