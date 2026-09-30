# `app/vision/` 视觉模块协作说明

## 当前链路

`VisionCore.process(frame, frame_index)` 负责共享的相机去畸变、YOLOv11 person、ByteTrack、低频球场关键点、RANSAC 单应性和球员底部中心场地投影。`semantics.py`、`role.py` 和 classification 模块在轨迹层补充角色/球队标签；`backends.py` 管理 PyTorch/ONNX/TensorRT 适配。

## 算法不变量

- 第 1 帧执行球场关键点，之后按配置间隔检测；跳帧只复用有限时长的单应性（默认不超过 0.5 秒），过期必须标记 stale/unavailable。
- 球员场地位置使用 bbox bottom-center；预测球框不能回灌 ByteTrack，也不能用于更新球队原型。
- 角色/球队置信度不足、模型缺失或校准不可用时保持 `UNKNOWN`/`-1`，不要用颜色或猜测制造确定身份。
- 足球预测有最大帧数和失效状态；无有效几何变换时不能输出伪造场地坐标。
- 推理计数、延迟、复用率、可用率、轨迹中断等指标必须与真实调用一致。

优先使用不依赖权重的合成帧、伪检测和状态测试；真实模型 benchmark 另行记录模型/设备/输入视频/帧数等条件。

```bash
uv run pytest tests/test_vision_core.py tests/test_semantics.py tests/test_backends.py -q
```
