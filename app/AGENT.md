# `app/` 后端协作说明

## 模块职责

- `state/`：Pydantic 线协议、线程安全 `StateStore`、内部 `EventBus`。
- `pipeline/`：本地文件/RTSP source、有界帧缓冲、推理线程和可选录制。
- `vision/`：共享球员/球场视觉、后端适配、球处理、轨迹语义和显示数据。
- `geometry/`：固定广角相机标定、去畸变、视角变换和球场投影。
- `events/`：基于球员场地坐标的越位等可解释事件候选（球识别已移除）。
- `multiview/`：多视角犯规推理、定位、规则评估和复核持久化；详见局部说明。
- `server/`：FastAPI 路由、WebSocket 生命周期、MJPEG 和 pipeline 控制。
- `services/`：WebSocket publisher 和共享 JPEG 编码缓存。
- `modes/`：历史离线/legacy 模式，不能假定它们是 Web Dashboard 主入口。

## 后端原则

`StateStore` 是线程之间的共享边界，事件、日志、帧和状态缓冲必须有上限。推理线程不得依赖浏览器连接存在，也不得在其中做不可控的网络等待。跨线程对象更新应使用已有 store/bus 接口，避免绕过锁直接改内部容器。

`FrameState`、`MetricsSnapshot`、`PipelineConfig` 和事件模型属于公共契约。字段、枚举、默认值和 `model_dump(mode="json")` 形态变更时，连同前端镜像、REST/WS 测试一起检查。

## Python 验证

```bash
uv run pytest tests/ -v
uv run pytest tests/test_pipeline_entities.py tests/test_api.py tests/test_websocket.py -q
uv run ruff check app tests tools experiments
```

需要真实权重、CUDA、RTSP 或外部模型的检查应单独标注环境前置条件；没有执行就不要写成通过。
