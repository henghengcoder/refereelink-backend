# `app/state/` 状态与线协议说明

`models.py` 定义后端与前端共享的 Pydantic 契约：`FrameState`、`MetricsSnapshot`、`GameEvent`、`PlayerState`、`PipelineConfig` 以及相关枚举。`web/src/types/messages.ts` 和 `multiview.ts` 是对应的 TypeScript 镜像。字段、枚举值、默认值、可空性和 JSON 序列化形态都属于兼容性边界。

`store.py` 提供线程安全的最新帧状态、配置、指标、事件、日志、source 状态和原始帧/JPEG 缓存；事件和日志是有界的（当前约束分别为 500/200）。`events.py` 的 `EventBus` 是进程内同步通知机制，不是前端订阅通道。

修改模型后至少检查：初始化默认值、`model_dump(mode="json")`、store 快照、REST/WS publisher、前端类型和 `tests/test_models.py`/`tests/test_state_store.py`。不要把原始图像放入 `FrameState`，也不要让无限历史进入 store。

```bash
uv run pytest tests/test_models.py tests/test_state_store.py tests/test_frame_encoder.py -q
```
