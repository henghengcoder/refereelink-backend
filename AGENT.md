# RefereeLink Backend 项目协作说明

## 适用范围

本文件适用于整个仓库。进入更深目录后，同时遵守该目录最近的 `AGENT.md`；更深层说明只补充本文件，不取代其中未被明确覆盖的约定。

## 项目定位

这是一个足球比赛实时计算机视觉与辅助判罚系统：Python 后端负责视频采集、相机去畸变、YOLO/ByteTrack、球场关键点与单应性投影、球员语义和事件候选；球的场地坐标留给后续传感器，视觉管线不再检测球。FastAPI 对外提供 REST、WebSocket 和 MJPEG；`web/` 是当前唯一积极维护的 React/Vite 前端；`app/multiview/` 提供多视角犯规证据与复核链路。

核心数据流是：

```text
VideoSource → BoundedFrameBuffer → InferencePipeline/VisionCore
           → FrameState + MetricsSnapshot → StateStore
           → REST/WebSocket/MJPEG → React Dashboard
```

## 目录地图

- `app/`：生产 Python 代码；状态、管线、视觉、事件、服务和 FastAPI 服务端。
- `web/`：React + TypeScript + Vite 仪表盘。
- `tests/`：Python 单元、接口、协议和合成视频集成测试。
- `tools/`：标定、资源准备、诊断视频和 benchmark 脚本。
- `docs/`：架构、迁移、路线图和性能/诊断证据。
- `assets/`：模型、视频、标定等运行时资产；大文件通常不入库。
- `experiments/`、`notebooks/`：实验性分析和训练笔记，不是生产运行入口。

## 常用工作流

Python 环境由 `pyproject.toml` 与 `uv.lock` 管理：

```bash
uv sync --dev
uv run pytest tests/ -v
uv run ruff check app tests tools experiments
```

前端验证：

```bash
cd web
npm install
npm run lint
npm run build
```

运行服务可参考 `README.md` 和 `SETUP_GUIDE.md`；通常后端监听 `8000`，Vite 开发服务器监听 `5173`。真实模型、视频、CUDA、RTSP 和外部 MVFoul 依赖可能使运行结果受本机环境影响。

## 必须保持的设计边界

- `app/state/models.py` 是后端状态与线协议的事实来源；变更 `FrameState`、`MetricsSnapshot`、`PipelineConfig`、事件或校准字段时，必须同步检查 `web/src/types/`、publisher、API 和相关测试。
- WebSocket `/ws/state` 只传结构化 JSON；视频走 `/video/stream` 的 MJPEG（未来可替换为 WebRTC），不要把 Base64 视频塞进 WebSocket。
- 实时模式应保持有界资源和低延迟：队列满时丢旧帧；离线模式才阻塞等待完整顺序。不要让 UI、网络或日志阻塞推理线程。
- 模型、标定、视频、CUDA 或外部模型缺失时优先清晰报错或降级到 `UNKNOWN`/`unavailable`/脚本 fallback；不要静默伪造有效结果。
- 多视角链路必须区分模型证据、确定性规则、辅助判定候选和人工复核结论；不要把模型输出直接写成裁判最终事实。
- 任何“测试通过”、性能数字或质量结论都要对应实际执行证据。静态测试清单、历史 benchmark 和已执行测试不可混写。
- 不要把密钥、模型权重、原始视频、运行时数据库或本地缓存提交到仓库；先查看 `.gitignore`，再决定生成物是否应保留。

## 变更与验证习惯

先读邻近模块和现有测试，再做最小改动。涉及线协议、线程生命周期、缓存、持久化或模型降级时，必须补充或更新针对性测试/验证。除非用户明确要求，不要顺手 commit、push 或覆盖业务代码。完成后报告实际改动和实际运行过的验证，不把未运行的命令描述为通过。
