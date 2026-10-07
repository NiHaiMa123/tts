# TTS 多后端角色语音平台实施计划

> 仓库：`NiHaiMa123/tts`
>
> 目标：把当前围绕 DotsTTS 积累的角色语音数据、质量审查、诊断、试听和 WebUI 能力抽成一个**与底座无关的角色 TTS 平台**。DotsTTS 作为 legacy baseline；新增 VoxCPM2、Qwen3-TTS。任何新底座必须先通过 codec / zero-shot 准入测试，再允许进入 LoRA / SFT。
>
> 当前优先角色：锁暝。后续同一平台复用守岸人、符玄等角色数据。

---

## 0. 核心原则

### 0.1 这不是另一个模型 fork

本仓库不能复制成 `dots.tts`、VoxCPM 或 Qwen3-TTS 的二次 fork。

本仓库只负责：

- 角色与数据集注册
- 多后端统一接口
- 后端进程/环境调度
- 推理任务
- codec roundtrip / zero-shot / fine-tune 评测
- 客观指标
- A/B / 盲听页
- WebUI
- 产物与 provenance 管理

具体模型实现尽量使用各自官方仓库/官方包，通过 backend worker 适配。

### 0.2 后端隔离

Dots / VoxCPM2 / Qwen3-TTS **不要强行共用同一个 Python 环境**。

原因：

- torch / transformers / accelerate / triton / flash-attn 等依赖可能冲突；
- Windows + RTX 5080（16GB）下不同项目的 CUDA 依赖差异较大；
- 后续升级一个后端不能破坏其他后端。

架构必须允许每个 backend 指定自己的：

- Python executable
- 工作目录
- 启动命令
- model root
- cache root
- 环境变量

主平台通过 subprocess worker 与后端交互。

### 0.3 一次只驻留一个大模型

RTX 5080 16GB 下默认：

1. 停止当前 backend worker；
2. 等待进程退出；
3. 释放 CUDA；
4. 启动目标 backend worker；
5. health check；
6. 开始生成。

不要同时常驻多个大模型。

### 0.4 先验证 codec，再训练

新后端接入顺序固定：

```text
环境可运行
  ↓
codec / reconstruction roundtrip
  ↓
zero-shot
  ↓
同文本 A/B
  ↓
人工试听通过
  ↓
才允许 LoRA / SFT
```

如果一个后端仅做原声 codec roundtrip 就明显出现不可接受的磨砂、金属感、失真或细节损失，直接淘汰，不投入训练时间。

### 0.5 不以单个指标代替人耳

- 不允许用 spectral flatness 单指标宣布“更干净”；
- 不允许因为 speaker cosine 高就宣布“更像”；
- 不允许因为 CER/WER 低就宣布整体质量好；
- 最终质量判断必须保留人工试听结果。

---

# 1. 下载与网络策略（强制）

用户要求：

> **所有依赖、模型、仓库下载优先使用镜像；镜像不可用时，再使用本机 7897 代理。**

实现时遵守以下优先级。

## 1.1 优先使用当前机器已有镜像配置

启动安装/下载前先检查：

- `HF_ENDPOINT`
- `HF_HOME`
- `HUGGINGFACE_HUB_CACHE`
- `PIP_INDEX_URL`
- `PIP_EXTRA_INDEX_URL`
- `UV_INDEX_URL`
- Git 当前 repo/local 配置

如果用户环境已经有可用镜像配置，优先沿用，不覆盖。

## 1.2 没有配置时，bootstrap 层允许使用镜像

为 Hugging Face / pip / GitHub 等下载设计独立 bootstrap 脚本，例如：

```text
scripts/bootstrap/
  common.ps1
  setup_voxcpm.ps1
  setup_qwen3_tts.ps1
  setup_dots_legacy.ps1
```

镜像设置只能放在：

- bootstrap 脚本；
- backend-local env；
- `.env.local`（gitignore）；

**禁止写死到业务 Python 源码。**

对于 Hugging Face，优先尝试用户已有/可用的镜像 endpoint；镜像失败必须显式打印原因，然后再进入代理 fallback。

不要静默从一个来源切到另一个来源。

## 1.3 镜像失败后使用 7897 代理

fallback proxy：

```text
http://127.0.0.1:7897
```

仅在镜像失败后，对当前安装/下载进程临时设置：

```text
HTTP_PROXY=http://127.0.0.1:7897
HTTPS_PROXY=http://127.0.0.1:7897
ALL_PROXY=http://127.0.0.1:7897
```

Git 优先使用单次命令参数或进程环境，例如：

```powershell
git -c http.proxy=http://127.0.0.1:7897 clone ...
```

**不要修改用户全局 Git proxy。**

## 1.4 下载必须可恢复

大模型下载必须：

- 使用官方支持的 cache / resume 能力；
- 不重复下载已有完整文件；
- 对关键模型记录 revision / commit / hash；
- 下载中断后能继续；
- 日志里记录最终使用的是 mirror 还是 proxy。

---

# 2. 数据与资产边界

## 2.1 不重复提交大文件

Git 仓库中默认不提交：

- pretrained model weights
- LoRA/SFT 大权重
- 大量 WAV
- Hugging Face cache
- backend venv
- 临时频谱图/中间 tensor

只提交：

- 配置
- manifests
- 小型 metrics/report
- scripts
- tests
- 文档

必要时可用 Git LFS，但第一阶段不要主动引入，除非确有必要。

## 2.2 Legacy 数据来源

现有 Dots 项目：

```text
NiHaiMa123/dotstts
```

本地旧仓库通常包含：

- 锁暝数据
- 守岸人数据
- 符玄数据
- 角色提取脚本
- 数据 freeze
- speaker filtering
- blind listening
- grit diagnostics

本项目可以读取/迁移这些**通用资产与方法**，但不要复制 `src/dots_tts/` 作为平台代码。

### 可以迁移/重构

- Wuthering Waves 音频/台词提取逻辑
- dataset manifest schema
- speaker filtering
- quality analysis
- prompt/reference selection
- codec roundtrip 诊断方法
- grit diagnostics
- blind listen page
- TXT → WAV 批处理流程
- WebUI 的状态机思路
- SHA256 / provenance

### 不作为平台核心迁移

- DotsTtsModel
- Dots runtime
- Dots AudioVAE/Vocoder 实现
- Dots LoRA artifact 格式
- ODE / guidance / vocoder_merge_steps 等 Dots 专用参数

---

# 3. 推荐仓库结构

第一版按以下边界实现，可根据实际包结构小幅调整，但不要把 backend-specific 参数重新泄漏到通用层。

```text
tts/
├─ PLAN.md
├─ README.md
├─ pyproject.toml
├─ .gitignore
│
├─ configs/
│  ├─ app.yaml
│  ├─ characters/
│  │  └─ suoming.yaml
│  ├─ backends/
│  │  ├─ dots_legacy.yaml
│  │  ├─ voxcpm2.yaml
│  │  └─ qwen3_tts.yaml
│  └─ evaluations/
│     └─ suoming_gate_v1.yaml
│
├─ src/
│  └─ character_tts/
│     ├─ app/
│     ├─ registry/
│     ├─ backends/
│     │  ├─ protocol.py
│     │  ├─ manager.py
│     │  ├─ worker_client.py
│     │  ├─ dots_legacy/
│     │  ├─ voxcpm/
│     │  └─ qwen3_tts/
│     ├─ datasets/
│     ├─ evaluation/
│     ├─ diagnostics/
│     ├─ audio/
│     └─ web/
│
├─ workers/
│  ├─ dots_worker.py
│  ├─ voxcpm_worker.py
│  └─ qwen3_tts_worker.py
│
├─ scripts/
│  ├─ bootstrap/
│  ├─ import_legacy_assets.py
│  ├─ run_gate.py
│  └─ start_webui.py
│
├─ tests/
└─ outputs/
   └─ .gitkeep
```

命名不再使用 `fuxuan_batch.py`、`dots_tts_lab` 这种绑定某角色/底座的名字。

---

# 4. 通用数据模型

## 4.1 CharacterProfile

角色配置只描述角色本身，不描述某个模型的 ODE 参数。

至少包含：

```yaml
character_id: suoming
display_name: 锁暝

dataset:
  source: legacy_dotstts
  root: ...
  train_manifest: ...
  validation_manifest: ...
  test_manifest: ...

reference:
  audio: ...
  text: ...
  sha256: ...

evaluation:
  anchor_texts:
    - id: rain
      text: 午后凉风拂过，雨云渐聚，细雨敲在屋檐上。我坐在窗边听雨，拭去玄朱锁上的薄尘。这确实是有些凄迷的场景，但……我很喜欢。
```

## 4.2 BackendProfile

backend 配置只描述后端。

通用字段：

```yaml
backend_id: voxcpm2
family: voxcpm
enabled: true

worker:
  python: ...
  cwd: ...
  command: ...
  startup_timeout_seconds: ...

model:
  path_or_id: ...
  revision: ...

cache:
  root: ...
```

专用采样参数放在 backend-local `generation` 下，不进入通用 schema。

例如 Dots：

```yaml
generation:
  num_steps: 16
  guidance_scale: 1.2
  speaker_scale: 1.5
```

Qwen/VoxCPM 使用各自字段。

---

# 5. Backend API

第一阶段不要设计巨大 RPC 系统。

使用简单、稳定、可测试的 worker protocol。

推荐：

- worker 独立 subprocess；
- localhost HTTP 或 JSONL stdin/stdout 二选一；
- 统一 request/response schema；
- 二进制音频通过临时 WAV 路径返回，避免 JSON base64 大包。

至少支持：

## 5.1 health

返回：

- backend id
- backend version
- model id/revision
- device
- dtype
- loaded
- VRAM（若可取）
- capabilities

## 5.2 capabilities

例如：

```json
{
  "zero_shot": true,
  "codec_roundtrip": true,
  "fine_tune": false,
  "lora": true,
  "voice_design": false,
  "streaming": false
}
```

不能假设所有后端功能一致。

## 5.3 generate

通用请求：

```text
text
reference_audio
reference_text
seed
output_path
backend_specific_options
```

统一返回：

```text
sample_rate
duration
output_path
generation_metadata
wall_seconds
error
```

## 5.4 codec_roundtrip

若后端可访问 codec/tokenizer reconstruction，则实现。

若架构无法提供真实 waveform roundtrip，明确返回：

`unsupported`

禁止伪造等价测试。

## 5.5 unload / shutdown

必须可显式退出并释放 CUDA。

---

# 6. Backend Manager

主进程维护：

```text
active_backend_id
worker_process
worker_client
worker_health
```

切换后端：

1. 拒绝并发生成；
2. shutdown 当前 worker；
3. 等待退出；
4. 超时才 terminate；
5. 确认进程结束；
6. 启动新 worker；
7. health check；
8. 更新 UI 状态。

错误不能导致 WebUI 主进程直接崩溃。

---

# 7. 第一阶段后端

## 7.1 Dots Legacy

目的：

- 作为现有质量基线；
- 复用已经得到的锁暝结果；
- 验证统一 backend API 可以包住现有系统。

优先不要复制 Dots 源码。

可以：

- 调用旧 `dotstts` 仓库现有 runtime；
- 使用旧项目 Python 环境；
- worker 中做薄适配。

必须能复现至少：

- 锁暝 Step500 raw
- 当前已知 reference
- 固定 seed=42
- 固定“午后凉风……”文本

不要求第一阶段复刻 v11 后处理。

## 7.2 VoxCPM2

优先级：**P1**

先完成：

1. 环境安装；
2. 模型下载；
3. 官方 smoke；
4. codec/reconstruction（若官方 API 可做）；
5. zero-shot；
6. 同文本试听。

**不要直接训练。**

只有 gate 通过后才新增 LoRA/SFT plan。

## 7.3 Qwen3-TTS

优先级：**P1**

同样先完成：

1. 环境安装；
2. 模型下载；
3. 官方 smoke；
4. codec/tokenizer reconstruction（若可做）；
5. zero-shot；
6. 同文本试听。

**不要直接训练。**

---

# 8. 锁暝首轮 Gate

## 8.1 固定测试文本

必须使用：

```text
午后凉风拂过，雨云渐聚，细雨敲在屋檐上。我坐在窗边听雨，拭去玄朱锁上的薄尘。这确实是有些凄迷的场景，但……我很喜欢。
```

## 8.2 固定 reference

从旧 `dotstts` 项目读取并验证 SHA256。

同时保留：

- validation 原声；
- 当前 25s prompt/reference。

路径不能只靠人工猜测；导入脚本需要验证存在性与 hash。

## 8.3 每个 backend 至少输出

```text
reference_original.wav

dots/
  codec_roundtrip.wav        # 若可用
  zero_shot.wav
  trained_or_lora.wav        # 仅复用已有结果，不新训练

voxcpm2/
  codec_roundtrip.wav        # 若支持
  zero_shot.wav

qwen3_tts/
  codec_roundtrip.wav        # 若支持
  zero_shot.wav
```

第一轮不要做后处理。

---

# 9. 通用诊断指标

从旧项目迁移并重构，但避免过度拟合 dots。

至少输出：

- duration
- sample rate
- RMS dBFS
- LUFS
- true peak
- 4–8 kHz relative energy
- 8–12 kHz relative energy
- 12–18 kHz relative energy
- voiced-frame spectral flatness
- spectral crest
- spectral entropy
- 2–9 kHz temporal spectral delta

若 speaker embedding 模型可复用，再输出 speaker cosine；但它只能作为辅助指标。

生成：

```text
outputs/gates/suoming_v1/metrics.json
outputs/gates/suoming_v1/report.md
```

---

# 10. 试听页

做一个统一试听页：

```text
outputs/gates/suoming_v1/listen/index.html
```

每个样本显示：

- backend
- case
- reference / zero-shot / roundtrip / fine-tuned
- 关键生成参数
- audio player
- 清澈度 1–5
- 磨砂/颗粒 1–5
- 像角色 1–5
- 自然度 1–5
- 稳定性备注
- 其他备注

评分存浏览器 localStorage，可导出 JSON。

不要自动替用户填写主观评分。

---

# 11. 首轮判定

## Gate A — reconstruction

如果 codec roundtrip 已明显出现：

- 磨砂
- 金属
- 塑料
- 高频颗粒
- 明显谐波损失

则该 backend 标记：

`codec_gate = fail`

原则上停止后续训练投入。

## Gate B — zero-shot

若 roundtrip 通过但 zero-shot：

- 音色明显错误；
- 不稳定；
- 大概率随机坏句；
- 音质明显退化；

标记：

`zero_shot_gate = weak/fail`

是否训练由报告单独判断。

## Gate C — training eligibility

只有：

`codec_gate != fail`

且 zero-shot 不存在结构性质量问题，才进入：

- LoRA
- SFT
- checkpoint sweep

---

# 12. 首轮完成后再决定训练

第一轮报告必须明确比较：

| Backend | Codec roundtrip | Zero-shot 清澈度 | Zero-shot 像角色 | 稳定性 | 16GB 可用性 | 是否进入训练 |
|---|---|---|---|---|---|---|
| Dots | | | | | | |
| VoxCPM2 | | | | | | |
| Qwen3-TTS | | | | | | |

没有用户试听结果时写：

`PENDING_USER_LISTENING`

不能擅自宣布某模型最终胜出。

---

# 13. WebUI v1

Gate CLI 跑通之后再做 WebUI，不要反过来。

WebUI v1 至少支持：

- 选择角色
- 选择 backend
- 显示 backend 状态
- 自动切换 worker
- 输入文本
- 选择 reference（默认角色 reference）
- 生成
- 播放生成结果
- 打开输出目录
- 任务进度
- 明确显示错误
- 显示当前模型/后端/版本

不要在 v1 做：

- 多用户
- 云部署
- 权限系统
- 复杂数据库
- 多 GPU 调度

---

# 14. Legacy 导入

实现：

`scripts/import_legacy_assets.py`

作用：

1. 定位旧 `dotstts`；
2. 读取锁暝相关：
   - dataset manifest
   - reference
   - validation anchor
   - 已有 Dots LoRA/profile metadata
3. 生成本项目 character config；
4. 不复制大 WAV，优先使用可配置外部路径；
5. 记录 hash；
6. 旧项目不存在时明确 BLOCKED，不创建假数据。

支持通过环境变量指定：

```text
DOTSTTS_ROOT
TTS_DATA_ROOT
TTS_MODEL_ROOT
```

本地默认路径可以做“检测候选”，但不要让业务逻辑依赖固定盘符。

---

# 15. 实施阶段

## Phase 0 — Scaffold

必须完成：

- 项目目录
- pyproject
- logging
- config loader
- registry schema
- Backend protocol
- worker manager skeleton
- tests
- README

验收：

```powershell
python -m pytest
```

通过。

## Phase 1 — Legacy import + Dots adapter

必须完成：

- legacy asset import
- Dots worker
- Dots health
- Dots generate
- 固定锁暝 sample
- 与旧项目结果可比

验收：

- 主平台不 import `dots_tts` internals；
- Dots 依赖只存在 worker/backend 侧；
- 切换/关闭 worker 后进程确实退出。

## Phase 2 — VoxCPM2 Gate

必须完成：

- 独立环境 bootstrap
- 镜像优先、7897 fallback
- 模型下载
- smoke
- roundtrip（若支持）
- zero-shot
- metrics
- listen sample

到此停止，不训练。

## Phase 3 — Qwen3-TTS Gate

要求同 Phase 2。

到此停止，不训练。

## Phase 4 — Unified Gate Report

必须生成：

- metrics.json
- report.md
- listen/index.html
- backend metadata
- 下载/环境 provenance
- BLOCKED case 说明

提交 GitHub。

## Phase 5 — 用户试听后再规划训练

**Phase 5 不在本轮自动执行。**

用户试听后，根据结果决定：

- VoxCPM2 LoRA/SFT
- Qwen3-TTS fine-tune
- 两者都训
- 或淘汰某 backend

不得在没有用户评价时自动进入大规模训练。

---

# 16. 测试要求

至少覆盖：

- config validation
- backend registry
- worker launch/stop
- worker crash recovery
- backend switch
- unsupported capability
- output path safety
- metadata serialization
- metric functions
- legacy import hash validation
- download strategy selection（mock，不真实下载）

Backend 真实模型 smoke 可标记 integration，不要求普通 unit test 每次下载/加载模型。

---

# 17. 日志与 provenance

每次生成必须可追踪：

- character id
- backend id
- backend version
- model id
- model revision/hash
- adapter/checkpoint（若有）
- reference SHA256
- text
- seed
- backend-specific generation args
- sample rate
- wall time
- output SHA256

写 sidecar JSON。

下载记录至少包括：

- asset
- source type：mirror / proxy / local cache
- resolved revision
- local path
- success/failure

禁止记录 token、cookie、账号密码等敏感内容。

---

# 18. Windows / RTX 5080 约束

目标机器：

- Windows
- RTX 5080 16GB
- NVIDIA sm_120

因此：

- 不得默认 CUDA 扩展一定支持 sm_120；
- 遇到 flash-attn / triton / custom op 不兼容，优先使用官方 fallback；
- 不为了跑 benchmark 强制编译不稳定 CUDA 扩展；
- OOM 必须记录，不得静默降低质量参数后冒充原 case；
- 可以使用 CPU offload 作为“可运行性测试”，但必须标记；
- 每个 backend 都要记录峰值 VRAM（能测则测）。

---

# 19. 代码质量约束

- 通用层禁止出现 `num_steps`、`guidance_scale` 等 Dots 专属字段；
- 通用层禁止 import 某个模型官方内部模块；
- backend-specific 依赖延迟加载；
- 不要用大量 `if backend == ...` 散落在业务代码；
- 使用 registry/factory；
- 不要捕获 Exception 后静默；
- 所有 BLOCKED/FAILED case 落盘；
- 不要为了“测试通过”跳过真实失败；
- 不要修改旧 `dotstts` 仓库，除非后续任务明确要求。

---

# 20. 本轮禁止事项

- 不训练 VoxCPM2；
- 不训练 Qwen3-TTS；
- 不重新训练 Dots；
- 不做新一轮 Dots voice-polish；
- 不做大量参数 sweep；
- 不复制整个 Dots/VoxCPM/Qwen 官方源码进本仓库；
- 不把模型权重 commit 到普通 Git；
- 不为了统一 API 抹掉各模型特有能力；
- 不自动判断主观音质赢家。

---

# 21. 本轮最终交付

Devin 完成本计划 Phase 0–4 后，至少提交：

1. 多 backend 平台骨架；
2. Dots legacy adapter；
3. VoxCPM2 adapter + gate 结果；
4. Qwen3-TTS adapter + gate 结果；
5. 锁暝统一 metrics；
6. 统一试听页；
7. `docs/reports/suoming-backend-gate-v1.md`；
8. bootstrap/download 日志与策略说明；
9. tests；
10. README 使用说明。

最终报告最后必须给：

### A. 环境状态

| Backend | Env | Model download | Smoke | Peak VRAM |
|---|---|---|---|---|
| Dots | | | | |
| VoxCPM2 | | | | |
| Qwen3-TTS | | | | |

### B. Gate 状态

| Backend | Codec gate | Zero-shot gate | 主观评分 | 是否建议训练 |
|---|---|---|---|---|
| Dots | | | PENDING_USER_LISTENING | |
| VoxCPM2 | | | PENDING_USER_LISTENING | |
| Qwen3-TTS | | | PENDING_USER_LISTENING | |

### C. 下一步

只能给“建议”，不能自动执行 Phase 5。

---

# 22. Devin 执行规则

开始时：

1. 拉取最新 `main`；
2. 阅读完整 `PLAN.md`；
3. 检查本机旧 `dotstts` 路径和已有模型/cache；
4. 优先复用本地 cache，避免重复下载；
5. 按 Phase 0 → 4 执行；
6. 每个 Phase 完成后先测试再继续；
7. 发现模型/API 与计划假设不一致时，以官方实际接口为准，但保持架构边界；
8. 不因单一 backend 阻塞整个任务：记录 BLOCKED 后继续其他 backend；
9. 全部完成后提交 GitHub；
10. 最终回复给出：
   - commit SHA
   - 报告路径
   - 试听页路径
   - 三个 backend 的 gate 结论
   - BLOCKED 项
   - 推荐下一步

**不要在本轮开始大规模训练。**
