# TTS 多后端角色语音平台实施计划

> 仓库：`NiHaiMa123/tts`
>
> 目标：把当前围绕 DotsTTS 积累的角色语音数据、质量审查、诊断、试听和 WebUI 能力抽成一个**与底座无关的角色 TTS 平台**。DotsTTS 作为 legacy baseline；新增 VoxCPM2、Qwen3-TTS。任何新底座必须先通过 codec / zero-shot 准入测试，再允许进入 LoRA / SFT。
>
> 当前优先角色：锁暝。后续同一平台复用守岸人、符玄等角色数据。

---

## 当前执行状态与优先任务（2026-10-08）

- **Phase 0–4：已由 Devin 实施并提交**，以 `docs/reports/suoming-backend-gate-v1.md` 和当前 `main` 为基线；不要重新搭平台、不要重新安装所有模型、不要重跑与本轮无关的全量 Gate。
- 已有用户盲听评分已回填报告。VoxCPM2 **zero-shot 清澈度/磨砂改善/角色相似度/自然度 = 5/5/5/3**；但 VoxCPM2 **Codec roundtrip = 1/1/4/5**。此冲突需要诊断，不能以 zero-shot 一条高分直接宣布训练准入。
- **本轮唯一执行目标：Phase 5A — VoxCPM2 小样本复现、Codec 异常排查、Prompt A/B 与训练决策报告**，详见第 23 节。优先复用本地环境与模型，不开展 LoRA/SFT。
- Dots 和 Qwen3-TTS 继续作为既有 baseline；不得因此删除后端、改动旧 `dotstts` 仓库或重构通用架构。
- 有效命令：Devin 拉取 `main` 后按第 23 节执行，完成后提交诊断代码、报告与脱敏评分元数据，**停在用户训练决策门前**。

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

## Phase 5 — 用户试听与单后端验证

**Phase 5A（当前允许执行）**：仅验证 VoxCPM2 的音质可重复性、Codec roundtrip 低分原因、不同参考音频对自然度的影响；详见第 23 节。

**Phase 5B（当前禁止执行）**：训练方案设计/小规模 LoRA/SFT 或 checkpoint 对比。必须在 Phase 5A 报告交付、用户确认后另行授权。当前即使存在 5/5/5/3 的 zero-shot 高分，也不构成训练许可。

Qwen3-TTS 和 Dots 保留现状作为对照，不自动启动额外训练。

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
5. 若 Phase 0–4 已在 Git 历史中完成，则**直接执行第 23 节的 Phase 5A**，不得为了“按计划”重复全部 Phase 0–4；
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

**当前轮次仅限 Phase 5A；不得开始任何 LoRA/SFT 训练。**


---

# 23. Phase 5A — 锁暝 VoxCPM2 稳定性与 Codec 异常专项（当前执行）

## 23.1 基线与问题定义

用户在本项目 \`docs/reports/suoming-backend-gate-v1.md\` 记录的真实盲听结果（评分 1–5，**越高越好**；\`grit\` 分数高表示磨砂问题更少，并非磨砂更多）：

| 样本 | clarity | grit（少磨砂为高） | likeness | naturalness |
|---|---:|---:|---:|---:|
| 克隆参考原声（谛天鉴） | 5 | 5 | 5 | 5 |
| validation 原声（午后凉风） | 3 | 3 | 4 | 5 |
| Dots Step500 LoRA | 5 | 5 | 2 | 2 |
| Dots zero-shot | 5 | 5 | 1 | 2 |
| VoxCPM2 Codec roundtrip | 1 | 1 | 4 | 5 |
| **VoxCPM2 zero-shot** | **5** | **5** | **5** | **3** |
| Qwen3-TTS Codec roundtrip | 1 | 1 | 4 | 5 |
| Qwen3-TTS zero-shot | 1 | 4 | 3 | 3 |

上面的数据是**用户对既有样本的评分，不是新实验预测**。禁止篡改、平均成“总分”或宣称 VoxCPM2 已通过所有音质 Gate。

待回答的三个核心问题：

1. VoxCPM2 zero-shot 的高分，在**不同文本和随机种子**上是否稳定？
2. VoxCPM2 原声 Codec roundtrip 的 1/1 是**接入错误、输入采样率/解码处理错误，还是 Codec 本身在这类游戏录音上的重建局限**？它与 zero-shot 的巨大差异是否可信？
3. 在不训练、不后处理、不调整模型权重的条件下，**更换 Prompt** 能否改善当前仅 3 分的自然度，同时保持角色相似度和清澈度？

**严禁使用单条主观高分推出“底座必然更好”，也严禁使用一条 Codec 低分推出“所有微调必然失败”。**

## 23.2 范围与预算

本轮仅允许以下 bounded cases：

- 稳定性集：**4 个测试文本 × 3 个 seed = 12 个 VoxCPM2 zero-shot WAV**。
- Prompt A/B 集：**3 个 Prompt（含现用 Prompt） × 2 个固定测试文本 × 1 个固定 seed = 6 个 WAV**；如与稳定性集完全同配置，可复用已有 WAV，不重复推理。
- Codec 专项：先**复查现有代码/已有 WAV**；只对 **当前 validation anchor + 另一条独立干净原声**做受控重建。必要时可为同一条音频新增不超过 2 种有明确假设的诊断变体（例如采样率处理 A/B），必须记录变更，不做广泛 sweep。
- Dots/Qwen：仅复用既有 baseline，不再为它们生成新一轮覆盖性测试。
- 若模型一次性加载后出现偶发 OOM、坏句，保留失败证据，不通过无限重试“刷高成功率”。

**预算上限：不计必要的人工重跑和代码单元测试，VoxCPM2 新生成的 zero-shot 案例原则上不超过 18 个；Codec 诊断只按上述限定范围。** 如扩展必须在报告中解释理由并停止待用户确认。

本轮不安装额外大型 ASR/TTS 模型。若本地已有可复用 ASR，可用作辅助检查；无则先给人工逐字核对 UI，不得因 ASR 缺失而假装文本正确。

## 23.3 任务 A — 冻结参考数据与公平复现

开始前：

1. 拉取最新 \`main\`，确认旧 Phase 0–4 文件不被覆盖。
2. 确认 \`configs/characters/suoming.yaml\` 中 clone Prompt（**谛天鉴**）和 held-out ground truth（**午后凉风**）分离，并验证 \`sha256\`；不要把待评测原声当 clone Prompt（数据泄漏）。
3. 保留既有 \`outputs/gates/suoming_v1/\`，**本轮放到新目录**：\`outputs/gates/suoming_voxcpm_phase5a/\`。
4. 冻结 base model revision、\`voxcpm\` package version、设备/精度、现用 inference_timesteps/cfg_value、参考音频与文本；不添加 denoiser、EQ、loudness polish，不静默启用 retry 或改变参数。
5. 创建 machine-readable experiment manifest，每个 case 记录：case_id、text_id、text、prompt_id、prompt hash、seed、effective generation args、model revision、sample rate、output hash、耗时、失败原因和 retry count。
6. 输出波形原始采样率，允许另导出仅供盲听的统一响度副本，但必须保留 raw，并且 A/B 不得混淆 raw 与 matched 文件。

为了检查自然度与稳定性，本轮生成时必须记录调用 \`retry_badcase\` 是否启用、底层是否真的重试以及次数；做不到就标为 \`unknown\`，不能宣称“一次生成成功率”。

## 23.4 任务 B — 稳定性集（12 WAV）

从已审查锁暝文本中选 **4 个测试文本**，按以下不同语境：

1. 现有“午后凉风……”：固定锚点（有独立 ground truth）。
2. 1 条短句：接近游戏内简短回应。
3. 1 条中等长度、连续说明文本：检验连读与韵律。
4. 1 条带停顿/情绪变化的台词：检验失稳和异常发声。

优先使用未参与训练的 validation/test 文本，标记 split；如采用新写文本，也明确标记 \`unseen_new_text\`。不得用属于当前 clone Prompt 的语音作“独立 ground truth”。

固定同一个参考 Prompt（当前谛天鉴）和所有采样参数；每条文本使用 **seed = 42、43、44**。生成命名应可反推 text/seed，如 \`stability/rain_seed42.wav\`。

对每个案例检查：

- WAV 可读、无 NaN/Inf、无全静音、无明显截断/削波；
- 句子是否完整：漏词/错词/吞字/异常插话/重复字，优先人工对照，有现成 ASR 则统计 CER 作为辅助；
- 发音异常：咬字、舌位、顶嗓/前顶、气声或“磨砂”、发音形态突变；
- 自然度：韵律、停顿、重音、尾音衰减；
- 角色相似度：相同文本不同 seed 的音色漂移；
- 每条实际运行耗时、raw 输出时长、是否自动 retry。

盲听表每个样本至少保留：\`clarity\`、\`grit\`（高分为好）、\`likeness\`、\`naturalness\`、\`text_complete\`（是/否/待审）、\`artifact_type\`、\`notes\`。

可计算客观频谱指标与时长差异，但不能把其变成代理“听感总分”；句子短不应自动被判定不稳。**未得到用户本人评分时全部记为 \`PENDING_USER_LISTENING\`，不可由 Agent 编造。**

## 23.5 任务 C — VoxCPM2 Codec roundtrip 1/1 低分专项

目标不是“把 Codec 指标调好看”，而是分清**实现 bug 和模型能力边界**。

要求：

1. 检查 \`workers/voxcpm_worker.py\` 的 \`handle_codec_roundtrip\` 与所安装 **voxcpm 2.0.3 的实际 AudioVAE V2 API**，逐项验证 tensor 形状、输入采样率、\`preprocess\`、\`encode()\`、\`decode()\` 返回结构、设备/dtype、解码真实输出采样率。
2. 核对 **16 kHz encode / 48 kHz decode** 的非对称链路；必须明确限制在 16k 采样输入时，原始 8kHz 以上的细节不能凭空恢复。高频指标变化应区分“原始数据不可恢复”和“实现错误/声码器失真”。
3. 对 ground truth 与 roundtrip WAV **先做延时对齐**，并对齐增益后再比较相同片段；另保留不对齐 raw 用于实际试听。
4. 比较 ground truth / 编码前 16k 输入（可供试听的重采样版本）/ Codec roundtrip 三条链；辨别纯下采样损失与 Codec 引入的额外颗粒、金属感或失真。另一条独立参考音频重复该诊断，避免单条特殊录音误判。
5. 给出共同有效频带内（例如 0–7.5kHz，必要时分 0–4k / 4–7.5k）的谱差、相关性/重建残差、能量和时间对齐证据，外加至少一张对齐的 spectrogram；仍以人工听感为准。
6. 检查 zero-shot 实际调用的 decoder 与这个 roundtrip 是否是同一条解码路径；若输入分布、条件或推理方式不同，明确给出差异，**不能把 roundtrip 低分直接外推到生成音质**。
7. 不得通过新加降噪、EQ、HF shelf、spectral stabilizer 掩盖问题；如发现确凿实现 bug，写回归测试并修正 worker 后**仅重跑必要 Codec case**。

本任务必须输出明确分类：

- \`implementation_bug_confirmed\`
- \`codec_reconstruction_limit_supported\`
- \`unresolved\`

每个结论要列证据与反证；不能只看 flatness 相似与否。

若仍为 \`codec_reconstruction_limit_supported\`，后续训练 eligibility 保持“需要用户批准的例外/风险”，不能由 Agent 自动判为通过。

## 23.6 任务 D — Prompt A/B（6 WAV）

在 \`dotstts\` 已审查音频资产中选 3 条：

- **P0**：当前 25s 谛天鉴 Prompt（必须保留基线）；
- **P1**：质量较高、较干净、发音自然的候选；
- **P2**：不同语气/句式但说话人一致的候选。

要求：

- 每个 Prompt 需有严格匹配的文本，逐字检查；有参考 hash、持续时长、来源及数据 split。优先无杂音、无角色串音、无爆音；不要默认越长越好。
- 同一 Prompt A/B 时固定模型、sampling、seed=42，使用 2 个文本（包括雨景锚点与另一个文本），保证唯一变量是 Prompt。
- 默认不做降噪、时间拉伸、动态压缩、EQ 等改变参考音色的处理。
- 依赖用户盲听判断是否更自然且不牺牲角色相似、清澈度。
- Prompt 必须与 ground truth 独立；不能拿锚点真实原声作为 Prompt 来生成同一句以取得虚假的相似度。

即使出现更高分，也不应直接修改用户日常 production 默认 Prompt；先将候选写为显式 \`candidate\`，待用户确认后切换。

## 23.7 任务 E — 修正试听评分资产与实验可追踪性

当前仓库报告已包含用户此前导出的评分摘要，但是需要保留**可机读、脱敏的原始评分记录**。

- 优先从本地现有 \`outputs/gates/suoming_v1/listen/listen-ratings.json\` 读取；若不存在，保留报告中的评分作为 \`report_transcription\` 并标明不是原始 JSON，**禁止杜撰原始评分文件**。
- 经脱敏后保存到 \`docs/reports/suoming-gate-v1-user-ratings.json\`，包括来源、评分方向（高分代表改善）、样本相对路径、原有四维分数与已有备注；不存机器私有绝对路径、账号或 token。
- 新试听页可以导出 Phase 5A 评分，包含 \`experiment_id\` + \`case_id\` + \`output_sha256\`，防止重生成后旧评分套在新 WAV 上。
- 静态试听包按 \`ground truth / baseline / stability / prompt AB / codec diagnostics\` 分类，**同时保留匿名盲听视图与可解盲映射**。人工主观栏默认 \`PENDING_USER_LISTENING\`。
- 修复已知残留风险：重跑同名 case 时必须保证对应 WAV 是本轮新生成或经 sidecar hash 和 case manifest 明确验证为合法复用；不能因“已有同名 WAV”而假报 \`ok\`。
- 尽量不改 WebUI 大框架，只复用现有 listen page / gate 工具；不要引入复杂数据库。

## 23.8 任务 F — 训练准入决策与退出条件

机检指标仅判断“可运行/可复现”；真正的音质 Gate 需用户评分：

| 维度 | 阶段性要求 | 由谁判断 |
|---|---|---|
| 稳定性 | 12 个样本都有明确完整性/异常记录；若有坏句保留案例，不无限刷 seed | Agent 提交证据，用户确认听感 |
| 自然度 | 不再只依据一条 3 分样本；要给用户可比较的多句 Prompt A/B 证据 | 用户 |
| 清澈度/磨砂 | 与现有 5/5 的高分 zero-shot 基线 A/B；新 case 不可被响度/后处理美化 | 用户 |
| 角色相似度 | P0/P1/P2 各有同文对照；避免因 Prompt 泄漏而虚高 | 用户 |
| Codec 异常 | 1/1 低分原因按 23.5 分类；若未解决，报告保留训练风险 | Agent 诊断 + 用户审阅 |
| 数据 | 训练集没有被本轮修改；所有样本来源、split、hash 可追踪 | Agent |

输出训练建议只能三选一，附可审计理由：

1. \`RECOMMEND_VOXCPM2_LORA_PLAN\`：稳定性和用户试听支持，Codec 风险已解释/经用户接受；
2. \`MORE_EVIDENCE_NEEDED\`：有明确未解决的声音或链路问题；
3. \`DO_NOT_TRAIN_YET\`：多 seed 大量不稳定、文本错误、音色漂移或明显声学上限。

**即使是选项 1，也只是“建议制定下一轮训练方案”，本轮不启动 LoRA/SFT。训练与是否接受 Codec 风险均需用户明确决定。**

### 停止规则

- 某根因假设最多做有限 A/B，连续 2 次不能提供新信息应停止该分支并标记 \`unresolved\`；
- 不得因为要把分数刷到 5/5 而自训练、无限换 Prompt、无限跑 seed；
- 成功的技术定义是生成**可信试听证据和清楚的下一步选择**，不是宣称质量问题解决。

## 23.9 本轮产物与提交

必须新增或更新：

- \`configs/evaluations/suoming_voxcpm_phase5a.yaml\`（或等价机器可读 manifest）；
- \`scripts/run_voxcpm_phase5a.py\`（或在现有 gate 中增加专用入口，避免复制旧业务代码）；
- \`docs/reports/suoming-voxcpm-phase5a.md\`（含 12-case 表、Prompt A/B 表、Codec 归因证据、异常实例、下一步建议）；
- \`docs/reports/suoming-gate-v1-user-ratings.json\`（若原始数据可用则真实转换，否则标记 \`report_transcription\`）；
- \`outputs/gates/suoming_voxcpm_phase5a/manifest.json\`、\`metrics.json\`、\`listen/index.html\` 和本地 WAV；大 WAV **不直接提交普通 Git**；
- 对新增诊断/重用逻辑的单元测试，以及必要的真实 backend smoke 结果。

报告末尾必须有：

1. **VoxCPM2 原因分类**：Codec 1/1 究竟是何种证据支持的结果；
2. **零样本稳定性**：12 个 case 的状态、重要异常、用户未听则 \`PENDING_USER_LISTENING\`；
3. **Prompt 选择**：P0/P1/P2 的利弊，未听则不擅自选赢家；
4. **训练建议**：三种枚举之一，以及明确的 \`PENDING_USER_DECISION\`；
5. **可复现性**：git commit、模型 revision、prompt sha、seed、case 文件、测试结果；
6. **BLOCKED / 未完成**：准确说明，不以“没有报错”冒充质量合格。

最后提交 GitHub，并向用户报告 commit 与试听页在其**本地工作目录**的真实位置。如果试听 WAV 未提交 Git，必须直说它们仅在本地存在；不编造在线可打开的 GitHub WAV 链接。

## 23.10 Devin 本轮执行顺序

1. \`git pull\` 最新 \`main\`，通读本节与现有评价；不要重复 Phase 0–4。
2. 校验本机旧资产、本地模型和现有评分来源。
3. 优先完成 Codec 1/1 异常的**静态代码检查和已有录音分析**，避免先重复下载/渲染。
4. 实施受限 Codec 实验，保留结果与测试。
5. 执行 12-case 稳定性集合、6-case Prompt A/B；生成新试听页面。
6. 计算描述性指标、逐条验证音频存在与 SHA256，写清缺失的人工评分。
7. 生成评测报告、运行测试、提交 GitHub。
8. **停止在用户听评与训练决策门前**，不启动微调、不扩展新后端。

网络仍遵守：**本地缓存 → 可用镜像 → 镜像失败后 127.0.0.1:7897 代理**；代理仅限相关进程，勿修改全局设置。
