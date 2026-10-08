# TTS 多后端角色语音平台实施计划

> 仓库：`NiHaiMa123/tts`
>
> 目标：把当前围绕 DotsTTS 积累的角色语音数据、质量审查、诊断、试听和 WebUI 能力抽成一个**与底座无关的角色 TTS 平台**。DotsTTS 作为 legacy baseline；新增 VoxCPM2、Qwen3-TTS。任何新底座必须先通过 codec / zero-shot 准入测试，再允许进入 LoRA / SFT。
>
> 当前优先角色：锁暝。后续同一平台复用守岸人、符玄等角色数据。

---

## 当前执行状态与优先任务（2026-10-08，Phase 5B Pilot）

- **Phase 0–4、5A、5A2 已完成且已有人耳评分**。最新 Phase 5A2 评分提交：`a3f8e57`；历史结果不得覆盖，详见 `docs/reports/suoming-voxcpm-phase5a2.md`、`docs/reports/suoming-voxcpm-phase5a2-ratings.json`。
- 用户说明**多次连续听非常相似的 TTS 音频后会感觉难以听出区别**。同 SHA 同音频跨场次分数剧变主要反映听觉适应/疲劳、试听上下文和评分尺度漂移，**不是音频重新合成出了不同缺陷**。既有分数保留作为原始观察，后续不得将其机械计算成真实坏句率或据此认定 P2 稳定失败。
- **当前唯一执行目标：第 25 节 Phase 5B —— VoxCPM2 单次、受控、低预算 LoRA pilot + 少量高可信 A/B 评测**。不再做 Phase 5A3/连续刷 Prompt/种子，优先自动筛查与首次印象，降低人的试听负担。
- P0（谛天鉴）仍是生产参考基线；P2（回到华亭）只作候选。Codec 重建失真已确认且记录为**试训风险**：这次试训用于验证 TTS 生成改善潜力，不代表接受 Codec 失真作为最终质量。
- 保留三个独立 backend 和当前 WebUI，**不重构基础平台、不修改旧 dotstts、不训练 Dots/Qwen、不更新生产默认 LoRA/Prompt**。
- Devin 拉取 `main` 后**只执行第 25 节**；最多 150 个优化器步骤与 3 个训练检查点，生成有界试听证据后停止，等待用户最终审核。若不具备可靠数据/显存/官方 API 条件，标明 BLOCKED，不勉强训练。
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

如果 codec roundtrip 明显失真，默认停止训练投入；但**codec roundtrip 质量与生成质量不等价**。对 VoxCPM2，Phase 5A 已通过原声重建/zero-shot 分离分析确认这两条路径的差异，本项目允许仅做第 25 节明示的**受控 LoRA 小试验**，风险写入报告，绝不视为 codec 问题已解决或已批准生产。

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

一般情况下必须：

`codec_gate != fail`

且 zero-shot 不存在结构性质量问题，才进入；**VoxCPM2 属于经用户要求规划的小规模研究性例外**，需完整保留 codec 重建失真风险、对照和停止条件（第 25 节）：

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

**Phase 5A（已完成）**：VoxCPM2 多 seed、Codec 归因、P0/P1/P2 对照及人工评分已完成；以 `d4c76d6` 报告为历史基线，不能重新跑后覆盖。

**Phase 5A2（已完成）**：P0/P2 双短句多 seed 与评分一致性复核已完成；同音频跨场次评分变化很大且用户报告听觉适应，因此不得将旧单项评分作为硬性失败率；详见第 24 节与评分文件。

**Phase 5B（当前 ACTIVE）**：仅执行第 25 节明示的**一次受控 LoRA pilot**。允许训练的只有 VoxCPM2 LoRA，最多 150 optimizer steps、3 个保存点；是否继续更多步数、提升为生产版本、扩大数据/模型须由用户再次明确授权。

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

# 20. Phase 0–4 历史禁止事项（当前以第 25 节为准）

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

# 21. Phase 0–4 历史交付（当前以第 25 节为准）

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

# 22. 历史 Devin 执行规则（当前以第 25 节为准）

开始时：

1. 拉取最新 `main`；
2. 阅读完整 `PLAN.md`；
3. 检查本机旧 `dotstts` 路径和已有模型/cache；
4. 优先复用本地 cache，避免重复下载；
5. Phase 0–5A2 已完成，本轮**只执行第 25 节 Phase 5B Pilot**；不得重复历史实验或覆盖旧评价；
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

**本轮仅按第 25 节允许一次受限 VoxCPM2 LoRA pilot；其余模型训练仍禁止。**


---

# 23. Phase 5A — 锁暝 VoxCPM2 稳定性与 Codec 异常专项（历史已完成）

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


---

# 24. Phase 5A2 — VoxCPM2 P2 短句稳定性及评分一致性复核（历史已完成）

> 状态：**COMPLETED**（2026-10-08；以 `a3f8e57` 的报告/评分为准）。以下为历史实验计划，已不再授权重复跑实验或限制当前第 25 节的 LoRA pilot。

## 24.1 冻结证据与未解问题

历史有效资产：

- Phase 5A 实施 commit：`270f109`；人工评分回填 commit：`d4c76d6`。
- 原始总结：`docs/reports/suoming-voxcpm-phase5a.md`。
- 用户评分：`docs/reports/suoming-voxcpm-phase5a-ratings.json`；已报告 22/22 评分记录与 manifest SHA 相符。
- P0 = `configs/evaluations/suoming_voxcpm_phase5a.yaml` 里的 25s「谛天鉴」参考；P2 = 28s「回到华亭」参考。P1 已两文本自然度 2，本轮不参加。
- Phase 5A：P2 × rain/mid_exposition × seed42，两句 c/g/l/n 均 5/5/5/5；**只覆盖 2 条文本、1 个 seed，尚未证明短句普适稳定**。
- P0 下：`short_response` 的 seed42、43、44 分别为 1/1/3/5、5/5/3/3、4/4/2/1；`mid_exposition_seed44` 自然度 1。
- **重复评分冲突**：`stability/rain_seed42` 与 `prompt_ab/P0_rain_seed42` 是同 SHA，却分别被评为 4/5/4/5 与 5/5/4/3；`mid_exposition_seed42` 在两组分别 5/5/5/5 与 5/5/5/4。相同音频在不同试听上下文中评分变化，属观察到的评测方差，**不能当成两个独立声学样本**。
- Codec roundtrip 1/1 已按双原声/双解码条件排查为 `codec_reconstruction_limit_supported`，保留训练风险，不重做 Codec、更不因 zero-shot 高分而宣称 Codec 恢复。

本轮只回答：

1. P2 能否在**同一句短句**的不同 seed 中，比 P0 更稳定、自然且像角色？
2. P2 的改善能否在**第二条独立短句**重现，而非只在一条文本上碰运气？
3. 对已有自然度 1 的 `mid_exposition_seed44`，换成 P2 是否改善？
4. 对同 SHA 不同分数，评测应如何保留上下文差异、避免重复样本权重？

## 24.2 严格受控实验矩阵（新增 WAV 上限 = 10）

| 组 | 文本 | Prompt | seed | 既有 WAV 复用 | 需新增 |
|---|---|---|---|---|---:|
| A | `short_response`（「开伞，由我来动手。」） | P0 | 42、43、44 | Phase 5A stability 3 条 | 0 |
| A | 同上 | P2 | 42、43、44 | 无 | 3 |
| B | 第二条独立短句 `short_response_2` | P0 | 42、43、44 | 无 | 3 |
| B | 同上 | P2 | 42、43、44 | 无 | 3 |
| C | `mid_exposition` | P0 | 44 | Phase 5A stability 1 条 | 0 |
| C | 同上 | P2 | 44 | 无 | 1 |
| **合计** | **2 条短句 + 1 条长句异常复核** | | | **4 条可复用** | **10 条新增** |

第二短句选择规则：

- 从**未用作 P0/P2 参考语音**的锁暝 validation/test 中选一条语义完整、简短且不同于「开伞」的台词；建议 6–20 汉字左右，避免单独的语气词；记录真实 `text`、split、源 manifest、若有录音则校验 SHA。
- 如果 held-out 中没有合适短句，可使用**人工明确新写的测试短句**，标记 `unseen_new_text`；禁止假称有真实录音/准确 ground truth。
- 不得把测试目标原声作为同次语音克隆的 Prompt，不允许文本泄漏。
- 不许因为某条比较差就替换文本或增跑 seed。若确需扩大预算，先交报告说明并停止。

**只变更 Prompt。** 固定 VoxCPM2 revision `32279effe8c19989596f05d353d1447f51d9e915`、已安装 voxcpm 2.0.3、采样方式、环境、cfg、inference steps、dtype、text、seed 等；参数原则上与 Phase 5A 相同：`cfg_value=2.0`、`inference_timesteps=10`、`normalize=True`、`denoise=False`、`retry_badcase=True`。记录真实 retry 次数；不能只根据 WAV 可读就说稳定。

注意 P0、P2 是**不同内容、不同长度的完整 Prompt 条件**；本次可验证“整个 Prompt 选项的效果”，不能把改善单独归因于长度、语速或文本某一项。

## 24.3 原文件复用与公平盲听

1. **在新目录运行**：`outputs/gates/suoming_voxcpm_phase5a2/`；现有 `suoming_voxcpm_phase5a` 只读。不得重命名、覆盖既有 WAV 或覆盖已导出的旧用户评分。
2. 对所有复用的四条 WAV，必须检查**磁盘实际文件 SHA256** 与旧 manifest 相符；同时校验同一 model revision、Prompt SHA、文本、seed、effective options、backend 版本。缺文件、hash 不匹配或参数不一致时标 `BLOCKED_REUSE`，**不能悄悄以新音频替代旧基线**；向报告写出如何恢复原始资产。
3. 对十条新增 WAV，记录 `case_id`、输入/输出 SHA、生成时长、sample rate、模型及 Python 包版本、实际 options、重试情况、生成耗时、wav sanity（长度/NaN/静音/削波）。输出成功需要本轮新的 sidecar 与真实 WAV hash，不可仅靠“路径上已经存在 WAV”判断。
4. 不对音频做 EQ、后期降噪、响度美化、改速、增混响、频谱稳定器、训练/权重修改。可以给盲听生成统一**播放响度**的额外副本，但保留 raw，并在评分标签里明确区分。
5. 试听页面按**同一文本配对 P0/P2**，不暴露型号/Prompt 身份的盲听标签；同一对尽量随机化左右位置，支持解盲、用户保存 1–5 四维评分及完整性/失真备注。
6. 评分导出绑定 `experiment_id + case_id + real_output_sha256`。**相同 SHA 的 WAV 在本实验只计一次音质证据**；旧 Phase 5A 里同 SHA 的评分差异要作为 `context_variance` 案例保存，不以平均后的数字掩盖，也不当独立样本加权。针对 P0/P2 可给出每句配对分数与例外说明，而不是做一个不可信的综合分。

## 24.4 文字完整性与异常分类

重点是先定位问题属于哪类，不能笼统标“坏句”：

- 文本：是否漏字、错字、重复、额外插话、句尾截断；
- 发音与声学：齿音/磨砂、幼态前顶、口型过大/空腔回声、声音粗糙或突变；
- 韵律：语速过快/过慢、停顿错误、重音不自然、尾音衰减、明显失真；
- 音色：是否保持锁暝声线、同文本 seed 之间漂移；
- 可用性：本地 raw 录音是否可读、是否完整、重试是否真实发生。

若环境已有可用中文 ASR，可用转写/CER 做**辅助证据**，并保留具体逐字 diff；否则试听页提供 `text_complete = yes/no/uncertain` 和 `transcription_notes`，交用户逐字判定，不能凭 WAV 时长宣称没有漏字。

特别把 `short_response_seed42` 的 1/1 声学异常和 `mid_exposition_seed44` 的自然度 1 拆分分析；**不能把不同缺陷混为同一种采样失败**。Agent 无听觉评分证据时，所有主观项目保持 `PENDING_USER_LISTENING`。

## 24.5 判定规则（先定义，再听结果）

比较的主要单位是**同文本、同 seed、P0 对 P2 配对**，共 7 组，其中 6 组短句、1 组长句。

- 若 **P2 在 6 对短句中多数提高自然度与角色相似度，且不明显损失清澈度/磨砂改善**，并且未出现新的严重文本/声学失败，可建议把 P2 设为后续 LoRA 的 **candidate reference baseline**。
- 若 P2 对长句 seed44 也改善，增强其泛化证据；若未改善，明确记录 seed 或文本特异性，不能整体宣传“已修复”。
- 若 P2 的短句失败率仍高、尤其再次出现清澈度/磨砂 1 分或自然度 1 分，结论为需要先解决稳定性或评估训练风险，不得从现有两句 5/5 推断稳健。
- 上述“多数”是**实验阶段预注册的建议性门槛而非统计显著性**；只有两条短句与三个 seed，不能推断生产长期故障率。
- Codec roundtrip 局限是单独维度，**本轮不重开 Codec，也不因 P2 通过就自动批准忽略 Codec 训练风险**。
- 训练决策三选一：`RECOMMEND_PHASE5B_LORA_PLAN`、`MORE_EVIDENCE_NEEDED`、`DO_NOT_TRAIN_YET`。即便推荐 Phase 5B，含义也只是**建议用户审核小规模训练方案**，本轮禁止真的启动训练。
- P2 只标记 `candidate`；不修改 `configs/characters/suoming.yaml` 的生产默认参考，也不改变日常 WebUI 默认行为，直到用户明确批准。

## 24.6 代码与交付范围

优先**复用**既有 `src/character_tts/evaluation/phase5a.py`、统一 worker、listen page、manifest、哈希校验和评分导出工具，不要复制成另一个巨大框架。合理新增：

- `configs/evaluations/suoming_voxcpm_phase5a2.yaml`：实验矩阵与版本/Prompt hash；
- `scripts/run_voxcpm_phase5a2.py`：轻量入口（也可合理扩展现有 CLI）；
- `docs/reports/suoming-voxcpm-phase5a2.md`：7 对配对样本表、逐条异常分类、P2 是否具有跨句改善证据、主观待审项和训练建议；
- `outputs/gates/suoming_voxcpm_phase5a2/manifest.json`、`listen/index.html`、`unblind_map.json`、sidecars、真实 WAV（本地，不进普通 Git）；
- 对复用哈希、真实 WAV 校验、重复 SHA 去重、配对 case 数量、`text_complete` 记录、输出路径隔离的单元测试。

报告必须列出：复用 4 条与新生成最多 10 条的明细、实际完成/失败数、每 case hash 与 retry、旧评分同 SHA 不一致说明、完整性检查状态、盲听入口、`PENDING_USER_LISTENING`（未评分时）以及 `PENDING_USER_DECISION`。

**不要把“运行成功”写成“音质通过”；不要杜撰用户评分；不要因缺 WAV 自动把当前轮所有基线重跑。**

## 24.7 停止规则及 Devin 执行顺序

1. `git pull` 最新 main；读取本第 24 节、Phase 5A 报告和评分 JSON。
2. 查本地模型、cache、旧 WAV/manifest，确定有 4 个可复用样本。旧文件不可获得则记录阻塞，不伪造数据。
3. 选并冻结第二短句；登记真实来源/文本/哈希，确认与 P0/P2 非同一句。
4. 实现最小差异的 7 配对用例与试听展示（严格不超过 10 条新增）。
5. 执行、核验文件级 SHA256、保存全部坏句、生成诊断 report。
6. 跑已有 tests 和新测试，提交 GitHub；原生产配置及旧 Phase 5A 不变。
7. **停止，等待用户对本轮试听样本评分和是否推进 Phase 5B 的决定。**

本轮不可：训练 VoxCPM2 / Qwen / Dots、新增后端、优化 Codec、无限扫 seed、重做 Phase 0–5A、套用 EQ 修好听感、修改生产默认 Prompt。

下载规则不变：**本地 cache → 镜像优先 → 镜像失败再走 `127.0.0.1:7897`**。不得修改全局代理。


---

# 25. Phase 5B — VoxCPM2 LoRA 小规模可行性试验与低疲劳盲听（ACTIVE）

> **唯一当前任务。** 这是研发 pilot 而不是产品质量验收；允许 Devin **在本地执行一次受限 LoRA 训练**，但不授权长期训练、不允许自动扩大训练步数或更改正式生产模型。用户已报告连续听相似音频会逐渐“听不出了”，因此试听设计必须降低听觉适应/疲劳造成的偏差。

## 25.1 决策依据与目标

历史证据：
- VoxCPM2 zero-shot 已能生成很清澈且接近锁暝的声线；局部存在语速、空腔/口型和短句韵律异常。
- P2 在 Phase 5A 个别句子明显优于 P0，但 Phase 5A2 配对结果方向不稳定；**不要把之前 P2 2/2 全 5 分当成普适优势**，也不应把同一 WAV 跨场次分数变化当成模型随机生成失败。
- 同一 SHA WAV 在不同试听情景中四维评分有明显变化；用户明确反映连听类似语音后听觉分辨力下降。
- VoxCPM2 AudioVAE 真实音频 16kHz 编码、48kHz 解码的 roundtrip 已存在主观可闻失真，LoRA **不能保证修复 codec**；TTS 生成 latent 路径与原声 encoder roundtrip 路径不同，值得做成本受限的生成质量试验。

核心研究问题只限：
1. **与完全相同的 zero-shot baseline 相比**，LoRA 能否在锁暝音色、自然度、稳定性、短句表现上带来**可闻且可复现**的改善？
2. LoRA 是否引入更强的磨砂、金属感、错误发音、音色漂移或过拟合？
3. 是否存在早期 checkpoint 已够好 / 继续训练反而变差的趋势，值得用户选择下一步？

**成功标准不是训练 loss 下降，也不是人耳一次打 5 分，而是可复现、可试听、可追踪的对照证据。**

## 25.2 官方训练接口与机器约束（先做 preflight）

优先调用 VoxCPM2 官方入口，而非发明自己的训练循环：

- 上游仓库：OpenBMB/VoxCPM；
- 官方脚本：scripts/train_voxcpm_finetune.py；
- 官方 v2 LoRA 模板：conf/voxcpm_v2/voxcpm_finetune_lora.yaml；
- 官方数据格式：JSONL，每条至少 {"audio":"...","text":"..."}，可带 duration / dataset_id；
- LoRA 能分别启用 lm、dit；不要擅自假定仅训练 LM 就等价于完整声学适配；
- 官方训练期编码 sample_rate 为 **16000**，输出监听是 **48000**，不要把 48k 训练音频直接错误当成 16k；
- 所有接口细节以**当前本地安装 voxcpm 2.0.3 对应的官方源码**为准，记录源码 commit、包版本及差异。如依赖不兼容，先修兼容/停下记录，不把绕路自研训练当完成。

目标设备 Windows / RTX 5080 16 GB。先做低成本训练 smoke 和一次完整的前向/反向/optimizer-step 测试，记录真实 peak_allocated/peak_reserved VRAM、精度、吞吐、是否用了官方 GradScaler/offload/checkpoint。**16 GB 能否训练成功必须以机器实测为准**，不能仅凭推理显存数推断。

只允许 VoxCPM2 当前固定 base revision：
32279effe8c19989596f05d353d1447f51d9e915。

先复用已有模型、源码、环境、数据。如果缺失：**本地缓存 → 镜像 → 镜像失败后 http://127.0.0.1:7897 代理**，仅影响子进程、不修改全局代理、不下载无关模型。禁止升级已验证的推理环境造成三个后端崩溃；优先训练独立环境，不能运行就如实 BLOCKED。

## 25.3 数据冻结、泄漏与质量门

1. **只使用旧 dotstts 锁暝数据集的 train split** 构建本次 VoxCPM2 训练 JSONL。validation/test split 不得进入训练。保留原始 train/val/test manifests，不重新分配以求“更好成绩”。
2. 严格按 sha256 + 文件路径/文本索引建立训练清单；校验 WAV 存在、长度、采样率、有效语音、空白/重复文本、重复音频、错 speaker、明显噪声和文字对齐。不能为了让程序跑通静默丢弃困难样本。滤掉样本必须有 reject_manifest 与原因计数。
3. 与冻结评测样本、当前 clone Prompt P0、候选 P2 的音频 SHA 作精确去重；优先对同源切段/重复台词做文本和音频相似度复核，避免同段泄漏。至少雨景、两句短句、中长句和情绪句不进入训练。
4. 验证文字规范、编码、绝对/相对路径和音频读取方式，与官方 JSONL loader 对齐；实际重采样在 official pipeline 内可追踪，避免重复失真。
5. 训练样本时长分布、总有效时长、拒绝率、说话人身份与原有质量标记一并入报告。若质量问题严重或剩余数据极少，**STOP: DATA_BLOCKED**，而不是启动高风险训练。
6. 固定 prompt **P0** 为训练前后对照的参考音频；P2 不进入训练、也不偷换成 baseline。P2 若作为额外推理诊断，只能单独标记为另一个条件。

训练源数据不复制进 Git。写入训练 manifest 应保留文件 hash 和来源索引，不泄露私有绝对路径到公开报告。

## 25.4 本次训练预算、默认超参和失败停止

**只允许 1 次 LoRA pilot**，不得自动搜索 rank、学习率、LoRA target、训练轮数，不全参 SFT，不在第一个结果不好时立即再训一轮。

建议配置基于官方 v2 LoRA 模板的保守起始值（经 preflight 可为显存单次调整，但必须记录变更和理由）：

| 字段 | Pilot 设置 |
|---|---|
| pretrained_path | 已固定的本地 VoxCPM2 base 快照 |
| train_manifest | 25.3 冻结的 train.jsonl |
| val_manifest | 独立 validation manifest，仅供记录/计算，不得训练 |
| sample_rate / out_sample_rate | 16000 / 48000（遵循官方） |
| LoRA modules | lm=true, dit=true, proj=false |
| LoRA r / alpha / dropout | 16 / 16 / 0.0（试验值，不宣称最优） |
| micro batch / grad accum | 1 / 8（如显存不够，优先参照官方支持的减载方式） |
| learning_rate | 1e-4 作为待验证起点，发生训练数值异常即停止，不得靠反复试错无限调参 |
| max optimizer steps | **150**（硬上限，包括恢复训练的累计步数） |
| save interval | 50 optimizer steps，最多 step50、step100、step150 三个候选 |
| validation/logging | 有独立验证数据时每 50 step 记指标；不能用 loss 自动评定最终音质 |
| GPU | RTX 5080 16GB 单卡；不得假定支持所有 CUDA/Triton 扩展 |

具体配置须先与上游 parser 核实：max_steps/num_iters/save_interval/valid_interval/warmup_steps/max_batch_tokens 等字段含义、真正计算的是 optimizer step 还是 microstep。允许做训练前 **1–2 步 smoke** 并保留日志，正式训练步数计入 150 步总预算；绝对不得以“迭代数”偷换真实 optimizer step。

**一次 OOM 回退**：仅允许记录原配置后，按官方支持的 batch/sequence/filter/caching 策略做一次有证据的减载；不能静默删长句或改目标采样率。仍 OOM 就记 **BLOCKED_16GB**，不继续盲试、不默认开 CPU 超长训练。

出现下述情况立即停止，保存失败证据，不把失败显示成通过：
- NaN/Inf、梯度/权重异常、持续 loss 爆炸；
- 模型/revision、数据 sha、speaker 识别不符；
- 训练结果写入生产路径、意外覆盖旧权重或训练集；
- 超过 GPU 资源限制、步数预算、第二次 OOM 或多次不可复现崩溃。

每次 checkpoint 需保存：optimizer step、model/adapter hash、official source revision、training args、有效数据 hash、loss/val loss、time、VRAM、resume 信息。不存在则标缺失；不得编造。

## 25.5 训练后推理：Zero-shot vs LoRA 必须公平

LoRA 训练并不意味着“自然度一定上升”。在不动生产后端的条件下，新增**独立评测 profile / 可选 adapter 路径**，加载 step50、100、150，固定 base / reference / generation args / seed、raw waveform、normalize 等。

**固定评测集合（4 条文本，6 个 case/候选）：**

| 文本 | Seed | 理由 |
|---|---|---|
| rain（雨景） | 42 | 原声独立 held-out，检验声音纹理 |
| short_response（开伞） | 42、43 | 历史短句问题 |
| short_response_2（有什么在跟着我们） | 42、43 | 独立第二短句 |
| mid_exposition（中长解释） | 42 | 长句连续韵律 |

将 pause_emotion 作为**预留补充样本**，本轮默认不自动扩展；若四句不含明显自然度挑战，报告说明不足即可。

- base zero-shot **6 个 case**：历史 WAV 只有在每项 model revision / reference SHA / text / seed / effective args 全部匹配且磁盘 SHA 校验通过时才能复用，否则新生成并记录；
- step50/100/150：每个 checkpoint **6 个 case**，共最多 **18** 条新增 LoRA 音频；
- 本轮评测最多 **24 条新生成 WAV**（6 base + 18 LoRA；不含官方 smoke/训练时内部 sample log）；且评估页面不得强迫用户听完 24 条；
- 不允许把训练期间的“teacher-forced reconstruction”拿来和 zero-shot 自由生成混作同一结果；必须通过正式推理路径加载 adapter 后合成；
- 推理参数与 zero-shot 基线一致，仅 adapter/checkpoint 是变量。P0 始终是固定 clone reference。
- 对于每个 WAV：写 sidecar（checkpoint、LoRA target、data hash、prompt sha、model revision、seed、retry、raw sha、耗时），同时保留训练失败/生成失败样本。

## 25.6 自动质量筛查先行（不要让用户反复听）

自动筛查分层，**只承担“筛选风险/标记可疑片段”，不得直接等同人耳音质评价**。

A. 声学/工程硬错误（可自动判断）：
- 文件可读、声道采样率、duration 合理性、NaN/Inf、全静音、异常截断、clipping；
- 每条是否发生内部 retry / badcase，推理参数与参考哈希是否一致；
- 单声道试听路径、音量/响度差不能成为“听起来更清晰”的混淆项。

B. 文本完整性：
- 已有 SenseVoice 能复用则跑 ASR + CER/字词差异与对齐定位；
- 专有名词/同音字（如 心月狐→新月湖）不能仅以 ASR CER 判为 TTS 读错；
- 标出「漏字、重复、错读、截尾」嫌疑，保存转写原文和时间位置，交用户确认；
- 不引入新的巨大 ASR 依赖。

C. 主观声音异常辅助定位：
- 可导出音频波形、spectrogram、明显不连续片段的时间戳；
- **不要**凭 flatness 单一数字判定磨砂、舌头后缩、口型过大、机械共鸣、失真；不做缺乏标签的“自动模型自评分”冒充真实音质；
- 若无可靠检测器，标为 **UNDETERMINED**，只提供异常候选供首次听感确认。

将自动检测结果压缩成**可疑片段清单**，而不是让用户连续听几十条完整相似音频。

## 25.7 新试听协议：低疲劳、首次印象、成对决策

原来四维 1–5 分可以保留为可选，但**不能再作为唯一或主决策依据**。

必须提供一个基于现有 listen page 的最小试听流程：

1. **先听用户此前最熟悉的干净参考原声短片段**，然后随机、隐藏模型标签，播放同文本相同 seed 的 A/B（base vs checkpoint）；不同时混入 P0/P2 Prompt 比较；
2. 每次默认只呈现 **3–5 对**，其他样本折叠，用户可显式进入下一组；不要求同一晚全部听完；
3. 每对用户先选 **A 更好 / B 更好 / 无明显差异 / 暂无法判断**，再选择最突出的问题标签，如「磨砂/粗糙、前顶/舌位、空腔/口型大、语速/停顿、音色不符、漏字、其他」；
4. 支持“**第一次听到时立即标记缺陷**”并记录 \`first_impression\`、\`defect_type\`、\`severity\`、\`segment_start/end\`（时间戳选填）；不要强迫回放到听习惯为止；
5. 若用户表示“听麻木了/分不出来”，允许直接点 \`UNCERTAIN_FATIGUE\` 并停止本轮，**不要把‘无差异’替代疲劳不确定**；
6. 支持回到干净 reference 再校准、单独复核一次重大缺陷；不强制长时间重播；
7. 隐藏 checkpoint 身份并在不同轮次平衡 A/B 左右顺序；识别同 SHA 的重复引用，禁止当作两个独立声音结果；
8. 导出：experiment_id、pair_id、A/B 的真实 sha/checkpoint、盲测顺序、用户选择、first_impression、缺陷标签、optional 分数、是否疲劳/不确定、时间信息、记录时间和评测批次；
9. **不合成或自动生成用户评价**；未听标 \`PENDING_USER_LISTENING\`。

首次听感是用于减少疲劳的操作方法，**不是客观真值**。只有稳定复现的缺陷或多组同文配对一致趋势，才能作为更强证据；对不确定/疲劳样本不强迫判输赢。

### 分阶段向用户呈现，而不是灌满 24 条

- **第一包（推荐）**：base vs step100，对固定集合选 4 条（两短、一中、一雨），默认最多 4 对。
- **第二包（仅如有必要）**：step50 vs step100 或 step100 vs step150，在**最有区分力的 1–2 条文本**做最多 2–4 对；这一步只生成浏览页面的可选比较，不意味着追加训练。
- 只有用户愿意继续，才扩展播放剩余固定样本。既有 WAV 保留，避免再生成和再次听疲劳。

不要把 3 个 checkpoint 的训练 loss 排名直接当人耳优劣排名，尤其不能使用纯均分把复杂缺陷冲淡。

## 25.8 Pilot 验收与停止后决策

工作完成至少满足：

- 官方 VoxCPM2 v2 LoRA 训练入口得到确认，数据清单可审计，训练隔离不破坏其他 worker；
- 训练实测能在 16GB 下运行，若不能则如实 BLOCKED 而非伪造或无限 offload；
- 不超过 150 optimizer steps，最多三个 checkpoint，具备模型/数据/参数/provenance；
- 推理生成至多 6 个 base + 18 个 LoRA 固定对照，输出真实 WAV 与校验 hash；
- 自动质量筛查生成可定位异常的对照清单，试听页面支持**每批 3–5 对及疲劳/不确定**；
- 不假设 LoRA 必然提高自然度；没有人耳证据时全部标 \`PENDING_USER_LISTENING\`，不允许自动升级默认 adapter；
- 训练结束**立即停止**，不要自动跑第二轮 rank/LR/steps sweep。

报告区分四种状态：
- \`PILOT_READY_FOR_USER_AB\`：样本与盲听包已生成，等用户审听；
- \`BLOCKED_16GB\` / \`DATA_BLOCKED\` / \`API_BLOCKED\`：训练无法可靠进行，附具体失败证据；
- \`PILOT_NO_CLEAR_GAIN\`：**仅在用户真实 A/B 评审后**才能写，不能根据 loss 宣称；
- \`PILOT_PROMISING\`：**仅在用户真实 A/B 评审后**才能写，仍需后续扩大数据/验证。

若首轮 LoRA 不明显优于 zero-shot，就优先保留 zero-shot；不要陷入“差一点、再训练 1000 steps”的循环。

## 25.9 交付与 Devin 任务顺序

在现有仓库添加**最小功能**，不要复制完整 OpenBMB/VoxCPM 到平台源码：

- \`configs/training/suoming_voxcpm_lora_pilot.yaml\`：官方 v2 配置的可审核副本/映射，并注明确切上游版本；
- \`scripts/prepare_voxcpm_lora_data.py\`：从旧 train split 导出训练 JSONL、sha/reject 清单与防泄漏核验；
- \`scripts/train_voxcpm_lora_pilot.py\` 或 \`.ps1\`：preflight、官方训练入口、日志、资源记录、最多 150 steps、失败退出/有限回退；
- \`scripts/evaluate_voxcpm_lora_pilot.py\`：加载原始 base 与三个 LoRA checkpoints，用共同条件输出 WAV 与 sidecar；
- \`docs/reports/suoming-voxcpm-lora-pilot-v1.md\`：训练前数据统计、训练配置、step/loss/VRAM、每检查点样例/初筛，明确 Codec 风险和未人工验证之处；
- \`outputs/gates/suoming_voxcpm_lora_pilot_v1/\`：本地 adapter、logs、manifest、metrics、WAV、listen/index.html；默认 gitignored，不提交原始语音/模型权重；
- 测试：train-only 过滤、防泄漏、步数硬上限、失败退出、hash、LoRA adapter 正确加载、训练前后配置同一性、盲测 A/B 去重、\`UNCERTAIN_FATIGUE\` 导出行为。

**执行顺序（严格）：**
1. 拉最新 main；阅读本第 25 节、Phase 5A/5A2 报告及最新人耳备注；
2. 核对官方 v2 LoRA 模板/训练器，确定本地已固定 base revision，准备 train-only 数据与 manifest；
3. 显存/训练 1–2 step smoke；失败记录 STOP，不能无限升级训练工具；
4. smoke 通过后，执行**一次**不超过 150 optimizer steps 的 LoRA pilot，保存 step50/100/150；
5. 加载各 checkpoint 跑受控 raw 推理，生成 baseline 与 18 个以内 LoRA 测试音频，自动初筛；
6. 生成低疲劳 A/B 页面，默认先呈现 4 对，导出 SHA 绑定评分；不要代替用户评审；
7. 跑测试，提交 GitHub 的代码、manifest 模板/脱敏统计和诊断报告；
8. **到此停止**，回报 commit、报告、可在本地打开的试听页与 BLOCKED/待人工项。后续是否扩训/升生产由用户决定。

本轮禁止：修改 dotstts、换 TTS 底座、训练 Qwen/Dots、全参数 SFT、重做 Phase5A/5A2、自动切 P2、添加 EQ/降噪掩盖问题、进行大规模参数搜索、把未经用户试听的分数称为“通过”。

网络规则保持：**本地缓存优先 → 镜像 → 失败才经 127.0.0.1:7897 代理**。
