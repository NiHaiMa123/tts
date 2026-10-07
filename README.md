# character-tts

与底座无关的角色 TTS 平台。DotsTTS 作为 legacy baseline，VoxCPM2 与
Qwen3-TTS 作为一等后端。主平台不复制任何模型实现——每个 backend 在自己
的 Python 环境里以 **JSONL stdin/stdout worker** 方式接入。

## 架构

```text
主平台 (src/character_tts)
  ├─ registry      角色/后端/评测 YAML 配置 + ${ENV} 展开
  ├─ backends      JSONL protocol + WorkerClient + BackendManager
  ├─ evaluation    gate 执行、sidecar provenance、report.md
  ├─ diagnostics   客观音频指标 (RMS/LUFS/TP/频段能量/flatness/...)
  ├─ audio         wav io、sha256、输出路径安全
  └─ web           试听页生成 + WebUI (FastAPI)

workers/           各后端薄适配（运行在 backend 自己的 env 里）
  ├─ dots_worker.py        legacy dotstts venv
  ├─ voxcpm_worker.py      backend_envs/voxcpm2
  └─ qwen3_tts_worker.py   backend_envs/qwen3_tts
```

约束（见 PLAN.md）：

- 一次只驻留一个大模型（RTX 5080 16GB）；切换后端 = shutdown → 等退出
  → 确认 → 启动 → health check
- 先 codec roundtrip，再 zero-shot，人工试听通过才允许训练
- 不允许任何单一客观指标宣布质量胜负；主观评分保留人工
- 下载顺序：本地 cache → 已有镜像配置 → hf-mirror.com → 127.0.0.1:7897
  代理；每次下载写 logs/downloads.jsonl

## 环境准备

```powershell
# 主平台 env（已有 .venv 则跳过）
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe -e ".[metrics,web,dev]"

# .env.local（gitignore；本机路径，不进 git）
#   DOTSTTS_ROOT=E:/project/dotstts
#   TTS_MODEL_ROOT=E:/project/tts/hf_cache
#   HF_ENDPOINT=...（可选，用户已有镜像）

# 后端 env（各跑各的，互不影响）
powershell -File scripts/bootstrap/setup_dots_legacy.ps1    # 只做校验
powershell -File scripts/bootstrap/setup_voxcpm.ps1         # venv+torch cu128+voxcpm+模型
powershell -File scripts/bootstrap/setup_qwen3_tts.ps1      # venv+torch cu128+qwen-tts+模型
```

## 导入 legacy 角色资产

```powershell
./.venv/Scripts/python.exe scripts/import_legacy_assets.py
# -> configs/characters/suoming.yaml（校验 reference/adapter sha256）
```

## 跑 gate

```powershell
# 全部后端
./.venv/Scripts/python.exe scripts/run_gate.py

# 只跑某个后端
./.venv/Scripts/python.exe scripts/run_gate.py --backend dots_legacy
```

产物：

```text
outputs/gates/suoming_v1/
  reference_original.wav          # 原声参考（sha256 校验）
  <backend>/*.wav + *.wav.json    # 每样本 sidecar provenance
  metrics.json  report.md  report.json
  listen/index.html               # 试听评分页（localStorage，可导出）
```

## WebUI v1

```powershell
./.venv/Scripts/python.exe scripts/start_webui.py
# -> http://127.0.0.1:7860
```

选角色/后端 → 自动切换 worker → 输入文本 → 生成 → 播放 → 打开输出目录。

## 测试

```powershell
./.venv/Scripts/python.exe -m pytest
# 45 tests：协议、配置、worker 生命周期(假worker)、指标、
# gate 端到端、legacy import 校验、下载策略选择（mock）
```

## 目录约定

- `configs/` — app/characters/backends/evaluations YAML
- `backend_envs/` — 各后端独立 venv（gitignore）
- `hf_cache/` — 模型下载缓存（gitignore，可用 TTS_MODEL_ROOT 改）
- `outputs/` — 生成物与 gate 结果（gitignore，除 .gitkeep）
- `logs/` — worker stderr 与 downloads.jsonl（gitignore）
