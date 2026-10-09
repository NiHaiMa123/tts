# character-tts

与底座无关的角色 TTS 平台，完全自包含（不依赖旧 dotstts 项目）。
VoxCPM2 与 Qwen3-TTS 为一等后端。主平台不复制模型实现——每个 backend
在自己的 Python 环境里以 **JSONL stdin/stdout worker** 方式接入。

**当前生产路径**：锁暝 = VoxCPM2 零样本（LoRA pilot 已关闭，见
`docs/reports/`；Dots legacy 后端已随依赖切断移除，需要时从旧项目
重新迁移）。

## 日常使用（WebUI）

双击 **`启动TTS.vbs`** → 弹出控制台（= 服务器）→ 自动打开 Edge
`http://127.0.0.1:7860`。

- 把台本存为 UTF-8 txt 放进 **`inputs/`** → 页面选**角色**（后端已锁定）
  → 「开始 TTS」→ 每个 txt 生成一个 WAV 到 `outputs/webui/<角色>/`
- **关掉控制台窗口 = 停止服务器**；关页面不影响任务

## 目录结构

```text
configs/
  app.yaml                    # 默认角色/后端、路径根
  characters/<id>.yaml        # 角色登记：backend 锁定、参考音、数据集、
                              #   adapters/postprocess（预留接口）、评测锚点
  backends/<id>.yaml          # 后端登记：worker 入口、env、模型、enabled
  evaluations/                # gate/实验配置（历史记录）
  training/                   # 训练配置（历史记录）
assets/characters/<id>/       # 角色轻资产（进 git 的只有 stub/文档）
  reference/ref.wav           # zero-shot 克隆参考音（*.wav gitignored）
  adapters/<backend>/<name>/  # 预留：转正后的 LoRA 放这（gitignored）
  postprocess.yaml            # 预留：后处理链配置（未实现）
data/characters/<id>/         # 角色重资产（gitignored）
  inbox/                      # 原始源音频（锁暝 270 条 / 142MB）
  datasets/v1/                # train/val/test jsonl + audio/ 自包含音频
src/character_tts/
  registry/                   # YAML 加载、${ENV}/${TTS_ROOT} 展开、模型
  backends/                   # JSONL 协议、WorkerClient、BackendManager
  evaluation/                 # gate 执行、sidecar provenance、report.md
  diagnostics/                # 客观音频指标
  audio/                      # wav io、sha256、输出路径安全
  web/                        # 试听页生成 + WebUI (FastAPI)
workers/                      # 各后端薄适配（跑在 backend 自己的 env 里）
backend_envs/                 # 各后端独立 venv（gitignore）
inputs/                       # WebUI 批量台本 txt（gitignored）
outputs/                      # 生成物与实验结果（gitignored）
  gates/<exp>/                #   评测产物 + listen/ 试听评分页
  webui/<角色>/               #   WebUI 产出 WAV
  user_gen/                   #   手工长文生成
logs/                         # worker stderr、downloads.jsonl
```

## 关键约定

- 一次只驻留一个大模型（RTX 5080 16GB）；生成进行中拒绝切换
- 角色选后端是**绑定关系**（`backend:` 字段），不是用户可选
- clone prompt（`reference`）与评测原声（anchor `ground_truth`）严格分离
- 人工试听权威；客观指标/ASR 只是辅助证据，不宣判胜负
- 先 codec roundtrip，再 zero-shot，人耳通过才允许训练
- 下载顺序：本地 cache → hf-mirror.com → 127.0.0.1:7897 代理；
  revision 固定，写 logs/downloads.jsonl

## 环境

```powershell
# 主平台
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe -e ".[metrics,web,dev]"

# .env.local（gitignore）
#   TTS_MODEL_ROOT=E:/project/tts/hf_cache

# 后端 env（按需）
powershell -File scripts/bootstrap/setup_voxcpm.ps1     # 锁暝生产环境
powershell -File scripts/bootstrap/setup_qwen3_tts.ps1

# ASR 辅助检查 env（三套，仅辅助证据用；均 py3.12）
uv venv backend_envs/asr_sensevoice --python 3.12
uv pip install --python backend_envs/asr_sensevoice/Scripts/python.exe \
  "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python backend_envs/asr_sensevoice/Scripts/python.exe \
  "funasr==1.4.4" soundfile

uv venv backend_envs/asr_faster_whisper --python 3.12
uv pip install --python backend_envs/asr_faster_whisper/Scripts/python.exe \
  "faster-whisper==1.2.1" "ctranslate2==4.8.1" "av==18.1.0" \
  "onnxruntime==1.29.0" "tokenizers==0.23.1" \
  nvidia-cublas-cu12 nvidia-cudnn-cu12

uv venv backend_envs/asr_qwen3 --python 3.12
uv pip install --python backend_envs/asr_qwen3/Scripts/python.exe \
  "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" "torchvision==0.23.0+cu128" \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python backend_envs/asr_qwen3/Scripts/python.exe \
  "qwen-asr==0.0.6" "transformers==4.57.6" "accelerate==1.12.0" \
  "librosa==1.0.0" "soundfile==0.14.0" "qwen-omni-utils==0.0.9"
```

## 加新角色

1. 源音频放 `data/characters/<id>/inbox/`，数据集放 `datasets/v1/`
2. 参考音放 `assets/characters/<id>/reference/ref.wav`，算 sha256
3. 建 `configs/characters/<id>.yaml`：`backend:` 锁后端 +
   `reference` + `dataset` + `evaluation.anchor_texts`
4. WebUI 下拉自动出现，无需改代码

## ASR 辅助检查（多后端）

```powershell
# manifest 批量（backend ∈ sensevoice | faster_whisper | qwen3_asr）
backend_envs\asr_<backend>\Scripts\python.exe scripts\asr_check.py ^
  --backend <backend> --manifest outputs/gates/<exp>/manifest.json ^
  --out <exp>/asr_check.json

# 任意 wav + 参考文本
backend_envs\asr_<backend>\Scripts\python.exe scripts\asr_check.py ^
  --backend <backend> --wav xxx.wav --text-file xxx.txt --out out.json
```

模型在 `models/asr/<backend>/<rev>/`（gitignored，共 ~4.3GB）。
输出 char-CER，仅辅助证据——`text_complete` 仍是人耳判断字段。

## 测试

```powershell
./.venv/Scripts/python.exe -m pytest    # 69 tests
```
