# 环境、辅助模型与下载

本文件集中原 README 的环境和 ASR 命令。角色入库见 [语音资产管线](voice-assets.md)，
当前任务见 [PLAN](../../PLAN.md)。以下 Windows 命令在仓库根目录运行。

## 环境隔离

| 环境 | 用途 |
|---|---|
| `.venv` | 主平台、WebUI、页面服务、入库及不依赖真实模型的测试 |
| `backend_envs/voxcpm2` | Vox 生产推理；bootstrap 安装 `triton-windows`，缺失时可能退回 eager |
| `backend_envs/qwen3_tts` | Qwen 零样本推理；旧微调脚本另存本地，不代表标准 worker 支持 LoRA |
| `backend_envs/voxcpm2_train` | 独立 Vox 训练环境，步骤见 [LoRA 管线](lora-training.md) |
| `backend_envs/asr_sensevoice` | SenseVoice、CAM++、FSMN-VAD 等辅助检查 |
| `backend_envs/asr_faster_whisper` | Whisper 独立复核 |
| `backend_envs/asr_qwen3` | Qwen3-ASR 独立复核 |

主平台和 TTS 后端安装入口见 [README](../../README.md#安装与检查)。
`.env.local` 可设置 `TTS_MODEL_ROOT=E:/project/tts/hf_cache`。
已有正常环境先取版本和源码 hash，实验依赖单独安装；不要用新包覆盖生产环境来试错。
固定模型 revision 不等于固定 Python 包、训练脚本或损失实现。

## 辅助环境安装

下面保留项目既有辅助环境配方，需按实际安装结果留出依赖版本；不表示本次在 Windows 重新安装验证过。

```powershell
uv venv backend_envs/asr_sensevoice --python 3.12
uv pip install --python backend_envs/asr_sensevoice/Scripts/python.exe `
  "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" `
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python backend_envs/asr_sensevoice/Scripts/python.exe `
  "funasr==1.4.4" soundfile

uv venv backend_envs/asr_faster_whisper --python 3.12
uv pip install --python backend_envs/asr_faster_whisper/Scripts/python.exe `
  "faster-whisper==1.2.1" "ctranslate2==4.8.1" "av==18.1.0" `
  "onnxruntime==1.29.0" "tokenizers==0.23.1" `
  nvidia-cublas-cu12 nvidia-cudnn-cu12

uv venv backend_envs/asr_qwen3 --python 3.12
uv pip install --python backend_envs/asr_qwen3/Scripts/python.exe `
  "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" "torchvision==0.23.0+cu128" `
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python backend_envs/asr_qwen3/Scripts/python.exe `
  "qwen-asr==0.0.6" "transformers==4.57.6" "accelerate==1.12.0" `
  "librosa==1.0.0" "soundfile==0.14.0" "qwen-omni-utils==0.0.9"
```

## ASR 与声纹调用

ASR 后端名称与环境名不同：

| `--backend` | Python 路径 |
|---|---|
| `sensevoice` | `backend_envs/asr_sensevoice/Scripts/python.exe` |
| `faster_whisper` | `backend_envs/asr_faster_whisper/Scripts/python.exe` |
| `qwen3_asr` | `backend_envs/asr_qwen3/Scripts/python.exe` |

以 SenseVoice 为例：

```powershell
backend_envs\asr_sensevoice\Scripts\python.exe scripts\asr_check.py `
  --backend sensevoice --manifest outputs\gates\<exp>\manifest.json `
  --out outputs\gates\<exp>\asr_check.json
# 单条音频可替换 manifest 为 --wav xxx.wav --text-file xxx.txt
```

ASR 模型位置为 `models/asr/<backend>/<rev>/`。CER 是辅助内容证据，
不能代替音质或用户的 `text_complete` 标签。规范化与第二 ASR 复核要求见 [统一自检](self-check.md)。

CAM++ 和 VAD 的现有调用、输出合并命令见 [语音资产管线](voice-assets.md)。
CAM++ 首次可能下载到 `models/speaker/`；评价生成物时必须使用独立原声参考，不能把生成物加入自中心。

## 下载与运行记录

- 顺序：本地 cache → 当前机器已有镜像 / `hf-mirror.com` → 已有 `127.0.0.1:7897` 代理。先检查现有配置，代理不可用时如实记录失败。
- 固定模型 revision，保存实际来源、路径、权重 SHA 和安装版本；下载记录见 `logs/downloads.jsonl`。bootstrap 相关逻辑位于 `scripts/bootstrap/common.ps1`。
- 生产后端配置要求本地文件与离线加载，先下载再推理；不能把下载失败或缺少权重记录为模型已执行。
- RTX 5080 16GB 上生成后先卸载 TTS，再串行运行评分模型。未来统一评价环境独立建立，见自检规范。
