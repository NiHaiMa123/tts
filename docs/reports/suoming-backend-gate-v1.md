# 锁暝 backend gate v1 — 平台验收报告

- 评测 ID: `suoming_v1`
- 角色: `suoming`（锁暝）
- 生成时间: 2026-10-07（UTC+8）
- 种子: 42
- 参考音频 sha256: `c4322e3230ab952edb7cd456329cb5b48195d9f33c5459c0ba76d8d79d3a7f92`
- 机器: RTX 5080 16GB / driver 591.86 / Windows
- 数据权威来源（生成物，不进 git）:
  `outputs/gates/suoming_v1/report.json`、`metrics.json`、`report.md`

## 结论

三个后端全部通过 codec roundtrip 与 zero-shot 机检门（7/7 cases ok）。
**主观音质与"是否进入训练"均保持 `PENDING_USER_LISTENING`，等待人工试听。**
按 PLAN.md 约束，本报告不以任何客观指标宣布胜负。

| Backend | Env | Codec roundtrip | Zero-shot | 主观评分 | 建议训练 |
|---|---|---|---|---|---|
| dots_legacy (LoRA step500 基线) | yes | ok (2.9s) | ok (25.3s) | PENDING_USER_LISTENING | PENDING_USER_LISTENING |
| voxcpm2 | yes | ok (0.1s) | ok (23.9s) | PENDING_USER_LISTENING | PENDING_USER_LISTENING |
| qwen3_tts | yes | ok (5.1s) | ok (22.5s) | PENDING_USER_LISTENING | PENDING_USER_LISTENING |

dots_legacy 另完成 `generate`（已训练 LoRA 路径）ok (32.3s)。

## 环境与下载溯源

下载顺序：本地 cache → HF_ENDPOINT 镜像 → `127.0.0.1:7897` 代理。
全部记录见 `logs/downloads.jsonl`；未记录任何 token/凭据。

| Asset | 来源 | Revision | 本地路径 |
|---|---|---|---|
| dots.tts-soar base | local（复用 dotstts 仓库） | `2f9b3e18` | `E:\project\dotstts\pretrained_models\dots.tts-soar` |
| suoming LoRA step500 | local | adapter sha256 `d6ffc625` | `E:\project\dotstts\data\work\suoming\suoming_lora_v1\checkpoint-00000500\model` |
| openbmb/VoxCPM2 | mirror | `32279eff` | `hf_cache/voxcpm2`（≈4.7GB） |
| Qwen/Qwen3-TTS-12Hz-1.7B-Base | proxy（镜像不完整后回退） | `fd4b2543` | `hf_cache/qwen3_tts` |
| Qwen/Qwen3-TTS-Tokenizer-12Hz | proxy | `7dd38ad4` | `hf_cache/qwen3_tts` |

后端运行时 `HF_HUB_OFFLINE=1`（voxcpm2 用 `local_files_only`），worker 经
`snapshot_download(local_files_only=True)` 解析本地快照——联网抖动不影响
gate 复现。

| Backend | Python env | 包 | Peak VRAM |
|---|---|---|---|
| dots_legacy | `E:\project\dotstts\.venv` (py3.12, 复用) | dots.tts 0.3.1 | 6.2 GiB |
| voxcpm2 | `backend_envs/voxcpm2` (py3.11) | voxcpm 2.0.3 + torch 2.11.0+cu128 | 6.1 GiB |
| qwen3_tts | `backend_envs/qwen3_tts` (py3.11) | qwen-tts + torch 2.11.0+cu128 (attn=sdpa) | 4.6 GiB |

## 客观指标（描述性，非评判）

节选 `metrics.json`（完整字段含 RMS/LUFS/true-peak/频段能量/flatness/crest/
频谱熵/2–9kHz 动态等）：

| Sample | dur(s) | LUFS | TP dBTP | flatness | Δ2-9k |
|---|---|---|---|---|---|
| reference_original | 25.22 | -21.7 | -8.9 | 0.126 | 0.261 |
| dots codec roundtrip | 25.24 | -22.0 | -8.6 | 0.063 | 0.237 |
| voxcpm2 codec roundtrip | 25.24 | -21.8 | -6.9 | 0.134 | 0.243 |
| qwen3_tts codec roundtrip | 25.28 | -22.4 | -9.5 | 0.327 | 0.368 |
| dots zero-shot | 13.92 | -21.9 | -7.3 | 0.049 | 0.246 |
| voxcpm2 zero-shot | 16.96 | -22.4 | -7.6 | 0.116 | 0.206 |
| qwen3_tts zero-shot | 12.80 | -22.5 | -10.2 | 0.322 | 0.409 |

可读差异（仅供人工试听时对照，不下结论）：qwen3_tts 为 24kHz 输出且
12–18kHz 能量近零（codec 上限），voxcpm2/dots 为 48kHz；codec roundtrip 的
flatness/Δ2-9k 差异提示各 codec 对参考样本的频谱重塑程度不同。

## 试听入口

`outputs/gates/suoming_v1/listen/index.html`（本地打开即可，评分存
localStorage，可导出 JSON）。页面内为盲听布局：A/B/参考 三栏。
另有 WebUI：`scripts/start_webui.py` → http://127.0.0.1:7860 。

## 已解决 / 记录的问题

- qwen3_tts `Qwen3TTSModel.from_pretrained` 会无条件下探 HF API；已改为
  `snapshot_download(local_files_only=True)` 预解析本地快照 + worker env
  `HF_HUB_OFFLINE=1`，离线可复现。
- `flash-attn` 在 sm_120 不可用，qwen3_tts 使用官方 `sdpa` 路径。
- qwen-tts 依赖 SoX 二进制做部分音频转换；当前 codec/zero-shot 路径未
  触发，若后续用到需安装 SoX（记为环境待办，非 gate 阻塞）。

## BLOCKED

无 gate 阻塞项。`PENDING_USER_LISTENING` 是唯一待办。

## 下一步（需人工试听结论后）

1. 试听 `listen/index.html` 并导出评分 → 回填报告主观列。
2. 仅当 codec+zero-shot 人耳通过后，才允许对对应后端启动 LoRA/SFT 方案
   设计（PLAN.md 顺序约束）。
3. 可选：补 SoX；扩展第二条 anchor text 做交叉验证。
