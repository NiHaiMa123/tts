# 锁暝 backend gate v1 — 平台验收报告（v2 修订）

- 评测 ID: `suoming_v1`
- 角色: `suoming`（锁暝）
- 生成时间: 2026-10-08（UTC+8）
- 种子: 42
- 克隆参考（prompt）sha256: `c4322e3230ab952edb7cd456329cb5b48195d9f33c5459c0ba76d8d79d3a7f92`
- 评测原声（ground truth）sha256: `0745cd41cc9d07e544ab83b9c9db43ac5d355fe2d4765ec970494fd0275693ab`
  - `data/work/standardized/fb/fb8e3c08…aac1ca.wav`，48 kHz / 21.18 s，
    文本=anchor「午后凉风拂过……」，来自 validation.jsonl（与 dotstts
    磨砂诊断 REF_WAV 同一条）
- 机器: RTX 5080 16GB / driver 591.86 / Windows
- 数据权威来源（生成物，不进 git）:
  `outputs/gates/suoming_v1/report.json`、`metrics.json`、`report.md`

## 结论

三个后端全部通过 codec roundtrip 与 zero-shot 机检门（7/7 cases ok）。
用户已完成盲听（`listen/listen-ratings.json`，1–5 越低越差，如实回填
见下表）。**"是否进入训练"保持 `PENDING_USER_DECISION`——听完了，
但是否训练由用户决定，平台不自动宣布胜负。**

| Backend | Env | Codec roundtrip | Zero-shot | 主观评分(用户 c/g/l/n) | 建议训练 |
|---|---|---|---|---|---|
| dots_legacy (LoRA step500 基线) | yes | ok (41.6s, 确定性) | ok (20.3s) | codec 3/3/4/5 · zs 5/5/1/2 · lora 5/5/2/2 | PENDING_USER_DECISION |
| voxcpm2 | yes | ok (0.1s) | ok (27.7s) | codec 1/1/4/5 · zs 5/5/5/3 | PENDING_USER_DECISION |
| qwen3_tts | yes | ok (4.5s) | ok (23.6s) | codec 1/1/4/5 · zs 1/4/3/3 | PENDING_USER_DECISION |

dots_legacy 另完成 `generate`（已训练 LoRA 路径）ok (5.4s)。

### 用户盲听全量评分（来自 listen/listen-ratings.json）

| Sample | 清澈度 | 磨砂/颗粒 | 像角色 | 自然度 |
|---|---|---|---|---|
| ground_truth_original（原声） | 3 | 3 | 4 | 5 |
| reference_original（克隆参考） | 5 | 5 | 5 | 5 |
| dots codec roundtrip | 3 | 3 | 4 | 5 |
| dots LoRA generate | 5 | 5 | 2 | 2 |
| dots zero-shot | 5 | 5 | 1 | 2 |
| qwen3_tts codec roundtrip | 1 | 1 | 4 | 5 |
| qwen3_tts zero-shot | 1 | 4 | 3 | 3 |
| voxcpm2 codec roundtrip | 1 | 1 | 4 | 5 |
| voxcpm2 zero-shot | 5 | 5 | 5 | 3 |

维度含义：c=清澈度 g=磨砂/颗粒 l=像角色 n=自然度。用户备注：
「得分越低越差」。

## v2 修复内容（Phase 0–4 审查）

1. **clone prompt 与评测 ground truth 分离**：importer 原取
   validation.jsonl 第一条（「我始终想着…」），已改为按 anchor 文本精确
   匹配（「午后凉风拂过…」，validation 第 12 条），写入
   `anchor_texts[].audio/sha256/source` 并在 gate 中校验 sha256。codec
   roundtrip 重建 ground truth；无 ground truth 的角色 → codec case
   `blocked`（不再静默用 prompt）。zero-shot 仍用 25s 谛天鉴 prompt。
2. **Dots codec roundtrip 确定性**：`vae.inference()` 内部
   `do_sample=True` 会从后验采样，已改为 `extract_latents` → 取后验均值
   m_q → `inference_from_latents(do_sample=False)`，可复现、可与
   dotstts 磨砂诊断（同一 REF_WAV）比较。
3. **WebUI 并发防护**：生成进行中 `switch`/`stop`/`ensure`（跨线程）
   一律抛 `BackendBusyError`（API 409）；任务先 acquire generate gate
   再 ensure（内部切换在 gate 内完成）；shutdown 等 30s 后 force_stop。
   新增慢速 fake worker 并发测试。
4. **Bootstrap**：改用 `huggingface_hub.snapshot_download`（revision 固定
   为后端配置值），镜像失败回退 127.0.0.1:7897 时显式清除 HF_ENDPOINT，
   downloads.jsonl 记录真实 resolved revision 与快照路径。
5. **跨采样率频谱指标**：所有频谱指标在公共 analysis_sr=24 kHz 上计算，
   对比表只含共享带（≤12 kHz）；`native_energy_above_12k` 作为 per-file
   描述项留在 metrics.json，不进对比表。

## 环境与下载溯源

下载顺序：本地 cache → hf-mirror.com 镜像 → `127.0.0.1:7897` 代理。
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
| dots_legacy | `E:\project\dotstts\.venv` (py3.12, 复用) | dots.tts 0.3.1 | 10.5 GiB |
| voxcpm2 | `backend_envs/voxcpm2` (py3.11) | voxcpm 2.0.3 + torch 2.11.0+cu128 | 6.3 GiB |
| qwen3_tts | `backend_envs/qwen3_tts` (py3.11) | qwen-tts + torch 2.11.0+cu128 (attn=sdpa) | 4.6 GiB |

## 客观指标（描述性，非评判；公共带 ≤12 kHz）

节选 `metrics.json`（完整字段含 RMS/LUFS/true-peak/频段能量/flatness/
crest/频谱熵/2–9kHz 动态/analysis_sr/native>12k 描述项）：

| Sample | SR | dur(s) | LUFS | TP dBTP | E4-8k% | E8-12k% | flatness | Δ2-9k |
|---|---|---|---|---|---|---|---|---|
| ground_truth_original（原声） | 48000 | 21.18 | -21.5 | -8.2 | 0.037 | 0.066 | 0.401 | 0.279 |
| dots codec roundtrip | 48000 | 21.20 | -21.9 | -8.3 | 0.031 | 0.054 | 0.407 | 0.268 |
| voxcpm2 codec roundtrip | 48000 | 21.20 | -21.6 | -6.1 | 0.053 | 0.048 | 0.404 | 0.283 |
| qwen3_tts codec roundtrip | 24000 | 21.20 | -22.5 | -8.6 | 0.030 | 0.026 | 0.332 | 0.280 |
| dots zero-shot | 48000 | 13.92 | -21.9 | -7.3 | 0.040 | 0.092 | 0.335 | 0.406 |
| voxcpm2 zero-shot | 48000 | 16.96 | -22.4 | -7.6 | 0.027 | 0.017 | 0.363 | 0.333 |
| qwen3_tts zero-shot | 24000 | 12.80 | -22.5 | -10.2 | 0.027 | 0.018 | 0.322 | 0.409 |

## 试听入口

`outputs/gates/suoming_v1/listen/index.html`（本地打开即可，评分存
localStorage，可导出 JSON）。排序：原声(validation) → 克隆参考 →
各后端 codec/zero-shot/LoRA 样本。
另有 WebUI：`scripts/start_webui.py` → http://127.0.0.1:7860 。

## BLOCKED

无 gate 阻塞项。

## 下一步

1. **用户决策**：基于上述盲听评分决定是否对某后端启动 LoRA/SFT
   （PLAN.md 顺序约束——codec+zero-shot 已过机检，人耳结果已回填）。
2. 可选：补 SoX；扩展第二条 anchor text 做交叉验证；
   用户若改变结论可在试听页改分后重新导出，重跑 finalize 即回填。
