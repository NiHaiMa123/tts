# Phase 5A — 锁暝 VoxCPM2 稳定性 / Codec 归因 / Prompt A/B

- 实验 ID：`suoming_voxcpm_phase5a`
- 产出目录：`outputs/gates/suoming_voxcpm_phase5a/`（WAV 不提交 Git，仅本地）
- 后端：`voxcpm2` · `openbmb/VoxCPM2` rev `32279effe8c19989596f05d353d1447f51d9e915` · voxcpm 2.0.3 · torch cu128 · bf16 模型
- 冻结生成参数：`cfg_value=2.0, inference_timesteps=10, normalize=True, denoise=False, retry_badcase=True`
- Git commit（生成时）：`4cd82a0`
- 预算：zero-shot 新生成 **16/18**（P0 两个 A/B 格复用稳定性 WAV，sha 校验一致）

> 评分方向：1–5 分**越高越好**；`grit` 分数高表示磨砂更少。

---

## 1. Codec 1/1 归因

**分类：`codec_reconstruction_limit_supported`**（codec 重建能力上限，非接入实现 bug）

证据（两条独立干净原声一致：validation 雨声锚点 + test 独立原声）：

| 链路 | 含义 | 波形相关 | NMSE | 带内谱差 |
|---|---|---|---:|---:|---|
| 原声48k → 编码输入16k | 纯降采样路径自检 | **0.9997–0.9998** | 0.0003–0.0005 | 0–4k rmse 1.7dB / corr 0.993 |
| 编码输入16k → roundtrip48k | **VAE 自身在可表示带内的失真** | 0.930–0.956 | 0.085–0.135 | 0–4k rmse ~3.9dB，4–7.5k rmse ~4.4dB |
| 原声 → roundtrip | 端到端总损 | 0.929–0.956 | 0.087–0.137 | 同上（几乎全部由 codec 贡献） |
| 同 latent `sr_cond`16k vs 48k | 高频再合成贡献的隔离 | 0.991–0.995 | 0.01–0.018 | 带内几乎一致 |

- **输入链路无 bug**：worker 的 48k→16k 重采样 + encode 输入与参考重采样波形相关 0.9997+，无延时（0ms）、无增益异常、dtype fp32、设备 cuda、encode 走后验均值 `mu`（确定性）。spectrogram 可见 input16k 带内谐波纹络与原声一样锐利。
- **失真全部来自 VAE 本体**：roundtrip 相对编码输入在 0–8k 带内已有 ~4dB 谱差，spectrogram 上辅音/高频谐波明显模糊涂抹 —— 与「磨砂/含糊」的 1/1 主观印象一致。
- **>8kHz 是解码器再合成而非重建**：编码输入为 16kHz（奈奎斯特 8k），roundtrip 的 8–16kHz 能量占比 0.051（原声 0.087）为解码器合成；`sr_cond` 16k/48k 两种解码在带内几乎相同（wcorr 0.99），说明 48k 解码并非"出错"，而是按设计生成上采音频段。
- **zero-shot 与 roundtrip 共用同一解码器**（`audio_vae.decode`，dict→`["audio"]`），但输入分布不同：zero-shot 输入 CFM 预测的 latent，roundtrip 输入真实录音的 encoder-μ latent。因此 **roundtrip 低分不能外推到生成音质**，反之亦然——本轮证据支持两者分离解释。
- **反证已查**：无非对称 API 误用、无张量形状错误、无后处理掩盖；若仍有怀疑可再听 codec 链三文件（input16k/roundtrip48/cond16k）人工定位感知差异。

含义：Codec roundtrip 的 1/1 主要反映 AudioVAE V2 在此类游戏录音上的重建上限（16k 编码带宽 + 带内涂抹 + 合成 HF），不是平台接入错误。**该风险作为"需用户接受的例外"保留**——若训练目标依赖 codec 闭环质量，需用户审听 `codec/` 三链后决定。

## 2. 零样本稳定性（12 cases，全部机械通过）

| case | dur(s) | wall(s) | retry | 峰值/RMS |
|---|---:|---:|---:|---|
| rain_seed42/43/44 | 16.96 / 17.76 / 16.64 | 21.4 / 16.3 / 15.6 | 0 | 无削波 |
| short_response_seed42/43/44 | 3.20 / 4.16 / 3.04 | 3.0 / 3.9 / 2.8 | 0 | 无削波 |
| mid_exposition_seed42/43/44 | 18.08 / 19.68 / 20.64 | 16.6 / 18.1 / 18.9 | 0 | 无削波 |
| pause_emotion_seed42/43/44 | 16.48 / 16.96 / 17.76 | 15.9 / 16.6 / 17.1 | 0 | 无削波 |

- 18/18 case 机械校验通过（可读、无 NaN/Inf、非静音、clip_ratio≈0）。
- `retry_badcase=True` 生效且**全部 0 次实际重试**（worker 侧 stderr 捕获 "Badcase detected" 计数，非推断）。
- 同文不同 seed 时长漂移：rain ±4.6%，short ±37%（短句韵律弹性大），mid ±13%，pause ±7%——时长漂移本身不代表不稳，需用户听判音色漂移/吞字。
- **文本完整性、漏字/吞字、音色一致性：全部 `PENDING_USER_LISTENING`**，平台不伪造主观结论。

## 3. Prompt A/B（6 格，seed=42 固定，唯一变量=Prompt）

| prompt | 来源 | 时长 | 特点 | rain | mid_exposition |
|---|---|---:|---|---|---|
| P0 基线 | inbox 谛天鉴（现用） | 25.2s | 现生产配置 | 复用 stability ✓sha | 复用 stability ✓sha |
| P1 候选 | test#8「恶瘴携雨」 | 10.1s | 干净、沉吟语气 | 21.12s | 22.88s |
| P2 候选 | validation#0「回到华亭」 | 28.1s | 长叙事、多停顿 | 19.20s | 21.76s |

- 三者文本均经 manifest 逐字核对、sha256 校验、split 标注；均未与 ground truth（validation#11 午后凉风）重叠。
- P1 生成时长偏长（+18~27%），是否更接近原声节奏或拖沓需人耳判断；**未擅自宣布赢家，P1/P2 仅标记 `candidate`**，生产默认仍是 P0。

## 4. 可复现性

- `manifest.json`：每 case 含 text_id/text/prompt_id/prompt_sha256/seed/effective_args/model revision/输出 sha256/耗时/retry 计数/sanity。
- `listen/index.html`：分组盲听页（原声 / 基线 / 稳定性 / Prompt A/B / Codec 诊断链），标签匿名 `S-xx/C-xx`，「解盲」按钮 + `unblind_map.json`；导出评分绑定 `experiment_id + case_id + output_sha256`，WAV 重生成后旧评分不会误套。
- `docs/reports/suoming-gate-v1-user-ratings.json`：v1 原始评分已脱敏入库（`raw_export`）。
- 测试：`pytest` **59 passed**（新增 7 项：sanity/对齐/谱差/PNG/manifest 复用/sha 拦截/盲听页）。

## 5. 训练建议

**`MORE_EVIDENCE_NEEDED`** — 理由：

1. Codec 低分已归因为模型重建上限（非实现 bug），且与 zero-shot 解码路径分离，不构成对生成质量的否定；但作为"codec 依赖型训练风险"仍需用户审听接受。
2. 12 个稳定性样本机械全通过、零重试，但**音色漂移/吞字/自然度必须用户盲听**，平台无权自判。
3. Prompt A/B 已备好可比较证据，同样待用户评分。

→ `PENDING_USER_LISTENING` → 之后 `PENDING_USER_DECISION`。本轮未启动、也不应启动任何 LoRA/SFT。

## 6. BLOCKED / 未完成

- 无 BLOCKED。
- 未完成：全部 24 个试听样本的用户评分（含 text_complete/artifact_type 维度）；codec 三链人耳复核；训练决定。

## 7. 建议下一步

1. 打开 `outputs/gates/suoming_voxcpm_phase5a/listen/index.html`，先听 originals 校准，再按组盲听；导出 `listen-ratings-phase5a.json`。
2. 重点对比：`S-xx`（同文三 seed 音色漂移）、`B-xx vs P-xx`（P0 基线 vs P1/P2）、`C-xx`（input16k vs roundtrip48 vs cond16k，定位磨砂感来自带宽还是 VAE）。
3. 评分回填后可复跑 `scripts/export_v1_ratings.py` 同类脱敏导出并入报告。
