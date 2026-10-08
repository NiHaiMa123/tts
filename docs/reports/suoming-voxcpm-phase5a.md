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

**用户复核**：codec 链 4 个样本（两原声 × cond48/cond16）主观评分全部 1/1 clarity+grit —— 人耳确认了这是 codec 重建上限，不是偶发样本问题。

- **输入链路无 bug**：worker 的 48k→16k 重采样 + encode 输入与参考重采样波形相关 0.9997+，无延时（0ms）、无增益异常、dtype fp32、设备 cuda、encode 走后验均值 `mu`（确定性）。spectrogram 可见 input16k 带内谐波纹络与原声一样锐利。
- **失真全部来自 VAE 本体**：roundtrip 相对编码输入在 0–8k 带内已有 ~4dB 谱差，spectrogram 上辅音/高频谐波明显模糊涂抹 —— 与「磨砂/含糊」的 1/1 主观印象一致。
- **>8kHz 是解码器再合成而非重建**：编码输入为 16kHz（奈奎斯特 8k），roundtrip 的 8–16kHz 能量占比 0.051（原声 0.087）为解码器合成；`sr_cond` 16k/48k 两种解码在带内几乎相同（wcorr 0.99），说明 48k 解码并非"出错"，而是按设计生成上采音频段。
- **zero-shot 与 roundtrip 共用同一解码器**（`audio_vae.decode`，dict→`["audio"]`），但输入分布不同：zero-shot 输入 CFM 预测的 latent，roundtrip 输入真实录音的 encoder-μ latent。因此 **roundtrip 低分不能外推到生成音质**，反之亦然——本轮证据支持两者分离解释。
- **反证已查**：无非对称 API 误用、无张量形状错误、无后处理掩盖；若仍有怀疑可再听 codec 链三文件（input16k/roundtrip48/cond16k）人工定位感知差异。

含义：Codec roundtrip 的 1/1 主要反映 AudioVAE V2 在此类游戏录音上的重建上限（16k 编码带宽 + 带内涂抹 + 合成 HF），不是平台接入错误。**该风险作为"需用户接受的例外"保留**——若训练目标依赖 codec 闭环质量，需用户审听 `codec/` 三链后决定。

## 2. 零样本稳定性（12 cases，全部机械通过）

| case | dur(s) | wall(s) | retry | 用户评分 c/g/l/n | 备注 |
|---|---:|---:|---:|---|---|
| rain_seed42 | 16.96 | 21.4 | 0 | 4/5/4/5 | |
| rain_seed43 | 17.76 | 16.3 | 0 | 4/5/4/5 | |
| rain_seed44 | 16.64 | 15.6 | 0 | 4/5/4/5 | |
| short_response_seed42 | 3.20 | 3.0 | 0 | **1/1/3/5** | ⚠ 坏句 |
| short_response_seed43 | 4.16 | 3.9 | 0 | 5/5/3/3 | |
| short_response_seed44 | 3.04 | 2.8 | 0 | 4/4/2/1 | ⚠ 像角色/自然度低 |
| mid_exposition_seed42 | 18.08 | 16.6 | 0 | 5/5/5/5 | |
| mid_exposition_seed43 | 19.68 | 18.1 | 0 | 5/5/5/5 | |
| mid_exposition_seed44 | 20.64 | 18.9 | 0 | 5/5/4/**1** | ⚠ 自然度 1 |
| pause_emotion_seed42 | 16.48 | 15.9 | 0 | 5/5/5/4 | |
| pause_emotion_seed43 | 16.96 | 16.6 | 0 | 4/5/5/4 | 用户注：偏快、回声偏大/口型大 |
| pause_emotion_seed44 | 17.76 | 17.1 | 0 | 5/5/4/4 | 用户注：回声偏大/口型大 |

- 18/18 case 机械校验通过（可读、无 NaN/Inf、非静音、clip_ratio≈0）。
- `retry_badcase=True` 生效且**全部 0 次实际重试**（worker 侧 stderr 捕获 "Badcase detected" 计数，非推断）——`short_response_seed42` 的 1/1 坏句**未被 badcase 检测捕获**，说明该启发式对这类失效不敏感。
- **稳定性结论（用户已听）**：12 条中 4 条存在 ≤3 分维度（≈33% 标记异常），集中在**短句类文本**（short_response 三个 seed 均不理想：1/1、3/3、2/1）和 mid_seed44 自然度 1；rain 与 pause_emotion 稳定（4-5 分）。这不是"全面不稳"，但短句类存在真实失稳，需列为后续专项问题。
- 同文不同 seed 时长漂移：rain ±4.6%，short ±37%，mid ±13%，pause ±7%——短句时长弹性与主观不稳相符。

## 3. Prompt A/B（6 格，seed=42 固定，唯一变量=Prompt）

| prompt | 来源 | 时长 | 特点 | rain c/g/l/n | mid c/g/l/n | 用户备注 |
|---|---|---:|---|---|---|---|
| P0 基线 | inbox 谛天鉴（现用） | 25.2s | 现生产配置 | 5/5/4/3 | 5/5/5/4 | 速度偏快 ×2 |
| P1 候选 | test#8「恶瘴携雨」 | 10.1s | 干净、沉吟语气 | 3/5/4/2 | 4/5/5/2 | 偏慢；回声/口型大 |
| **P2 候选** | validation#0「回到华亭」 | 28.1s | 长叙事、多停顿 | **5/5/5/5** | **5/5/5/5** | — |

- 三者文本均经 manifest 逐字核对、sha256 校验、split 标注；均未与 ground truth（validation#11 午后凉风）重叠。
- **用户已听**：P2 两格全 5 分优于 P0（P0 两格均被注"速度偏快"，rain 自然度 3）与 P1（自然度 2、回声/口型大）。P2 是当前最强候选，**但平台不擅自切换生产默认**——P2 仅标记 `candidate`，是否替换 P0 待用户明确确认；另注意 P2 来自 validation split，若未来该 split 需严格 held-out 评测，应再选独立音频复核。

## 4. 可复现性

- `manifest.json`：每 case 含 text_id/text/prompt_id/prompt_sha256/seed/effective_args/model revision/输出 sha256/耗时/retry 计数/sanity。
- `listen/index.html`：分组盲听页（原声 / 基线 / 稳定性 / Prompt A/B / Codec 诊断链），标签匿名 `S-xx/C-xx`，「解盲」按钮 + `unblind_map.json`；导出评分绑定 `experiment_id + case_id + output_sha256`，WAV 重生成后旧评分不会误套。
- `docs/reports/suoming-gate-v1-user-ratings.json`：v1 原始评分已脱敏入库（`raw_export`）。
- `docs/reports/suoming-voxcpm-phase5a-ratings.json`：本轮 22 条用户评分脱敏入库，**22/22 条 `output_sha256` 与 manifest 校验一致**（`scripts/export_phase5a_ratings.py` 可复跑）。
- 测试：`pytest` **59 passed**（新增 7 项：sanity/对齐/谱差/PNG/manifest 复用/sha 拦截/盲听页）。

## 5. 训练建议

**`MORE_EVIDENCE_NEEDED`** — 理由（用户已盲听 22/24 样本后更新）：

1. **Codec 异常已关闭**：双原声 × 双解码变体 1/1 人耳确认 = `codec_reconstruction_limit_supported`（VAE 重建上限，非接入 bug）；但该项作为"codec 依赖型训练风险"仍需用户显式接受。
2. **稳定性存在真实异常**：12 条中 4 条有 ≤3 分维度——短句类三 seed 均不理想（含一条 1/1 坏句，`retry_badcase` 未捕获），mid_seed44 自然度 1。rain/pause_emotion 稳定。短句失稳需在下一轮定位（文本类问题还是 seed 运气），不应直接判"达标"或"否决"。
3. **Prompt 已有人证**：P2 两格全 5 分为当前最优候选，P0"速度偏快"、P1 更弱；是否切换生产默认待用户确认。
4. 未达 `DO_NOT_TRAIN_YET`（非大面积不稳、无系统性文本错误），也未达 `RECOMMEND`（短句异常未解释、codec 风险未接受）。

→ `PENDING_USER_DECISION`（用户需决定：是否接受 codec 风险、是否认可 P2、是否就短句异常追加 bounded 复测）。本轮未启动、也不应启动任何 LoRA/SFT。

## 6. BLOCKED / 未完成

- 无 BLOCKED。
- 未完成：2 个原声样本评分（校准用，可选）；训练决定；P0/P2 默认切换确认；短句类失稳的根因复查（如需）。

## 7. 建议下一步

1. 用户复核异常案例：`stability/short_response_seed42`（1/1 坏句）与其余两 seed 对比，确认是文本类系统性问题还是单 seed 运气。
2. 用户决定是否将 P2 设为生产 Prompt（平台不会自动切换；若担心 validation split 洁净度，可另选一条独立干净音频复核）。
3. 若倾向推进训练：先就 codec 风险形成书面接受，再开 Phase 5B 训练方案设计；如需更多稳定性证据，可对短句类文本做第二轮 bounded seed 测试（需新批准预算）。
