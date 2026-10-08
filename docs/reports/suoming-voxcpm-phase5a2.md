# Phase 5A2 — VoxCPM2 P0/P2 短句稳定性配对对照

日期：2026-10-08 · 实验 ID：`suoming_voxcpm_phase5a2`
状态：**已生成 14 格（4 复用 + 10 新增），主观评分 PENDING_USER_LISTENING**

## 目的

Phase 5A 发现 P0 Prompt 下 `short_response`（开伞，由我来动手。）
三 seed 人耳评分均差（含 1/1 坏句），而 P2 在两格有限对照中全 5 分。
本节按 PLAN.md §24 做**最小有界配对验证**：同文同 seed、唯一变量
是 Prompt，检验 P2 优势是否跨短句与 seed 泛化，以及长句 seed44 的
自然度异常是否随 P2 改善。

**不做的事**：不训练、不改生产默认 Prompt、不重新打开 codec 结论、
不做无界 seed 扫掠与无限重试。

## 冻结基线

| 项 | 值 |
|---|---|
| 后端 | voxcpm2 (`openbmb/VoxCPM2`) |
| 模型 revision | `32279effe8c19989596f05d353d1447f51d9e915`（与复用清单一致，`reuse_revision_match=true`） |
| voxcpm 包 | 2.0.3 |
| 固定生成参数 | cfg_value=2.0, inference_timesteps=10, normalize=true, denoise=false, retry_badcase=true |
| 冻结提交 | 见 manifest `git_commit` |

## Prompt 与文本溯源（全部 sha256 校验通过）

| ID | 角色 | 来源 | sha256 |
|---|---|---|---|
| P0 | baseline | `data/inbox/锁暝/中立_neutral/【中立_neutral】谛天鉴…`（人工审核资产） | `c4322e32…` |
| P2 | **candidate**（仅候选，未提升为默认） | `validation.jsonl#0` fid=27cd4b76528acdb6「回到华亭」28.1s | `fb77c9bb…` |

| text_id | 文本 | split | 原声 sha256 |
|---|---|---|---|
| short_response | 开伞，由我来动手。 | validation.jsonl#3 fid=4256545f053373b0 | `36a16737…` |
| short_response_2 | 有什么在跟着我们 | **test.jsonl#14** fid=c3dbe43cdc84（新独立短句，7 字，语义完整，非 Prompt 音频） | `55656bc9…` |
| mid_exposition | 若论诞生年岁，心月狐还比我小些…… | validation.jsonl#2 fid=33e707acde1f4e22 | `674473ff…` |

第二短句选自 test split 真实录音，满足：6–20 字、语义完整、
与 `short_response` 不同、不作本轮 Prompt 音频。三组文本的原声
录音仅作溯源记录，不参与生成。

## 配对矩阵（7 对 = 14 格）

| pair | text | seed | P0 | P2 |
|---|---|---|---|---|
| A·s42 | short_response | 42 | **复用** `stability/short_response_seed42` | 新增 |
| A·s43 | short_response | 43 | **复用** `stability/short_response_seed43` | 新增 |
| A·s44 | short_response | 44 | **复用** `stability/short_response_seed44` | 新增 |
| B·s42 | short_response_2 | 42 | 新增 | 新增 |
| B·s43 | short_response_2 | 43 | 新增 | 新增 |
| B·s44 | short_response_2 | 44 | 新增 | 新增 |
| C·s44 | mid_exposition | 44 | **复用** `stability/mid_exposition_seed44` | 新增 |

- 复用校验：文件存在 + 磁盘 sha256 == 旧 manifest + prompt_sha256 ==
  P0 + seed/text/effective_args 全部逐项比对；任一失败记
  `BLOCKED_REUSE`，**绝不静默重生成**。本轮 4/4 全部校验通过。
- 预算：`zero_shot_generated = 10/10`（硬上限，超出记 `blocked_budget`）。

## 执行结果

- 14/14 格 `ok`；`retry_count` 全 0（`retry_badcase` 无触发）
- 无削波（clip_ratio 全 0）、无 NaN、时长 2.2–21.8s 合理
- 每格 sidecar/manifest 记录：case_id、prompt_id+sha、seed、text、
  output+sha256、duration、sample_rate、effective_args、retry_count、
  reused_from、sanity

## ASR 辅助证据（SenseVoice rev 3847d57b，仅辅助，不替代人耳）

字符级 CER（去标点/标签）：

| pair | P0 | P2 |
|---|---|---|
| A·s42 | 0.143（开→凯） | **0.000** |
| A·s43 | 0.143（开→凯） | 0.143（开→凯） |
| A·s44 | **0.000** | 0.143（开→凯） |
| B·s42/43/44 | 0.000 ×3 | 0.000 ×3 |
| C·s44 | 0.099 | 0.085 |

读法：

- **pair B** 两 Prompt 全种子字全对——第二短句无文本完整性问题。
- **pair A**「开」→「凯」在 6 格中 4 格复现（P0 两次、P2 两次），
  是同音字级发音偏移，非 Prompt 特异；短句失稳的主矛盾仍在声学
  质量（Phase 5A 人耳 1/1 坏句）而非成句错误。
- **pair C** 两侧同错「心月狐→新月湖」「祂→他」，专有名词/
  同音字层面，P2 略低但不构成判据。
- ASR 只证明「说了什么」，不证明「说得好不好」；`text_complete`
  最终仍由试听页人工判定。

## 同 SHA 上下文评分差异（记录，不合并）

Phase 5A 评分中同一 output_sha256 在两个试听上下文得分不同，
属评测上下文方差，**不视作独立声学样本**：

- `15228e11…`（mid_exposition_seed42）：stability vs prompt_ab 两
  上下文 naturalness 分歧
- `3f722e6e…`（rain_seed42）：clarity/naturalness 分歧

manifest `context_variance` 逐字保留全部原始评分。

## Codec 风险声明

Phase 5A 已确认 `codec_reconstruction_limit_supported`（VAE 本体
重建上限，双原声 × 双解码条件人耳 1/1）。本实验**不重新打开**该
结论；zero-shot 与 codec 重建共享 decoder 但 latent 分布不同，
codec 上限不构成对本轮生成质量的直接推论。

## 试听页与解盲

- 页面：`outputs/gates/suoming_voxcpm_phase5a2/listen/index.html`
- 结构：校准原声（GT + P0/P2 Prompt 原声）+ 7 个配对组，组内
  甲/乙为匿名条件（同文同 seed 仅 Prompt 不同），「解盲」按钮 +
  `unblind_map.json` 提供真值映射
- 导出评分绑定 `experiment_id + case_id + output_sha256`，含
  `text_complete` 字段
- **主观维度当前全部 `PENDING_USER_LISTENING`；未预填任何分数**

## 结论

- P2 的 Phase 5A 优势能否泛化到两条短句 × 3 seed 与长句 seed44：
  **PENDING_USER_LISTENING**（需要甲/乙盲听评分）
- `short_response` 声学失稳是否在 P2 下消失：**PENDING_USER_LISTENING**
- `mid_exposition_seed44` 自然度 1 的异常是否在 P2 下改善：
  **PENDING_USER_LISTENING**
- **P2 仍是 `candidate`，生产默认 Prompt 未改变**

## 训练建议

`PENDING_USER_LISTENING` → 待人工评分后仅在
`RECOMMEND_PHASE5B_LORA_PLAN` / `MORE_EVIDENCE_NEEDED` /
`DO_NOT_TRAIN_YET` 中选择 → **`PENDING_USER_DECISION`**。
本轮不启动任何 LoRA/SFT 训练。

## 产物

```
outputs/gates/suoming_voxcpm_phase5a2/
  manifest.json            # 14 格 + 7 pair + 溯源 + context_variance
  metrics.json             # 新增 10 条客观指标（24k 分析带）
  report.md                # 自动汇总
  asr_check.json           # SenseVoice 辅助转写
  listen/index.html        # 配对盲听页
  listen/unblind_map.json  # 甲/乙 → 真条件映射
  pair_*.wav               # 10 条新增（本地，gitignored）
  *_original.wav           # GT + 两个 Prompt 校准原声
logs/                      # worker 日志
```
