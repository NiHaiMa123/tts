# Phase 5A2 — VoxCPM2 P0/P2 短句稳定性配对对照

日期：2026-10-08 · 实验 ID：`suoming_voxcpm_phase5a2`
状态：**14 格全部完成人工盲听（14/14 sha 校验通过）**

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
  P0 + seed/text/effective_args 逐项比对；任一失败记
  `BLOCKED_REUSE`，绝不静默重生成。本轮 4/4 全部校验通过。
- 预算：`zero_shot_generated = 10/10`（硬上限）。

## 执行结果

- 14/14 格 `ok`；`retry_count` 全 0；无削波、无 NaN；时长 2.2–21.8s
- 14/14 盲听评分已回收，`output_sha256` 与当前 manifest 全部一致
  （`docs/reports/suoming-voxcpm-phase5a2-ratings.json`）

## ASR 辅助证据（SenseVoice rev 3847d57b，仅辅助）

| pair | P0 CER | P2 CER |
|---|---|---|
| A·s42 | 0.143（开→凯） | **0.000** |
| A·s43 | 0.143（开→凯） | 0.143（开→凯） |
| A·s44 | **0.000** | 0.143（开→凯） |
| B 全部 | 0.000 ×3 | 0.000 ×3 |
| C·s44 | 0.099 | 0.085 |

pair B 两 Prompt 字全对；pair A「开→凯」P0/P2 各两次，非 Prompt
特异；pair C 两侧同错「心月狐→新月湖」「祂→他」专有名词层面。
ASR 只证明说了什么，`text_complete` 仍以人耳为准。

## 人耳结果（1–5，越高越好；grit 越高越不磨砂）

### 每对四维均值（P2 − P0）

| pair | P0 均值 | P2 均值 | delta |
|---|---|---|---|
| A·s42 | 3.75 | 4.50 | **+0.75** |
| A·s43 | 5.00 | 4.00 | **−1.00** |
| A·s44 | 3.75 | 4.00 | +0.25 |
| B·s42 | 5.00 | 4.50 | −0.50 |
| B·s43 | 5.00 | 4.75 | −0.25 |
| B·s44 | 4.75 | 5.00 | +0.25 |
| C·s44 | 4.75 | 5.00 | +0.25 |

全 7 对平均 delta ≈ **+0.11**，方向不一致。

### 逐格明细（c/g/l/n）

| case | P | c | g | l | n |
|---|---|---|---|---|---|
| A·s42 | P0 | 3 | 3 | 4 | 5 |
| A·s42 | P2 | 5 | 5 | 5 | **3** |
| A·s43 | P0 | 5 | 5 | 5 | 5 |
| A·s43 | P2 | 5 | 5 | 5 | **1** |
| A·s44 | P0 | **2** | 3 | 5 | 5 |
| A·s44 | P2 | 5 | 5 | 3 | 3 |
| B·s42 | P0 | 5 | 5 | 5 | 5 |
| B·s42 | P2 | 5 | 5 | 3 | 5 |
| B·s43 | P0 | 5 | 5 | 5 | 5 |
| B·s43 | P2 | 5 | 5 | 4 | 5 |
| B·s44 | P0 | 5 | 5 | 4 | 5 |
| B·s44 | P2 | 5 | 5 | 5 | 5 |
| C·s44 | P0 | 5 | 5 | 4 | 5 |
| C·s44 | P2 | 5 | 5 | 5 | 5 |

## 同 SHA 跨场次评分差异（重要诚实记录）

4 条复用 WAV 在 Phase 5A 与本轮（同一评审、不同试听上下文）评分
明显不同——**同一文件**，差异即上下文方差：

| sha | 5A 评分 c/g/l/n | 5A2 评分 c/g/l/n |
|---|---|---|
| `652929db` A·s42/P0 | 1/1/3/5 | 3/3/4/5 |
| `e05fd651` A·s43/P0 | 5/5/3/3 | 5/5/5/5 |
| `321fcba2` A·s44/P0 | 4/4/2/1 | 2/3/5/5 |
| `254fed39` C·s44/P0 | 5/5/4/**1** | 5/5/4/**5** |

含义：(a) 单次盲听评分方差不小，Phase 5A「short_response 三 seed
均差」的方向仍成立（本轮 P0 在 s42/s44 仍 ≤3 分），但幅度不可精
确复现；(b) `mid_exposition_seed44` 的 n=1 **未被本轮重现**——该
文件不携带稳定缺陷，原异常更可能是一次性合成瑕疵 + 上下文效应；
(c) 同 SHA 评分只计一次声学证据，不重复计数。

## 结论

1. **P2 优势不泛化**：Phase 5A 中 P2 全 5 分的领先未在本轮 7 对
   中复现——A 组互有胜负（P2 自身在 s43 出现 n=1 崩坏），B 组 P0
   反而略优，C 组 P2 仅 +0.25。Prompt 间的差异远小于 seed/文本
   间的方差。
2. **短句失稳非 Prompt 特异**：P0（s42 c3/g3、s44 c2）与 P2
   （s43 n1、s44 l3）都出现短句低分——这是模型在 ~7–8 字输入上
   的固有方差，换 Prompt 不能消除。
3. **长句 seed44 异常未复现**：同一 WAV 本轮 n5，P2 新增格 n5。
4. **P2 维持 `candidate`**：无足够证据提升为生产默认，生产
   Prompt 未改变。
5. **Codec 风险未重开**：Phase 5A 的
   `codec_reconstruction_limit_supported` 维持有效，与本轮无关。

## 训练建议

`MORE_EVIDENCE_NEEDED` → **`PENDING_USER_DECISION`**

理由：Prompt 选择的收益不确定（P2 未泛化），短句失稳是两侧共有
的模型级问题；LoRA/SFT 是否应针对短句稳定性设计，需要用户先决定
方向。本轮不启动任何训练。

## 产物

```
outputs/gates/suoming_voxcpm_phase5a2/      # 本地，WAV gitignored
  manifest.json  metrics.json  report.md  asr_check.json
  listen/index.html  listen/unblind_map.json
  listen/listen-ratings-phase5a2.json       # 原始导出（本地）
  pair_*.wav (10 新增)  *_original.wav (3 校准原声)
docs/reports/suoming-voxcpm-phase5a2-ratings.json  # 脱敏入库（本仓库）
```
