# Phase 5B — VoxCPM2 LoRA 小规模可行性试验（Pilot v1）

日期：2026-10-08 · 实验 ID：`suoming_voxcpm_lora_pilot_v1`
状态：**`PILOT_READY_FOR_USER_AB`** — 试听包已就绪，等用户低疲劳 A/B 评审

## 结论先行

- 训练在 16GB 单卡上**实测可行**：150 optimizer steps 完成，
  官方入口、独立训练环境、无 OOM、无 NaN、无越界写生产路径。
- 样本与盲听包已生成：6 base（全 sha 复用）+ 18 LoRA（3 ckpt × 6），
  24/24 ok、retry 全 0、无削波。
- **未做任何音质结论**：`PILOT_NO_CLEAR_GAIN` / `PILOT_PROMISING`
  只能在用户真实 A/B 后填写；当前主观项全 `PENDING_USER_LISTENING`。
- 未自动追加任何训练；P2 未参与；生产默认未变。

## 训练溯源

| 项 | 值 |
|---|---|
| 官方入口 | `scripts/train_voxcpm_finetune.py` @ **tag 2.0.3**（commit `19b6bf75`），voxcpm 包 2.0.3 自带 `voxcpm.training` |
| 官方模板 | `conf/voxcpm_v2/voxcpm_finetune_lora.yaml`（可审副本 `configs/training/suoming_voxcpm_lora_pilot.yaml`） |
| 训练环境 | `backend_envs/voxcpm2_train`（独立 venv，torch 2.11+cu128；**不动推理环境**） |
| Base | `openbmb/VoxCPM2` snapshot `32279eff`（architecture=voxcpm2 已校验） |
| LoRA | lm+dit，proj 关；r=16 α=16 dropout=0（试验值，不宣称最优） |
| Optimizer step 口径 | 官方外层 `num_iters`（每轮 1 次 optimizer.step；grad_accum 8 是 micro-step，**不偷换口径**） |
| 步数 | 恰好 150（cap 硬执行；resume 超 cap 会被拒绝） |
| 保存点 | step_0000050 / step_0000100 / step_0000150（官方另存了 step 0/149 的附带目录，未纳入评测） |
| 时长 | 2262s（~38min）；VRAM 峰值 15.8GB（nvidia-smi 全程采样） |
| 失败 | 无 OOM / 无 NaN / 无异常；smoke（2 step）先行通过 |

### Loss 记录（如实，不作音质判据）

| step | train loss/diff | val loss/total | grad_norm |
|---|---|---|---|
| 0 | 0.756 | 0.960 | 0.035 |
| 50 | 0.760 | 0.904 | 0.062 |
| 100 | 0.826 | 0.945 | 0.072 |
| 149/150 | 0.620 | 0.976 | 0.073 |

train loss 在 0.6–0.9 间波动、val loss 无持续下降——150 step
对 1.4B 级模型只够“轻微碰一下权重”，这**不**说明 LoRA 无效、
**也**不说明有效；判据只能来自人耳 A/B。

## 数据冻结与防泄漏（PLAN 25.3）

- 源：`dotstts/datasets/suoming/v1/train.jsonl`（train split only）
- **接受 127 / 拒绝 1**，总时长 809.5s（均值 6.37s，范围 2.0–30.2s）
- 唯一拒绝：`leak_text_exact`——**train split 中混入了 P0 谛天鉴
  Prompt 本体的文本记录**（fid=c4322e32…），若放入会污染评测；
  已记入 `audit/rejects.json`
- 校验项：文件存在/可读、时长 0.4–90s、非静音（rms>-50dBFS）、
  非 NaN、文本非空、train 内音频/文本去重、与冻结资产（P0/P2
  prompt、rain GT、全部评测文本及源录音 sha）做 sha+fid+精确文本+
  相似度（≥0.85）四重去重
- val_manifest：8 条 validation 记录，**仅供 loss 记录**，未训练、
  且避开了全部冻结资产 fid/text/sha（列模式与 train.jsonl 一致）
- 训练数据不入 Git；`audit/audit.jsonl` 保留每行 fid+sha256+判定

## 评测矩阵（PLAN 25.5，P0 固定参考）

| case | text | seed | base 来源 | LoRA |
|---|---|---|---|---|
| rain | 雨景 | 42 | 复用 `stability/rain_seed42` | 3 ckpt 新生成 |
| short_response | 开伞 | 42,43 | 复用 phase5a ×2 | 6 |
| short_response_2 | 跟着我们 | 42,43 | 复用 phase5a2 P0 ×2 | 6 |
| mid_exposition | 中长句 | 42 | 复用 | 3 |

- Base 6/6 全部**复用成功**：模型 revision、P0 sha、text、seed、
  effective_args、磁盘 sha 逐项验证，无一静默重生成
- LoRA 18 条走**正式推理路径**（`VoxCPM.from_pretrained` +
  `lora_weights_path`，同一推理 env、同一 P0、同一冻结参数），
  绝非训练期 teacher-forced 重建
- 预算 18/24；adapter weights sha 三 ckpt 各异并已记入 manifest
- 每格 sidecar：checkpoint、lora targets、train data sha、prompt sha、
  model rev、seed、retry、raw sha、耗时

## 自动筛查（只标记，不评判）

- 工程项全绿：24/24 可读、48kHz mono、无 NaN/削波/静音、retry=0、
  `screening.json` flags 为空
- 时长一致性：同一 (text,seed) 下 base 与 ckpt 时长差 ≤0.7s；
  唯一例外 `mid_exposition_seed42` ckpt100 22.4s vs base 18.1s
  （silence 0.52 vs 0.39，疑似尾部静音拉长——已列为待确认候选，
  供首听验证）
- ASR（SenseVoice 3847d57b，辅助）：`short_response_2` 全 0 CER；
  `short_response` 多数格同错「开→凯」（含 base，非 LoRA 引入）；
  `mid`/`rain` 0.08–0.14 以专有名词/同音字为主，无随 step 单调
  恶化；`text_complete` 仍由人耳判
- 主观类异常检测无可靠标签 → 全部 `UNDETERMINED`，只出候选清单

## 低疲劳 A/B 试听包

`outputs/gates/suoming_voxcpm_lora_pilot_v1/listen/index.html`

- **校准**：先听干净原声（rain GT）+ P0 prompt，随时可回校准
- **第 1 组（默认展开，4 对）**：base vs step100 ×
  {rain42, short42, short2_42, mid42}（两短一中一雨）
- **第 2 组（折叠，2 对）**：step50 vs step150 × 两条短句——
  检验“早期 checkpoint 是否已够/更多步是否反坏”
- **第 3 组（折叠，14 对）**：其余 base-vs-ckpt 配对，可不听
- 每对：甲/乙 随机隐藏（pair_id 决定论洗牌）、选项
  「甲更好/乙更好/无明显差异/暂无法判断」、缺陷标签
  （磨砂/舌位/口型/语速/音色/漏字/金属感/其他）、
  first_impression + severity + 可选时间戳、可折叠四维分
- **UNCERTAIN_FATIGUE**：一键标记会话疲劳并停止；疲劳≠无差异，
  未听项导出为 `PENDING_USER_LISTENING`
- 同 SHA 重复样本自动标记 `duplicate_sha`，不计独立证据
- 导出含：experiment_id、pair_id、A/B 真实 sha+checkpoint、
  盲测顺序、choice（含真实条件映射）、first_impression、
  defect_tags、可选分数、fatigue 标记、时间戳、批次

## 验收核对（PLAN 25.8）

- [x] 官方入口确认 + 数据可审计 + 训练隔离（专用 env，未动生产）
- [x] 16GB 实测可行（15.8GB 峰值，含桌面占用）
- [x] ≤150 optimizer steps、3 候选 ckpt、全 provenance
- [x] 6 base 复用校验 + 18 LoRA 正式推理、真 SHA
- [x] 自动初筛 + 每批 3–5 对 + 疲劳/不确定导出
- [x] 主观全 `PENDING_USER_LISTENING`，无自动升级 adapter
- [x] 训练已停止，无第二轮 sweep

## 待用户决定

1. 打开试听页完成第 1 组（4 对，约 5–10 分钟），疲劳即停；
2. 据 A/B 结果选：`PILOT_NO_CLEAR_GAIN`（保留 zero-shot）或
   `PILOT_PROMISING`（再谈扩数据/步数，仍需用户授权）；
3. 是否使用 step50/100/150 中某一个做后续——未授权前生产默认
   仍是 base zero-shot。

## 产物（本地，全部 gitignored）

```
outputs/training/suoming_voxcpm_lora_pilot/   # 数据+checkpoints+日志+audit
outputs/gates/suoming_voxcpm_lora_pilot_v1/   # 24 wav+manifest+screening+listen
docs/reports/suoming-voxcpm-lora-pilot-v1.md  # 本报告（入库）
backend_envs/voxcpm2_train/                   # 独立训练 env（不入 git）
backend_envs/voxcpm2/official_v203/           # 官方脚本+模板副本（不入 git）
```
