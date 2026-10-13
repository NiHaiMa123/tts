# VoxCPM2 LoRA 训练管线（角色级 adapter）

> 本文件保留 Vox 训练工具的操作规范与角色案例，不构成新的训练任务。任务范围见 [PLAN](../../PLAN.md)，评价见 [统一自检](self-check.md)，试听见 [统一试听规范](listening-review.md)。

> 适用场景：零样本克隆相似度不够用（人耳判断"不像"）时，用角色冻结数据集
> 训一个小 LoRA 看能否增益。**训练是最后一道手段**，顺序永远是：
> 数据集入库 → 选参考音 → 零样本评测 → 人耳确认不够 → 才训练。

## 前置事实（锁暝 pilot 实测，勿当假设）

- 训练在 **16GB 单卡可行**：150 optimizer steps ≈ 38min，VRAM 峰值 15.8GB
- 150 optimizer steps 是该次 pilot 预算；不能仅凭步数或模型规模判断更新幅度、过拟合或音质
- 锁暝 pilot 结论 `PILOT_NO_CLEAR_GAIN`：LoRA 对**已经很像的零样本**无清晰增益。
  该结论不能推广到所有角色；爱弥斯后续另有 v3 预设，见本文件角色案例
- 音质、音色和习惯分别判断；loss 曲线不作听感依据。经过项目校准的指标可筛查，最终候选须保留有效的人耳证据
- adapter 转正后放 `assets/characters/<id>/adapters/voxcpm2/<name>/`（gitignored）

## 环境

| 项 | 位置 |
|---|---|
| 推理 env | `backend_envs/voxcpm2`（生产，**不要动**） |
| 训练 env | `backend_envs/voxcpm2_train`（独立 venv，torch 2.11+cu128） |
| 官方训练入口 | `backend_envs/voxcpm2/official_v203/train_voxcpm_finetune.py`（tag 2.0.3） |
| 官方模板 | `backend_envs/voxcpm2/official_v203/conf/voxcpm_v2/voxcpm_finetune_lora.yaml` |

## 步骤

### ⓪ 数据与计算契约检查

爱弥斯曾出现习惯更像但磨砂；其全部原因尚未确定。先审计数据、参考与实际加载，
不能将此直接归因于 DiT 或声学签名，也不能把 VAD 清洗视为已经奏效的修复。
[资产规范第1b节](voice-assets.md) 列明 `--cut` 已确认而尚未修复的缺陷。
需删除有声内容的样本先隔离并校对台词；权威来源、文本和原划分必须保留，
VAD失败/全段拒绝不得自动进入训练。现有 v3 先做审计，再确定修复方向。

以下 CLI 仅保留作复现与修复验证，通过质量门后才标准化、审核和冻结新版本：

```powershell
.\.venv\Scripts\python.exe scripts\ingest\preprocess_audio.py `
    --inbox data\characters\<id>\inbox `
    --out   data\characters\<id>\inbox_clean --cut
# 然后 standardize+freeze 到 datasets\v<N>，train/val manifest 指 v<N>
```

### ① 数据准备（防泄漏导出）

```powershell
.\.venv\Scripts\python.exe scripts\lora\prepare_voxcpm_lora_data.py `
    --character aimisi
# → outputs/training/aimisi_voxcpm_lora_pilot/{train.jsonl, val_metrics.jsonl, audit/}
# 用预处理 v2 集训练时覆盖清单：
#   --train-manifest data/characters/aimisi/datasets/v2/train.jsonl `
#   --val-manifest   data/characters/aimisi/datasets/v2/validation.jsonl `
#   --out-dir outputs/training/aimisi_lora_v2
```

- 读角色 yaml 的 `dataset.train_manifest` / `validation_manifest`
- 逐条校验：文件存在、时长 0.4–90s、非静音、音频/文本去重
- **防泄漏四重校验**：参考音 sha、角色 yaml `evaluation.anchor_texts`、
  （锁暝另有 phase5a/5a2 冻结配置）——sha + fid + 精确文本 + 相似度≥0.85
- 每条接受/拒绝都记 `audit/audit.jsonl`；接受 <30 条或 <120s 时
  `decision=DATA_BLOCKED` 拒绝继续

### ② 训练配置

复制 `configs/training/suoming_voxcpm_lora_pilot.yaml`，改四处路径：

```yaml
train_manifest: E:/project/tts/outputs/training/aimisi_voxcpm_lora_pilot/train.jsonl
val_manifest:   E:/project/tts/outputs/training/aimisi_voxcpm_lora_pilot/val_metrics.jsonl
save_path:      E:/project/tts/outputs/training/aimisi_voxcpm_lora_pilot/checkpoints
tensorboard:    E:/project/tts/outputs/training/aimisi_voxcpm_lora_pilot/tb
```

这些是现有 pilot 模板值（`num_iters: 150` 硬上限、lr 1e-4、LoRA r/α=16 lm+dit），
不是所有角色的最优参数或新训练授权。新试验先固定具体范围、评价与停止条件。

### ③ 训练

```powershell
# 冒烟先行（2 step，不留 checkpoint）
.\.venv\Scripts\python.exe scripts\lora\train_voxcpm_lora_pilot.py --smoke `
    --config configs/training/aimisi_voxcpm_lora_pilot.yaml `
    --out    outputs/training/aimisi_voxcpm_lora_pilot

# 正式跑（≤150 step，OOM 自动拒绝；真 OOM 才允许 --oom-fallback 一次）
.\.venv\Scripts\python.exe scripts\lora\train_voxcpm_lora_pilot.py `
    --config configs/training/aimisi_voxcpm_lora_pilot.yaml `
    --out    outputs/training/aimisi_voxcpm_lora_pilot
```

产出：`checkpoints/step_0000050|100|150/lora_weights.safetensors`、
`logs/train.log`、`run_log.json`（VRAM/耗时/状态）。

安全机制：150 步硬上限（resume 超上限拒绝）、NaN/OOM/异常即停、
VRAM 全程 nvidia-smi 采样。

### ④ 评测 + 人耳 A/B

```powershell
# 复制 configs/evaluations/suoming_voxcpm_lora_pilot_v1.yaml 改成角色版，
# 指向新 ckpt，然后：
.\.venv\Scripts\python.exe scripts\lora\run_voxcpm_lora_eval.py `
    --config <eval_config> --character aimisi
# → <output_dir>/listen/index.html 盲听页（serve_review.py 起服务，点选即写盘）
```

盲听和有效自检共同形成证据；`PILOT_PROMISING` 表示候选，
`PILOT_NO_CLEAR_GAIN` 表示未观察到清晰收益。候选先验证生产路径和留出样本，再进入下一节。

### ⑤ 转正（仅当人耳确认增益）

把选中 ckpt 复制到 `assets/characters/<id>/adapters/voxcpm2/<name>/`，
推理侧现有 worker 从后端 YAML 的 `model.lora` 读取配置，通过
`VoxCPM.from_pretrained(lora_config=..., lora_weights_path=...)` 加载。
角色 YAML 的 `backend` 绑定对应实例；仅填写角色 `adapters` 块不等于已经接入加载。
现有爱弥斯仍从下文的 `outputs/training/` 路径加载，迁移资产时需同步配置与权重 hash。

## 不变量

- 训练不动推理 env、不动生产角色 yaml；一切在 `outputs/training/` 沙箱
- 冻结数据集不可变；要换数据先 freeze v2
- 人耳标签与计算分数分开；质量不能被音色收益抵消，验证有效后才进入生产候选

## 爱弥斯生产配置记录（2026-10）

当前配置见 [后端](../../configs/backends/voxcpm2_aimisi.yaml) 与 [角色](../../configs/characters/aimisi.yaml)。
以下是既有实验及人耳记录；对全部残余问题的归因尚未验证。

- **adapter**：`outputs/training/aimisi_lora_v3/checkpoints/step_0000150`
  （v3 VAD 剪切数据，LM+DiT r16α16）。v3 vs v1：音色更像；v1 节奏
  略优但差距不足以翻盘；v2 降噪版已弃。
- **参考音**：`assets/characters/aimisi/reference/ref_d8b9.wav`
  （克隆锚点实测最优，cos_centroid 0.899 vs 旧 ref 0.842）。
- **seed**：123（crack_* A/B：s42/s123/cfg15/t15/ref39d0/step100 七变量
  盲评，s123 压嗓毛刺最少——破音部分来自采样运气）。
- **接线**：专属后端 `configs/backends/voxcpm2_aimisi.yaml`
  （`model.lora` 注入），角色 yaml `backend: voxcpm2_aimisi` +
  `generation.seed: 123`；与共享 voxcpm2 worker 互斥，切角色时
  manager 停旧起新。
- 已知残余问题：偶发「要破音」压嗓段仍存在（v1/v3 同源数据）。
  F0/HNR 等只能作为定位证据，气声或情绪发声不自动是坏样本；后续按自检规范审计，再决定是否改数据或训练。

## Qwen3-TTS LoRA 试验（2026-10，关闭）

- 管线：官方 finetune + 社区 LoRA patch（r16α32，Q/K/V/O+MLP，
  lr2e-6×10epoch），179 条 v3 数据，19.2M/1.95B 可训，loss 14.7→10.2。
- **结论：失败**。零样本清晰但音色不够像角色；LoRA e9 在
  scale 0.3/0.6 下输出破损扭曲，scale 1.0 直接 EOS 失控（33s→327s）。
  当时归因是“adapter 学坏了声学分布，非缩放问题”；这并未排除错误监督、加载或生成模式失配。
  [Qwen 专项审查](../reports/qwen3-tts-training-distortion-audit-2026-10-13.md) 已在被核查的上游版本发现四项训练不一致，
  是否落在本地实际 patch/包源码仍待取证。先核对计算契约，再考虑 LR、数据或模块范围；不能把旧归因当作已确定根因。
- 资产保留：`outputs/training/aimisi_qwen3_lora/`（jsonl+codes+10 个
  ckpt）、`backend_envs/qwen3_tts/official_finetune/`（脚本）。
