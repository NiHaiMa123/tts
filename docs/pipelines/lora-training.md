# VoxCPM2 LoRA 训练管线（角色级 adapter）

> 适用场景：零样本克隆相似度不够用（人耳判断"不像"）时，用角色冻结数据集
> 训一个小 LoRA 看能否增益。**训练是最后一道手段**，顺序永远是：
> 数据集入库 → 选参考音 → 零样本评测 → 人耳确认不够 → 才训练。

## 前置事实（锁暝 pilot 实测，勿当假设）

- 训练在 **16GB 单卡可行**：150 optimizer steps ≈ 38min，VRAM 峰值 15.8GB
- 150 步对 1.4B 级模型只是"轻碰权重"——train loss 波动、val loss 不降属正常
- 锁暝 pilot 结论 `PILOT_NO_CLEAR_GAIN`：LoRA 对**已经很像的零样本**无清晰增益。
  但对零样本本就偏弱的角色（爱弥斯 cos_centroid ~0.84），仍值得一试
- **判定只能来自人耳 A/B**，loss 曲线不作音质依据
- adapter 转正后放 `assets/characters/<id>/adapters/voxcpm2/<name>/`（gitignored）

## 环境

| 项 | 位置 |
|---|---|
| 推理 env | `backend_envs/voxcpm2`（生产，**不要动**） |
| 训练 env | `backend_envs/voxcpm2_train`（独立 venv，torch 2.11+cu128） |
| 官方训练入口 | `backend_envs/voxcpm2/official_v203/train_voxcpm_finetune.py`（tag 2.0.3） |
| 官方模板 | `backend_envs/voxcpm2/official_v203/conf/voxcpm_v2/voxcpm_finetune_lora.yaml` |

## 步骤

### ① 数据准备（防泄漏导出）

```powershell
.\.venv\Scripts\python.exe scripts\lora\prepare_voxcpm_lora_data.py `
    --character aimisi
# → outputs/training/aimisi_voxcpm_lora_pilot/{train.jsonl, val_metrics.jsonl, audit/}
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

其余保持模板值（`num_iters: 150` 硬上限、lr 1e-4、LoRA r/α=16 lm+dit）。

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

盲听给出结论：`PILOT_PROMISING`（转正 adapter 到
`assets/characters/<id>/adapters/voxcpm2/`）或 `PILOT_NO_CLEAR_GAIN`（关闭）。

### ⑤ 转正（仅当人耳确认增益）

把选中 ckpt 复制到 `assets/characters/<id>/adapters/voxcpm2/<name>/`，
角色 yaml `adapters` 块登记——推理侧 worker 通过
`VoxCPM.from_pretrained(lora_config=..., lora_weights_path=...)` 加载。

## 不变量

- 训练不动推理 env、不动生产角色 yaml；一切在 `outputs/training/` 沙箱
- 冻结数据集不可变；要换数据先 freeze v2
- 人耳 > 一切指标；LoRA 默认不升级，证明有效才转正
