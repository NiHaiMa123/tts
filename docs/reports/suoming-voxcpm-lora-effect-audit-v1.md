# Phase 5C — VoxCPM2 LoRA 加载与生效审计（Audit v1）

日期：2026-10-08 · 审计 ID：`suoming_voxcpm_lora_effect_v1`
最终判定：**`ADAPTER_EFFECTIVE_NO_CLEAR_GAIN`** —— LoRA 确实被完整加载、
真实生效且可复现；Phase 5B 的对照条件可比性全部通过；但既有用户 A/B
（base 3:1，无清晰增益）维持有效 → **保留 zero-shot，不升级 adapter**。

## 证据链（每条均可复现）

### 1. 权重静态审计 —— 训练确实更新了 LM+DiT

`outputs/diagnostics/.../weights_metrics.json`（本地）

| ckpt | keys | lm (A+B) | dit (A+B) | numel | 组 L2 | 异常 |
|---|---|---|---|---|---|---|
| step50 | 384 | 144+144 | 48+48 | 9.04M | lm 28.24 / dit 16.33 | 0 |
| step100 | 384 | 144+144 | 48+48 | 9.04M | lm 28.75 / dit 16.64 | 0 |
| step150 | 384 | 144+144 | 48+48 | 9.04M | lm 28.82 / dit 16.69 | 0 |

- 无 NaN/Inf、无全零张量；三个 ckpt 的 `lora_config.json` 完全一致
  （lm+dit、proj 关、r=16、α=16、dropout=0）
- **步间差异是真实更新**，非元数据差异：
  50→100 全部 384 键变化（max rel_l2 = 0.674）；
  100→150 全部变化（max rel_l2 = 0.091，幅度收敛）；L2 范数单调增长
- 结构覆盖：192 个 LoRALinear 模块（2 个 LM × 36 层 × 4 投影 + DiT
  12 层 × 4 投影），与训练配置 target_modules 一致，proj 未注入

### 2. 加载完整性 —— `LOAD_OK`，384/0

`outputs/diagnostics/.../load_report.json`（本地）

- 模型按 ckpt `lora_config` 构建 192 个 LoRA 模块（lm 144 + dit 48，
  对应 384 个参数键）
- `load_lora_weights(step_0000100)` → **loaded 384 / skipped 0**，
  无 missing/unexpected key
- `set_lora_enabled(False)` → 192/192 模块 disabled；
  `(True)` → 192/192 enabled（scaling buffer 机制，compile 安全）

### 3. 最小开关实验 —— `ADAPTER_EFFECTIVE`（4 条受控 WAV）

同 P0、同文（「开伞，由我来动手。」）、seed 42、冻结生成参数，
同一推理 env（voxcpm 2.0.3）：

| 条件 | 构造 | WAV sha256 | 结论 |
|---|---|---|---|
| A | 原生 base（无 LoRA 结构） | `652929db…b323` | 冻结基线 |
| B | 加载 step100 + `set_lora_enabled(false)` | `652929db…b323` | **与 A 逐字节相同** → 关闭真实归零 |
| C | 同实例 + `set_lora_enabled(true)` | `dd4e3c29…c7d5` | ≠ B（波形 corr 0.105）→ **LoRA 在计算图中实际生效** |
| C-repeat | 同 C 重跑 | `dd4e3c29…c7d5` | 与 C 逐字节相同 → 确定性 |

### 4. 跨次复现 —— Phase 5B 条件逐字节复现

- 审计 A == Phase 5B `base/short_response_seed42` 完整 sha 一致
- 审计 C == Phase 5B `ckpt100/short_response_seed42` 完整 sha 一致
- → Phase 5B 的 seed 传递（`torch.manual_seed`+`cuda.manual_seed_all`，
  tag 2.0.3 无 `seed` API）、prompt、参数、模型 revision 全部真实一致；
  生成在该机制下是**逐字节确定性**的

### 5. Phase 5B 可比性复核 —— `COMPARABLE`

`outputs/diagnostics/.../comparability.json`（本地）

- 24/24 输出 WAV 重新哈希 == manifest 记录，无覆盖、无漂移
- 全部配对：seed 一致、P0 prompt sha 一致、retry=0、同 snapshot、
  同生成参数、同输出格式（48k float WAV）
- base（worker 路径）与 LoRA（eval 脚本路径）用**同一 RNG 固定机制**；
  `load_denoiser=false`、`denoise=False` 两侧一致

## 排除的假设

- ~~没有成功加载/权重被跳过~~ → 384/0、enabled 192/192、B≠C 且 C 复现
  Phase 5B 输出
- ~~测试链路不可比~~ → 同一 case 的 base/LoRA 输出分别在两次独立运行
  中逐字节复现
- 剩余解释：**LoRA 生效但效果方向不受欢迎**（用户备注「更加清冷」）
  —— 属训练目标/数据代表性问题，不是实现错误

## 发现的缺陷（已修，不影响 Phase 5B 结论）

审计脚本初版把 ckpt `lora_config.json` 的**外层包装**
（`{base_model, lora_config:{…}}`）直接喂给 `LoRAConfigV2(**…)`；
pydantic 默认忽略未知字段 → 静默构造了全 disable 的配置 → 模型 0 个
LoRA 模块 → 384 键全 skipped。该事故恰好演示了 PLAN 26.4 要求记录
loaded/skipped 而非只看「启动成功」的必要性。修复：`_inner_lora_cfg`
提取内层配置（`_lora_cfg.json` 平铺格式不受影响——**Phase 5B eval
当时用的就是平铺文件，未踩此坑**）。

## 建议（PLAN 26.7）

- **结束本次 LoRA pilot，保留 zero-shot**；
- 若继续研究：优先查数据代表性（127 条 ~13.5min、短句主导）、
  训练目标（diff loss 与听感脱节）与微调引导下的韵律/情感漂移，
  而非盲目加步数；val 集仅 8 条，0.904→0.976 不构成过拟合定论
- 生产默认未变；未训练；无需用户补听

## 复现命令

```bat
backend_envs\voxcpm2\Scripts\python.exe scripts\audit_voxcpm_lora_effect.py ^
  --snapshot <VoxCPM2 snapshot 32279eff> ^
  --ckpt-root outputs\training\suoming_voxcpm_lora_pilot\checkpoints ^
  --out-dir outputs\diagnostics\suoming_voxcpm_lora_effect_v1 ^
  --steps static,load,onoff --text "开伞，由我来动手。" ^
  --prompt-wav <P0 wav> --prompt-text <P0 text> --seed 42
```

入库：`scripts/audit_voxcpm_lora_effect.py`、
`src/character_tts/evaluation/lora_audit.py`、
`tests/test_lora_effect_audit.py`、本报告。
本地（不入 git）：`outputs/diagnostics/suoming_voxcpm_lora_effect_v1/`
（4 WAV + 全部 JSON）。
