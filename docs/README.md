# 文档导航

项目用法从 [README](../README.md) 开始，Agent 的当前任务与完成状态从
[PLAN](../PLAN.md) 开始。本目录把操作规范、研究证据和历史记录分开。

## 文档职责

| 类型 | 负责什么 | 更新方式 |
|---|---|---|
| README | 使用入口、运行配置摘要、专题导航 | 保持简短；详细命令进入对应 pipeline |
| PLAN | 当前状态、待办顺序、交接与完成条件 | 更新既有条目，不累积历史阶段全文 |
| pipelines | 可重复执行的操作规范 | 一个操作只保留一份规范，其他文档链接到它 |
| reports | 固定日期/版本下的观察、论文依据、复现与限制 | 保留历史事实；候选建议不自动成为当前任务 |
| archive | 被替代的实施计划 | 原文保留，仅用于追溯；其中 ACTIVE 等措辞无当前效力 |

运行事实以实际配置、源码和可追溯记录为准；当前任务范围以用户明确指令为准。
文档冲突时先修正当前入口和专题规范，再标注报告适用范围。

## 操作规范

| 要做的事 | 阅读入口 | 当前实现情况 |
|---|---|---|
| 安装环境、下载模型、运行 ASR/CAM++ | [environment.md](pipelines/environment.md) | 主平台、后端及辅助脚本已存在；实际安装版本需留证 |
| 音频入库、审查、选参考与冻结数据 | [voice-assets.md](pipelines/voice-assets.md) | 已有管线；预处理 `--cut` 已确认的缺陷尚未修复，先按文中质量门处理 |
| Vox LoRA 数据准备、训练、加载与评测 | [lora-training.md](pipelines/lora-training.md) | 已有工具；现有模板是 pilot 配方，不代表任何新角色必然增益 |
| 展示实验音频、选择后写盘、说“继续” | [listening-review.md](pipelines/listening-review.md) | 同一模板与保存/读取流程已实现；无导出步骤 |
| 统一计算音质、音色、习惯、内容、稳定性 | [self-check.md](pipelines/self-check.md) | 规范已完成；统一评分入口与项目校准尚待实现 |

## 研究与代码审查

以下四份报告各保留一类证据，任务顺序已合并到 PLAN。无需每次从头读全部报告。

| 问题 | 对应证据 | 适用边界 |
|---|---|---|
| 本项目代码是否会伤害效果？ | [音质与数据链审查](reports/tts-quality-review-2026-10-13.md) | 有合成信号复现；本地 v3 触发数量与全部音质根因未确定 |
| 开源 TTS 是否都存在这种取舍，是否到达边界？ | [开源能力边界](reports/open-source-tts-capability-boundaries-2026-10-13.md) | 文献、官方实现与候选优化；不是本项目新实验结果 |
| 为什么 Qwen 清晰零样本，训练后严重破损？ | [Qwen 训练专项](reports/qwen3-tts-training-distortion-audit-2026-10-13.md) | 上游四项不一致已核查；是否落在本地实际训练路径仍待证实 |
| 用什么方法评分，数学定义和依据是什么？ | [自检方法研究](reports/tts-unified-self-check-protocol-2026-10-13.md) | 保留论文、指标、校准、预算及验收依据；操作和实施任务统一到自检 pipeline |

## 历史实验

锁暝早期阶段均已完成；历史建议不触发重跑。各报告的复现路径可能是当时目录，
现有脚本入口在 `scripts/eval/` 与 `scripts/lora/`。原始评分 JSON 原路径保留。

| 阶段 | 报告 | 评分原件 / 最终状态 |
|---|---|---|
| Backend gate v1 | [backend-gate](reports/suoming-backend-gate-v1.md) | [gate 评分](reports/suoming-gate-v1-user-ratings.json) |
| Phase 5A：稳定性与 codec | [phase5a](reports/suoming-voxcpm-phase5a.md) | [5A 评分](reports/suoming-voxcpm-phase5a-ratings.json) |
| Phase 5A2：P2 短句与评分一致性 | [phase5a2](reports/suoming-voxcpm-phase5a2.md) | [5A2 评分](reports/suoming-voxcpm-phase5a2-ratings.json) |
| Phase 5B：LoRA pilot | [lora-pilot](reports/suoming-voxcpm-lora-pilot-v1.md) | [pilot 评分](reports/suoming-voxcpm-lora-pilot-v1-ratings.json)；`PILOT_NO_CLEAR_GAIN` |
| Phase 5C：LoRA 加载与生效 | [lora-effect-audit](reports/suoming-voxcpm-lora-effect-audit-v1.md) | `ADAPTER_EFFECTIVE_NO_CLEAR_GAIN`；锁暝保留零样本 |

旧阶段设计、预算、当时 Devin 指令和原架构约定完整保存于
[历史实施计划](archive/implementation-plan-through-2026-10-13.md)。

## 常用阅读顺序

- 日常生成：README → 环境说明（需要时）。
- Agent 接任务：PLAN → 本次操作的 pipeline → 对应证据报告。
- 查评分与论文：自检 pipeline → 自检研究中的算法、校准和来源。
- 查当时为何作某决定：历史实验报告 → 对应评分原件 → 归档计划。
