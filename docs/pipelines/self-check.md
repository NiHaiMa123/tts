# 统一自检：执行与实施规范

本文件是所有后端及 adapter 的自检操作入口；当前任务顺序见 [PLAN](../../PLAN.md)。
方法、公式、论文、具体版本、校准与验收依据保留在
[自检研究报告](../reports/tts-unified-self-check-protocol-2026-10-13.md)，避免复制第二套算法说明。

## 实现状态

| 能力 | 2026-10-13 状态 |
|---|---|
| manifest、生成 sidecar、基础诊断、CAM++ 入库检查、ASR 检查 | 已有，可复用；尚未形成统一评分接口 |
| 共用试听模板、选择/备注写盘、恢复、Agent 读取 | 已实现，见 [试听规范](listening-review.md) |
| `scripts/eval/run_selfcheck.py`、评分适配器、统一配置/报告 | **拟新增，当前不存在** |
| UTMOSv2、SCOREQ、生成窗口身份、同文本韵律与失败定位 | 已调研，尚未接入统一运行 |
| 项目校准及覆盖内 `PASS_AUTO` | 未建立；目前不能声称真实 WAV 评分或质量自动通过 |

## 输入与输出

每次评价要有固定的基线、候选、目标文本、独立原声 gallery 与资产划分。
记录源码、包、model/tokenizer/adapter revision/hash、数据清单、reference、seed、
重试、生成模式、实际参数，以及 raw/analysis/listening 视图。缺少关键身份则 BLOCKED。
优先复用身份一致的既有 WAV；更新页面不重新合成音频。

输出分别覆盖 **音质、声音身份、韵律/习惯、内容、稳定性**，不汇成可互相抵消的总分。
每项保存原始值、基线差、状态、窗口/时间点、模型与权重/config hash。
字段契约见本文件实施顺序，结果示例见研究报告第12节；值不可用时保存 null 与原因，不能以零伪装真实测量。

## 每次自检的顺序

| 步骤 | 操作与结果 | 对应研究定义 |
|---|---|---|
| 0 | 固定资产、输入、版本与 hash，写 `run.json` | 第4节资产；第7节音频视图 |
| 1 | 新训练路线先验训练/推理、零更新、保存重载与合并契约；失败停止升级 | Qwen 特定检查见 [专项报告](../reports/qwen3-tts-training-distortion-audit-2026-10-13.md) 第3、6、7节 |
| 2 | 共用生产 worker 生成或收集原始 WAV、sidecar、manifest；保留全部尝试 | 第9节比较与预算 |
| 3 | 检查解码、空/全零/NaN/Inf、采样率、声道、停止与预算；疑似削波保存定位 | 第6.5节稳定性；第13节验收 |
| 4 | 主 ASR 的 CER/错误位置与关键字；疑似错误用第二 ASR | 第6.3节内容 |
| 5 | 独立多参考 CAM++ 及窗口漂移；按 profile 运行第二编码器/感知评分 | 第6.1节声音身份 |
| 6 | UTMOSv2 与 SCOREQ-synthetic-NR；保存整体及局部诊断，可选 NISQA | 第6.2节音质；第2节模型配置 |
| 7 | 内容有效后对齐，同文本原声算 F0/时长/停顿；无同文本原声仅分析分布/异常 | 第6.4节韵律，不能输出虚假曲线相似度 |
| 8 | 按校准覆盖与必需项汇总 verdict、逐条/逐窗 JSONL、CSV、MD/HTML | 第8节校准及判定 |
| 9 | 仅复核最差局部、冲突与域外样本的小型 A/B；直接写本地评分 | 试听规范；第8.2节低疲劳校准 |

以上自动步骤是目标实现规范，不能把尚未运行的评分项写成已执行。
窗口分析和平均值同时保留，不能用重试、响度修复或平均分遮盖破损。

## 统一判定

| 状态 | 条件与动作 |
|---|---|
| `FAIL` | 确定性故障或已校准失败规则命中；记录原因和片段，停止该候选升级 |
| `REVIEW` | 未校准、指标冲突、疑似损伤或超出域覆盖；保留不确定并有限复核 |
| `BLOCKED` | 文件、依赖、必需评分器或关键身份缺失；补条件，不能跳过后宣称通过 |
| `PASS_AUTO` | 全部必需项成功且处于有效校准覆盖内，均满足规则；只进入下一轮验证，不自动替换生产默认 |

未校准时可报告原始分数和确定性 FAIL，其余质量判断为 REVIEW；必需项缺失仍 BLOCKED。
预测 MOS 不是真实听众 MOS，cosine 乘100也不是音色百分比。校准定义、漏检区间与分组留出见研究报告第8节。

## 人工复核与继续

模板、服务、文件及保存保证只以 [listening-review.md](listening-review.md) 为准。
用户在页面看到已保存后说“继续”，Agent 运行已有入口：

```powershell
.\.venv\Scripts\python.exe scripts\eval\read_listening.py `
  --bundle outputs\gates\<experiment_id>
```

`RATED / UNCERTAIN_FATIGUE / PENDING_USER_LISTENING / BLOCKED` 是**试听文件读取状态**，
不等于上述自检 verdict。Agent 使用已确认选择，保留未评项，疲劳不算平局；
无需用户导出、下载或补交 JSON。每批通常3–5对，优先使用已有标签和真实坏例。

## 实施顺序

| 阶段 | 需要实现的内容 | 完成条件 |
|---|---|---|
| A：统一输入与原始评分 | 拟建 `scripts/eval/run_selfcheck.py`、`src/character_tts/evaluation/selfcheck/`、`configs/evaluations/selfcheck_v1.yaml`；复用 CAM++/ASR，接 UTMOSv2、SCOREQ-NR | 所有项返回统一字段 `status/value/unit/direction/model_revision/weights_sha256/config_sha256/audio_view/windows/error`；输出在 `outputs/selfcheck/<run_id>/`；未校准不放行音质 |
| B：定位与校准 | ASR 编辑序列/时间、关键词、同文本 F0/时长/停顿、跨块身份漂移；导入 hash 可核验的本地好坏例 | 按源片段/文本分组留出；核验真实破损漏检、方向和冲突后声明有限覆盖 |
| C：方法增量 | 在相同资产上评估 WavLM、VoiceMOS 感知头、DS-WED、NISQA 的新增价值 | 改善坏例召回/假警报且成本可接受才进入默认；CAL-MOS 无已核验可部署权重，先不依赖 |
| D：接入迭代 | 固定回归 case 检查 checkpoint，发布候选跑 qualification，保留 REVIEW 与全部失败 | 多维受约束比较；不以单一 loss/cosine/MOS 或可被反复优化的单个评价器决定最优 |

目标 CLI 在阶段A完成后才可运行：
`python scripts/eval/run_selfcheck.py --manifest <manifest.json> --profile quick --out <out_dir>`。
`quick/qualification/diagnostic` 的资产及预算定义保留在研究报告第9节；它们是协议设计，
不自动触发新生成。先复用已有样本；具体运行矩阵在本次实验配置中固定。

新依赖放独立评价环境，评分器串行驻留，下载与留证见 [环境说明](environment.md)。
实施验收逐项使用研究报告第13节，明确未运行项与用户本地材料限制。
