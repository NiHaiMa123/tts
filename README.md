# character-tts

与底座无关的角色 TTS 平台，不依赖旧 dotstts 项目。VoxCPM2 与 Qwen3-TTS
通过各自 Python 环境中的 JSONL worker 接入；主平台管理角色、数据、生成、评价与试听。

**使用项目读本页；执行任务先读 [PLAN.md](PLAN.md)；专题文档从 [docs/README.md](docs/README.md) 进入。**

## 当前运行配置

以下为 2026-10-13 仓库配置快照，实际运行以 [角色配置](configs/characters/)
和 [后端配置](configs/backends/) 为准；本地 WAV、模型与 adapter 不随 Git 提交。

| 角色 | 绑定后端 | 当前方案 |
|---|---|---|
| 锁暝 | `voxcpm2` | 零样本；LoRA pilot 已关闭，保留既有基线 |
| 符玄 | `voxcpm2` | 零样本，seed 123 |
| 爱弥斯 | `voxcpm2_aimisi` | VoxCPM2 v3 LoRA step150，d8b9 参考，seed 123 |

Qwen3-TTS 零样本后端可用于实验；旧 Qwen LoRA 试验出现破损及停止失控，实际训练实现仍待核对。
用户还试过 Index 与 Fish，其具体版本和本地试验记录待登记，不能视为尚未尝试。

## 日常使用

双击 `启动TTS.vbs`，控制台启动服务器并打开 Edge 的 `http://127.0.0.1:7860`。
把 UTF-8 台本 txt 放进 `inputs/`，页面选择角色后开始生成。每个台本输出一个 WAV
到 `outputs/webui/<角色>/`，旁边的 `.wav.json` 保存参考、seed、参数、分块和耗时。
关掉控制台会停止服务器；只关页面不影响任务。

Vox 长文按配置分句，每块重新使用参考条件；`chunk_chars` 控制上限。
分块与固定间隔仍可能影响停顿、韵律和跨块音色，不能只凭生成成功判定效果。

## 安装与检查

在 Windows 仓库根目录运行：

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe -e ".[metrics,web,dev]"
powershell -File scripts/bootstrap/setup_voxcpm.ps1
# Qwen 实验按需安装：
powershell -File scripts/bootstrap/setup_qwen3_tts.ps1
```

后端环境、ASR/CAM++ 安装与调用、下载策略见 [环境说明](docs/pipelines/environment.md)。
已有环境先核对实际版本，bootstrap 不等于固定了训练实现。

```powershell
.\.venv\Scripts\python.exe -m pytest
```

真实模型相关检查需要本地环境与资产。试听回归的范围和已知平台限制见
[统一试听规范](docs/pipelines/listening-review.md)。

## 新角色与实验试听

1. 按 [语音资产管线](docs/pipelines/voice-assets.md) 入库、审核、冻结数据并选择参考。
2. 在 `configs/characters/<id>.yaml` 登记角色、绑定后端、数据集及独立评测原声；WebUI 自动读取。
3. 先检查零样本与生成链路；涉及微调时按 [LoRA 管线](docs/pipelines/lora-training.md) 验证数据和加载契约。
4. 按 [统一自检](docs/pipelines/self-check.md) 分开评价音质、音色、表达、内容和稳定性。自动评分入口尚待实现。

所有实验试听使用同一模板，通过本地服务保存选择及备注：

```powershell
.\.venv\Scripts\python.exe scripts\ingest\serve_review.py `
  --bundle outputs\gates\<experiment_id> --port 7865 --open-browser
```

页面确认已保存后，用户说“继续”，Agent 直接读取本地文件：

```powershell
.\.venv\Scripts\python.exe scripts\eval\read_listening.py `
  --bundle outputs\gates\<experiment_id>
```

没有导出或额外提交步骤。未评项保持待试听，疲劳保持不确定；`file://` 直开试听页只读。
保存、恢复、旧页面更新与历史评分兼容统一见 [试听规范](docs/pipelines/listening-review.md)。
数据审核和参考音选择的专用页面仍直接写盘，命令见 [资产管线](docs/pipelines/voice-assets.md)。

## 代码与资产位置

| 位置 | 用途 |
|---|---|
| `configs/` | 角色、后端、评测及训练配置；配置文件存在不代表相关任务尚待执行 |
| `src/character_tts/` | registry、worker 管理、入库、诊断、评价、试听与生成 WebUI |
| `workers/` | Vox/Qwen 薄适配；每个后端运行在独立环境 |
| `scripts/ingest/` | 入库、审核、参考音选择和本地页面服务 |
| `scripts/eval/` | gate/历史实验、读取试听记录；`export_*` 仅供 Agent 脱敏归档 |
| `scripts/lora/` | LoRA 数据准备、训练、评测和生效审计 |
| `scripts/bootstrap/` | 后端环境安装与模型下载 |
| `tests/` | 不加载真实模型的单元测试及可选模型集成检查 |
| `data/characters/`、`assets/characters/` | 本地数据集、参考及 adapter；大资产不入 Git |
| `backend_envs/`、`models/`、`hf_cache/` | 本地隔离环境、辅助模型和权重缓存 |
| `outputs/gates/`、`outputs/training/` | 可追溯实验和训练结果；旧记录保持完整 |
| `outputs/_tmp/`、`logs/` | 临时检查产物与日志 |
| `docs/` | 操作规范、研究证据及归档计划；入口为文档导航 |

## 共用约定

- RTX 5080 16GB 上一次只驻留一个大模型，生成期间不切换后端。
- 角色绑定后端；参考提示、训练、校准和最终评测资产明确区分，防止泄漏。
- 冻结集改动建立新版本；保留 raw WAV、hash、实际版本和生成条件，复用旧证据时核对身份。
- 人耳标签与计算分数分开。经过项目校准的自检可筛查覆盖内样本，未校准不宣称音质通过。
- codec 重建与生成质量分别记录；历史 pilot 不构成新训练任务。生产候选的升级遵循当前任务和有效评测证据。
