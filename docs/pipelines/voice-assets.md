# 角色语音资产管线（获取 → 入库）

> 本文件负责获取、入库、审核、参考音选择与冻结。当前任务见 [PLAN](../../PLAN.md)；环境见 [environment.md](environment.md)；实验试听交互见 [listening-review.md](listening-review.md)。

从游戏资源到「可注册角色」的完整链路。本项目只负责 **入库之后** 的环节；
解包/提取由 **Ludiglot** 项目完成（`E:\project\Ludiglot`，原理文档在该项目的
`docs/design/audio-system.md`，不在本仓库）。

## 全景

游戏资源由 Ludiglot 解包并转码，人工筛选后进入角色 inbox；本项目再做评估、
标准化、审核、参考音选择与数据冻结，最后注册角色。源台词、来源和划分必须随音频保留。

## ① 解包（Ludiglot，不在本仓库）

《鸣潮》语音是 Wwise `.wem`，打包在 `pakchunk13-WindowsNoEditor.pak`
（`.../Saved/Resources/<版本>/Lang_zh/Base/`），文件名是事件名的
FNV-1a 32 位哈希：

- **FModel**（`Ludiglot/tools/FModelCLI.exe`）：挂 pak + AES key，导出 `.wem`
- **vgmstream**：`.wem` → `.ogg`/`.wav` 转码
- **文本↔音频映射**：`PlotAudio.json` 的 `FileName`/`EventName` →
  `wwise_fnv_hash(name.lower())` → `<hash>.wem`；
  实现见 `ludiglot/core/audio_extract.py`（`find_wem_by_hash` /
  `find_wem_by_event_name`）与 `voice_event_index.py`

产出：一批 `<hash>.ogg` + 台词文本映射表。

## ② 筛选分类（人工 + 工具）

从解包音频中挑出目标角色台词，按情绪标签分类重命名后放入：

```
data/characters/<id>/inbox/<情绪_标签>/【<情绪>】<台词>.wav
```

现有 inbox：

| 角色 | 条数 | 筛选依据 | provenance |
|---|---|---|---|
| suoming 锁暝 | 161 | v1 冻结集回剪 | v1 台词/标签/split |
| shouanren 守岸人 | 262 | shouanren_v2 freeze（256 资产，6 条内容重复） | split |
| fuxuan 符玄 | 273 | fuxuan_v1 freeze（四情绪） | split |
| aimisi 爱弥斯 | 223 | 人工「可以/」目录 | reviewed=可以 |

文件名约定 `【标签】台词原文.wav`；编号导出 `【标签】_<n>.wav` 与
哈希命名 `<hash>.wav` 视为无文本，由 provenance sidecar 或 ASR 补。

## ③ 入库管线（`src/character_tts/ingest/` + `scripts/ingest/`）

已实现为通用模块，与具体角色无关。全部产物可落 `outputs/_tmp/` 验证。

### 0) 导入（可选，Ludiglot 产物）

```bat
:: manifest.jsonl 每行: {"audio":"<stem|relpath>","text":"台词",
::   "label"?, "event"?, "wem_hash"?, "source_wem"?, "tool"?, ...}
.venv\Scripts\python.exe scripts\ingest\import_ludiglot.py ^
    --audio-dir <解码后的音频目录> --manifest manifest.jsonl ^
    --out data\characters\<id>\inbox [--label 中立_neutral]
```

产物：`<inbox>/<label>/<stem>.<ext>` + `<inbox>/provenance.jsonl`
（standardize 自动合并：sidecar 的 text/label 优先于文件名，其余字段
原样进入 index/冻结行的 `provenance`）。

> Ludiglot 侧的事件→wem 绑定由它运行时解析（bnk/wwiser/txtp），manifest
> 是两边约定边界：任何工具只要产出上述 jsonl 即可接入。

### 1) 评估

```bat
:: sanity 报告：时长/峰值/RMS/静音比/削波，机器可读
.venv\Scripts\python.exe scripts\ingest\assess_inbox.py ^
    --inbox data\characters\<id>\inbox --out outputs\_tmp\assess_<id>.json
```

### 1b) 预处理（现有剪切实现与质量门）

**现有 `--cut` 不应作为已验证的音质修复直接接入新训练。**
[代码审查](../reports/tts-quality-review-2026-10-13.md) 已确认：被拒片段可能重新拼回、
删除有声内容后台词未同步、来源/文本元数据未完整传递，以及全段拒绝/VAD失败没有自动隔离。
这些缺陷尚未修复；它们在现有 v3 的触发数量仍待本地审计。

当前质量门：优先保守首尾静音处理；需删除有声内容的样本先隔离，再校对保留段台词。
明确传递来源、权威文本及原划分，审核拒绝或失败条目，不凭 `preprocess` 成功自动冻结入训练。
LoRA 可能放大数据声学问题，但 VAD 不能证明段内磨砂已清除，更不能锁定全部退化原因。

以下保留既有 CLI 和实现说明，供审计、复现或修复验证；输出写镜像目录，原 inbox 不动：

```bat
.venv\Scripts\python.exe scripts\ingest\preprocess_audio.py ^
    --inbox data\characters\<id>\inbox ^
    --out   data\characters\<id>\inbox_clean --cut
```

`--cut` 处理链：70Hz高通 → **FSMN-VAD 切段**（asr env 子进程，
模型首次自动下载到 `models/vad/`）→ **逐段质检丢弃**（非语音段
必然不在段表内；<150ms 孤立段按爆音丢；|x|>0.98 占比>2% 的段按
削波失真丢）→ 保留段之间截断原静音缝(≤300ms)拼接 + 10ms 余弦
淡化 → RMS 归一 -25dBFS。

报告 `preprocess_report.jsonl` 逐文件记：duration_in/out、
cut.segments_in/kept、cut.dropped[{ms,reason}]、removed_s、
all_dropped flag（全段被丢→保留原音频待人工）。`--denoise`
（ZipEnhancer）已弃用：16k 输出抹掉 ≥8kHz 音色细节，且实测
磨砂感并非底噪。

注意：VAD 只切"非语音"边界——段内的呼吸声/口水音切不掉；
数据集本已人工筛过时剪切量会很小（爱弥斯 223 条仅剪 29s，
全是首尾静音缝），此时磨砂另有来源。

只有逐条通过上述质量门后，才把 `--inbox` 换成 `inbox_clean` 继续走 2)~4)，
并先走审核再冻结。数据集另存新版本；沿源片段保留划分，不能因音频 hash 改变就默默重分。
以下是标准化、冻结和审计命令摘要，审核步骤仍按第3节执行：

```bat
.venv\Scripts\python.exe scripts\ingest\standardize_inbox.py ^
    --inbox data\characters\<id>\inbox_clean ^
    --pool data\characters\<id>\datasets\v<N>\audio ^
    --index data\characters\<id>\datasets\v<N>\index.jsonl
.venv\Scripts\python.exe scripts\ingest\freeze_dataset.py ^
    --index data\characters\<id>\datasets\v<N>\index.jsonl ^
    --pool  data\characters\<id>\datasets\v<N>\audio ^
    --dataset-dir data\characters\<id>\datasets\v<N>
.venv\Scripts\python.exe scripts\ingest\audit_dataset.py ^
    --dataset data\characters\<id>\datasets\v<N>
```

### 2) 标准化

```bat
.venv\Scripts\python.exe scripts\ingest\standardize_inbox.py ^
    --inbox data\characters\<id>\inbox ^
    --pool data\characters\<id>\datasets\v1\audio ^
    --index data\characters\<id>\datasets\v1\index.jsonl
```

### 2b) 声纹 + ASR 辅助检查（可选，跑在 ASR env）

```bat
:: CampPlus 声纹 vs 角色参考音（模型首次自动下载到 models\speaker\）
backend_envs\asr_sensevoice\Scripts\python.exe scripts\ingest\speaker_check.py ^
    --index <index.jsonl> --audio-root <pool> ^
    --reference assets\characters\<id>\reference\ref.wav ^
    --out outputs\_tmp\speaker_<id>.json

:: ASR：有文本行算 CER；无文本行产出 hypothesis（可回填 text_asr）
backend_envs\asr_sensevoice\Scripts\python.exe scripts\asr_check.py ^
    --backend sensevoice --index <index.jsonl> --audio-root <pool> ^
    --out outputs\_tmp\asr_<id>.json

:: 合并进 index flags（speaker_cos/speaker_mismatch/asr_cer/asr_mismatch/
:: text_from_asr+text_asr），原地改写
.venv\Scripts\python.exe scripts\ingest\enrich_index.py ^
    --index <index.jsonl> --speaker speaker_<id>.json --asr asr_<id>.json
```

均为**辅助证据**：flag 只提示人耳审核，freeze 不因 flag 自动丢弃。

### 3) 审核（网页，点击即写盘）

```bat
.venv\Scripts\python.exe scripts\ingest\build_review.py ^
    --index <index.jsonl> --audio-root <pool> --out-dir outputs\_tmp\review_<id>

.venv\Scripts\python.exe scripts\ingest\serve_review.py ^
    --bundle outputs\_tmp\review_<id> --port 7865
```

浏览器开 `http://127.0.0.1:7865/`：逐条听、点 `drop`（不入库）和问题 tag
（wrong_speaker/noisy/wrong_text/clipped/emotion_off），
无文本行显示 ASR 建议文本（前缀 `ASR:`）。**每次点击直接写入
`<bundle>/decisions.json`，无需导出**；刷新/重开会自动恢复历史决策。

### 3b) 选参考音（同一套交互）

```bat
.venv\Scripts\python.exe scripts\ingest\build_refpick.py ^
    --index <index.jsonl> --audio-root <pool> --char <id> ^
    --out-dir outputs\_tmp\refpick_<id> [--top-n 40] [--min-cos 0.75]

.venv\Scripts\python.exe scripts\ingest\serve_review.py ^
    --bundle outputs\_tmp\refpick_<id> --port 7866
```

开 `http://127.0.0.1:7866/`：候选按声纹 cos 降序，点「选为参考音」
即写入 `assets/characters/<id>/reference/ref.wav` + `ref_choice.json`，
并自动回填 `configs/characters/<id>.yaml` 的 `reference.sha256/text`。

### 4) 冻结 + 审计

```bat
.venv\Scripts\python.exe scripts\ingest\freeze_dataset.py ^
    --index <index.jsonl> --pool <pool> ^
    --dataset-dir data\characters\<id>\datasets\v1 ^
    [--exclude-file outputs\_tmp\review_<id>\decisions.json]

:: 完整性审计：音频存在/sha 重算/fid 复算/跨 split 泄漏检查，违规 exit 1
.venv\Scripts\python.exe scripts\ingest\audit_dataset.py ^
    --dataset-dir data\characters\<id>\datasets\v1
```


关键约定：

- **fid** = `sha256(norm_text | audio_sha256)`——规范化文本（去标点空白）与
  音频哈希联合，确定且可复现；与旧 dotstts 的 fid 语义不同（旧的是语句 id）
- **切分**：`sha256("split"|fid)` 哈希分桶（默认 val/test 各 10%），重跑稳定，
  新增语句不会打乱已有划分
- **丢弃原因**记入 manifest：`no_text`（文件名无台词）/ `dup_audio` /
  `dup_text` / `reviewed_out` / `missing_audio`
- **硬拒绝**（standardize 阶段）：`too_short`(<0.4s) / `non_finite` / `empty`；
  其余旗标（mostly_silent/too_quiet/clipped/too_long/too_hot）仅警告，由审核决定
- 冻结时音频从 `--pool` 复制进 `dataset/audio/`，逐条校验存在——
  jsonl 不产生悬空引用
- **文本来源**逐行记入 `text_source`：文件名 `【标签】台词` → `filename`；
  导入 sidecar → `provenance`；无文本时 ASR 回填 → `asr`。
  锁暝 inbox 实测文件名多为 `【标签】_<编号>.wav`（无台词），
  187/263 条文本由 ASR 回填——审核页会标注 `ASR:` 前缀
- **审核与参考音页面直接写盘**：分别写 `decisions.json` 和 `ref.wav`+yaml。
  实验试听的共用模板、保存确认与“选完说继续”完整约定以 [统一试听规范](listening-review.md) 为准，不要求手动导出。

锁暝实测（全量真实跑通）：270 inbox → 4 硬拒（超短）→ 266 标准化 →
声纹 266 行（大部分 cos 0.64–0.87，1 条 0.35 疑似异声）+ ASR 266 行 →
1 no_text + 2 去重 → **263 条**（211 train / 32 val / 20 test），
audit 0 违规。

## ④ 注册角色

`configs/characters/<id>.yaml` 绑齐：

| 字段 | 指向 |
|---|---|
| `backend` | 锁定后端（锁暝=voxcpm2） |
| `reference.audio` | `assets/characters/<id>/reference/ref.wav`（zero-shot 克隆参考，附 sha256） |
| `dataset.*` | `data/characters/<id>/datasets/v1/` |
| `evaluation.anchor_texts` | 评测锚点（音频进 datasets/v1/audio/） |
| `adapters` / `postprocess` | 预留接口 |

## 边界与约定

- **ASR 与声纹是辅助证据**：`speaker_mismatch`/`asr_mismatch` flag 仅供
  审核参考，freeze 只认 `reviewed_out`——人耳仍是权威
- **ASR 回填的文本**（`text_source=asr`）未经人工校对；审核页会看到
  `ASR:` 前缀提示，错字应打 `wrong_text` 标签或 drop
- Ludiglot 侧事件→wem 绑定不在本仓库复刻；manifest jsonl 是交接边界
- 冻结集视为不可变——改动会破坏历史评测可比性；改了就递增 `datasets/v<N>`
