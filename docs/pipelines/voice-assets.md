# 角色语音资产管线（获取 → 入库）

从游戏资源到「可注册角色」的完整链路。本项目只负责 **入库之后** 的环节；
解包/提取由 **Ludiglot** 项目完成（`E:\project\Ludiglot`，原理文档见
`docs/design/audio-system.md`）。

## 全景

```
游戏 pak ──①解包──> wem (hash 命名) ──②转码──> wav/ogg
    ──③筛选分类──> data/characters/<id>/inbox/<情绪>/*.wav
    ──④标准化──> datasets/v1/audio/<sha>.wav（内容寻址）
    ──⑤冻结──> datasets/v1/{train,validation,test}.jsonl
    ──⑥注册──> configs/characters/<id>.yaml
```

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

锁暝现状：161 条（中立_neutral），已按 v1 数据集回剪——每条有
`provenance.jsonl` 记录 v1 台词/标签/split + 源 sha256。文件名约定
`【标签】台词原文.wav`；编号导出 `【标签】_<n>.wav` 与哈希命名
`<hash>.wav` 视为无文本，由 provenance sidecar 或 ASR 补。

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

### 3) 审核（自包含网页包）

```bat
.venv\Scripts\python.exe scripts\ingest\build_review.py ^
    --index <index.jsonl> --audio-root <pool> --out-dir outputs\_tmp\review_<id>
```

打开 `review.html`：逐条听、标 `drop`（不入库）和问题 tag
（wrong_speaker/noisy/wrong_text/clipped/emotion_off），
无文本行显示 ASR 建议文本（前缀 `ASR:`）。导出 `decisions.json`。

### 4) 冻结 + 审计

```bat
.venv\Scripts\python.exe scripts\ingest\freeze_dataset.py ^
    --index <index.jsonl> --pool <pool> ^
    --dataset-dir data\characters\<id>\datasets\v1 ^
    [--exclude-file decisions.json]

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
- 冻结时音频从 `--pool` 复制进 `dataset/audio/`，逐条校验存在——
  jsonl 不产生悬空引用

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
