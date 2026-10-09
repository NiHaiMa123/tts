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

锁暝现状：270 条 / 142MB（中立_neutral 等目录），文件名 = `【标签】台词原文`。

## ③ 入库管线（`src/character_tts/ingest/` + `scripts/ingest/`）

已实现为通用模块，与具体角色无关。四步，全部产物可落 `outputs/_tmp/` 验证：

```bat
:: 1) 评估：逐条 sanity（时长/峰值/RMS/静音比/削波），产出机器可读报告
.venv\Scripts\python.exe scripts\ingest\assess_inbox.py ^
    --inbox data\characters\<id>\inbox --out outputs\_tmp\assess_<id>.json

:: 2) 标准化：统一 48kHz mono PCM16，按内容 sha256 寻址入池 + index.jsonl
.venv\Scripts\python.exe scripts\ingest\standardize_inbox.py ^
    --inbox data\characters\<id>\inbox ^
    --pool data\characters\<id>\datasets\v1\audio ^
    --index data\characters\<id>\datasets\v1\index.jsonl

:: 3) 审核：生成自包含网页包（音频复制进 <out>/audio/），浏览器标 drop，
::    导出 decisions.json
.venv\Scripts\python.exe scripts\ingest\build_review.py ^
    --index data\characters\<id>\datasets\v1\index.jsonl ^
    --audio-root data\characters\<id>\datasets\v1\audio ^
    --out-dir outputs\_tmp\review_<id>

:: 4) 冻结：去重（同音频/同规范化文本）+ 确定性哈希切分 + 复制音频 +
::    manifest.json；--exclude-file 应用审核结果
.venv\Scripts\python.exe scripts\ingest\freeze_dataset.py ^
    --index data\characters\<id>\datasets\v1\index.jsonl ^
    --pool data\characters\<id>\datasets\v1\audio ^
    --dataset-dir data\characters\<id>\datasets\v1 ^
    [--exclude-file decisions.json]
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

锁暝实测：270 inbox → 4 硬拒 + 1 no_text + 2 去重 → 263 条
（207 train / 36 val / 20 test）。

## ④ 注册角色

`configs/characters/<id>.yaml` 绑齐：

| 字段 | 指向 |
|---|---|
| `backend` | 锁定后端（锁暝=voxcpm2） |
| `reference.audio` | `assets/characters/<id>/reference/ref.wav`（zero-shot 克隆参考，附 sha256） |
| `dataset.*` | `data/characters/<id>/datasets/v1/` |
| `evaluation.anchor_texts` | 评测锚点（音频进 datasets/v1/audio/） |
| `adapters` / `postprocess` | 预留接口 |

## 仍未实现的环节（有接口/目录占位）

- **声纹 embedding 比对**（assess 留了 `speaker_check` 钩子，需 encoder 后端）
- **ASR 文本校验**：`scripts/asr_check.py` 已可对 index/冻结集批量转写比对
  CER，但未接入管线默认步骤——人耳审核仍是权威
- **重复图谱 / 盲听细粒度标注**（旧项目 `apply_*_review.py` 那套）：现有
  review 页只支持 keep/drop，够用但简单
