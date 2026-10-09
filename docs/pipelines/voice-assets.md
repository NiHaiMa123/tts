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

## ③ 标准化

统一到交付格式（项目约定 48kHz mono 或 16k 训练域），按音频内容
sha256 寻址存入 `datasets/v1/audio/`。**本项目的池是迁移产物**；
通用标准化工具（重采样/响度/削波检查）待迁移成 `src/character_tts/ingest/`。

## ④ 冻结数据集

`train.jsonl` / `validation.jsonl` / `test.jsonl`，每行
`{"audio": "<repo相对路径>", "fid": "<sha256>", "text": "..."}`，
外加 `manifest.json`（计数/统计）。冻结后不再修改——评测资产改动会破坏
历史可比性。

## ⑤ 注册角色

`configs/characters/<id>.yaml` 绑齐：

| 字段 | 指向 |
|---|---|
| `backend` | 锁定后端（锁暝=voxcpm2） |
| `reference.audio` | `assets/characters/<id>/reference/ref.wav`（zero-shot 克隆参考，附 sha256） |
| `dataset.*` | `data/characters/<id>/datasets/v1/` |
| `evaluation.anchor_texts` | 评测锚点（音频进 datasets/v1/audio/） |
| `adapters` / `postprocess` | 预留接口 |

## 待迁移清单（老 dotstts 里有、本项目还没有）

- `assess_incoming_source.py`：源音频质量评估 + 声纹 embedding 比对
- speaker filtering / ASR 文本校验（可用本项目 `scripts/asr_check.py` 替代部分）
- 标准化构建器（inbox → datasets/v1/audio 内容寻址池）
- `freeze_dataset.py`：jsonl 冻结 + manifest 生成
- 盲听审核页（`apply_*_review.py` 系列 → 本项目已有 listen 页模板可复用）

> 注意：这些迁移时按本项目结构重写（放 `src/character_tts/ingest/`），
> 不复制 dots_tts_lab 代码。
