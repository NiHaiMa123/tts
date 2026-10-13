# Qwen3-TTS 专项核查：零样本清晰但不像，LoRA 后破损与停止失控

- 核查日期：2026-10-13。
- 项目：`NiHaiMa123/tts`，读取时远端 `main` 为 `9da9af6fbea3cd12d86a8a1d96206fd6b13b5279`。
- 官方代码：`QwenLM/Qwen3-TTS`，核查提交 `022e286b98fbec7e1e916cb940cdf532cd9f488e`。
- 社区补丁样本：`instavar/qwen3-tts-lora-finetuning`，核查提交 `079db4ba844037f970b488dc96b32e5c7490642f`。**尚未确认这是本项目实际使用的补丁。**
- 本次完成源码检查、上游问题与修复讨论核对、标签及隐藏状态的索引验证；没有取得本地训练环境、权重和 WAV，没有运行模型、试听或重新训练。

## 1. 当前判断

**Qwen 仍值得保留。训练后严重破损和停止失控，不能作为“音质与角色习惯必然交换”或“已经到达模型能力边界”的证据。**

两种现象应分别处理：

1. **零样本清晰，但音色不够像**：说明当前条件下声音质量可用，身份还原不足；应检查 Qwen 专用参考音、转写、场景表达和身份条件。它可能包含底座的实际能力限制，但当前试验尚未测清。
2. **LoRA 后破音、扭曲、破损，并且 EOS 失控**：先排查训练与生成的计算契约、残差码本损失、模型导出及加载。当前官方训练实现中确实能找到对应的代码缺陷，不能先把结果归因为一般过拟合或小模型上限。

证据分层：

| 判断 | 证据程度 |
|---|---|
| 当前核查版本的官方训练代码存在四项不一致 | 源码与 Transformers 损失机制已核对，索引检查通过 |
| 上游其他使用者遭遇过 loss 下降、声音损坏、无法停止 | 第一手问题与复现实验报告；不是维护者保证 |
| 本项目实际训练也包含这些问题 | 尚待核对实际脚本及已安装包源码 |
| 修正后本角色一定能同时保持音质、音色及习惯 | 未验证，需小规模重训与独立试听 |

## 2. 项目中能确认的试验事实

`docs/pipelines/lora-training.md` 记录：

- 官方 finetune 加社区 LoRA patch；rank 16、alpha 32，Q/K/V/O 与 MLP。
- 学习率 `2e-6`，10 个 epoch；179 条 v3 数据。
- 可训练参数约 19.2M / 1.95B；loss 从 14.7 降到 10.2。
- epoch 9 在 scale 0.3 和 0.6 下破损、扭曲；scale 1.0 下停止失控，台本输出从约 33 秒变成约 327 秒。

**失败是记录事实；“adapter 学坏了声学分布，非缩放问题”只是当时的归因，尚未排除错误监督、导出与推理失配。**

错误目标也能得到下降的 loss；减小学习率只会减缓按错误目标更新。低 scale 仍然失真，也不能单独证明缩放实现、模型加载和条件配置正确。

实际资产位于被 `.gitignore` 排除的位置：

- `backend_envs/qwen3_tts/official_finetune/`：训练与试验脚本。
- `outputs/training/aimisi_qwen3_lora/`：清单、codec codes、checkpoint。

因此这次对项目训练故障的定位是有明确代码依据的高优先级候选，尚不是对本地运行的最终诊断。

## 3. 官方当前训练代码中的四项关键问题

### 3.1 主 Talker 标签被移动两次

官方 `finetuning/sft_12hz.py` 同时截取输入的前 T−1 项、标签的后 T−1 项，再将标签传给 Talker。Talker 调用 `self.loss_function`。

核查官方依赖的 Transformers 4.57.3：该模型默认使用 `ForCausalLMLoss`，传入 `labels` 时内部还会自动移动标签一次。于是位置 p 本应预测 p+1，却被要求预测 p+2。[S1][S3][S4]

这是训练目标与自回归生成方式不一致。它可影响内容、语速和 EOS；下降的训练 loss 不再说明正常生成会更好。

**修正原则**：标签只移动一次。可以让模型接收完整、未移动的标签，由标准损失处理；也可以保留截断输入，使用手工交叉熵处理已移动的目标。不能混用两种移动方式。

### 3.2 子模型使用了已看见答案的隐藏状态

每个音频帧的输入包含本帧 codec 0 和其余残差码本的 embedding。原脚本按 `codec_mask[:, :-1]` 取当前帧位置的 Talker 隐藏状态，供子模型预测同一帧的残差码本。[S1][S2]

该状态已经看过本帧的完整 codec 信息；实际生成子模型得到的是上一位置的 `past_hidden` 和刚生成的 codec 0。训练因此泄漏了答案，推理时又没有这些信息。

**修正原则**：帧 p 的子模型使用 h[p−1]。对于长度 T−1 的 Talker 输出，可按 `codec_mask[:, 1:]` 选择前一位置；其他张量长度下需相应构造前移掩码，不能直接复制切片而忽略形状。

### 3.3 子模型的 15 个残差码本标签又被自动移动

`forward_sub_talker_finetune` 向 `code_predictor.forward_finetune` 传入 codec 1…15 的目标；其 15 组 logits 已经分别对应这 15 个目标。但 `forward_finetune` 再使用默认因果损失，会把目标移动成 codec 2…15、忽略项。[S1][S3][S5]

结果是第一组输出学错码本，最后一组输出没有正确监督。残差码本直接参与音频重建，这项问题尤其值得检查声音颗粒、破损及细节损伤。

**修正原则**：对已经对齐的码本输出使用不再移动标签的交叉熵；或在确实兼容的损失版本中用 `shift_labels` 明确传入对齐目标。需要同时核对实际包中的函数，而不只改外围训练脚本。

### 3.4 训练漏掉了 text_projection

官方训练直接使用原始 text embedding；正常生成先经过 Talker 的 `text_projection`。[S3][S5]

0.6B 可能因维度不一致报错；1.7B 即使形状允许相加，训练仍可能使用与推理不同的文本特征空间。**“能开始训练、loss 正常”不等于输入一致。**

**修正原则**：训练文本 embedding 应经过与生成一致的投影，再按训练掩码参与输入。

### 3.5 本次验证的范围

用小型数组验证了实际索引关系，没有模拟模型音质：

| 项目 | 应有对应关系 | 原实现对应关系 |
|---|---|---|
| 主模型位置 0 的标签 | 位置 1 | 位置 2 |
| 子模型目标帧 3、4、5 的条件 | 隐藏状态 2、3、4 | 隐藏状态 3、4、5 |
| 残差输出目标 | codec 1…15 | codec 2…15，最后一项忽略 |

投影遗漏则由训练与生成源码直接对照确认。上述检查证明被核查实现存在不一致，不能替代本项目实际运行源码核对。

## 4. 上游证据：相近失败存在，而且有实现修复后的改善

- issue #371 于 2026-09-16 报告三项对齐错误，并提供真实 batch 的交叉熵比较；其模型 loss 正常下降，但生成不能正常停止。[S1]
- PR #278 提出标签对齐和文本投影修复。**截至本次核查仍为 Open，尚未合并；不能认为升级最新官方 main 就自动解决。**[S3]
- 2026-09-29 的一名 1.7B 使用者报告：原配方留出 loss 从 14.0 降到 5.6，但 5 秒台词生成 13–201 秒；修正后的独立训练中，44 条测试均正常结束。配置和数据量与本项目不同，不能把提升幅度直接迁移。[S3]
- 0.6B 的另一项复现报告修正后音色及频段表现改善，但小数据训练仍有内容错误与过拟合。[S3]

这些属于第一手用户报告，尚无本项目的可比复现。它们支持优先检查实现，**不支持“修了就必定完美克隆”**。

PR #278 的补丁文本还出现过在普通 `train()` 函数中误用 `self.model` 的问题；讨论明确指出会触发 NameError。采用修复思路时必须结合实际版本审查与验证，不能直接把整份 PR 当作已验证配方。

## 5. 社区 LoRA patch 可能只修了部分问题

核查 `instavar/qwen3-tts-lora-finetuning` 上述提交的 `patches/0001-qwen3-tts-lora.patch`：[S6]

| 检查项 | 其 LoRA 训练函数的状态 |
|---|---|
| 主模型双移动 | 使用手工交叉熵，已避开 |
| text_projection | 已补上 |
| 前一位置的隐藏状态 | 使用 `codec_mask[:, 1:]`，已调整 |
| 子模型残差标签移动 | 仍调用上游 `forward_sub_talker_finetune`；该 patch 未修改上游模型文件中的损失 |

**若与本次核查的未修复上游模型代码组合，最后一项仍然存在。**如果安装的包另有修复，则必须检查其实际源码，不能仅从这个 patch 推断。

这个社区补丁的 rank、alpha、目标模块和学习率默认值与项目记录相近，值得核对来源；相近参数不构成“本项目肯定用了它”的证明。

## 6. 还要核对训练、导出和推理是否走同一条路径

### 6.1 项目标准 worker 目前没有接入这次 LoRA

`workers/qwen3_tts_worker.py` 声明 `lora: False`，调用的是 Base 模型的 `generate_voice_clone`。提交的 `scripts/eval/gen_qwen3_train_txt.py` 也是零样本克隆脚本。

因此 Git 中这两处代码不能说明旧 LoRA 试验的加载、合并与生成方式正确；实际调用只能从本地试验脚本确认。

官方 SFT 导出会将模型类型改成 `custom_voice`，写入说话人 token 的 embedding，并移除 speaker encoder。它与“Base 加 adapter 后继续参考克隆”的接口和身份条件不同。[S3][S7]

应核对 `tts_model_type`、说话人名称与 token ID、embedding 文件、加载完整性、生成 API，以及合并前后权重。不能把已改成 CustomVoice 的 checkpoint 直接视为原 Base 克隆路径的可替换权重。

### 6.2 scale 0 不是无条件的零样本复现

只有生成模式、说话人条件和非 LoRA 参数一致时，关闭 adapter 才适合与原模型比较。如果加载过程写入了新的说话人 embedding 或改成 CustomVoice，scale 0 也可能与原零样本不同。

应先验证未经训练、标准零增量初始化的 adapter 在**同一生成路径**中与原权重一致；再验证保存与重载、合并与不合并后的有效更新一致。初始 LoRA dropout 等随机因素需在 eval 模式下排除。

通常的 LoRA 更新包含 alpha/r；本次 r=16、alpha=32 的比例为 2。外部 scale 的解释仍要看具体实现，不能仅用 0.3 或 0.6 这个数判断适配一定很弱。

### 6.3 包版本和脚本版本需分别留证

项目固定了 Qwen 模型及 tokenizer revision；bootstrap 中 `qwen-tts` 等安装项没有在脚本中固定具体版本。固定模型权重不等于固定损失函数实现。

最小核查材料应包括：

- 实际训练脚本、dataset/collate、LoRA 导出及推理脚本。
- 社区补丁来源、提交或 release，以及本地修改 diff。
- 实际 import 的 `qwen_tts` 路径；其中 Talker、CodePredictor 的相关函数源码及 hash。
- `qwen-tts`、`transformers`、`peft`、`torch`、`accelerate` 版本。
- 运行参数、`adapter_config.json`、导出 `config.json`、实际可训练模块名称。
- 清单条数、总时长、音频采样率、reference 选择规则、codec 数量及 EOS 处理。

第一步不需要上传大权重；这组文本材料即可判断四项错误是否落在本项目运行路径中。音频与 checkpoint 留在本地用于后续试听。

## 7. 建议的优化顺序

### P0：先检查并修正计算契约

1. 核对实际包与脚本是否包含四项问题，尤其子模型内部损失。
2. 记录主模型、残差码本和 EOS 的独立损失；检查每组输出的目标索引与有效监督数。
3. 对同一 batch 做训练与生成单步对齐检查，确认子模型条件相同。
4. 验证零更新 adapter、保存重载和合并一致性，明确 Base/CustomVoice 身份条件。

可以比较未训练基模在正确与错误标签对齐下的 teacher-forcing 交叉熵，辅助定位。**不应要求模型对自己采样的音频 loss 必须接近零**：采样不一定选择最大概率 token，存在输出熵。索引和计算契约是主要验证依据。

如果旧 checkpoint 由错误目标训练，推理端修复并不能保证恢复；新试验应从干净的原始底座开始。

### P1：修正后做小规模、可停止的重训

- 固定正确转写、干净 reference 和同一底座/tokenizer。
- 从现有清单中选有代表性的完整语句，不用极少的单类台词代替整个任务。
- 先验证前几十个 optimizer step，密集保存早期 checkpoint；逐步扩展，避免再次直接跑满 10 epoch。
- 保留未参与调参的原声与新文本，分别听内容、清澈度、音色、习惯和停止。
- 新增明显破损、漏字或时长失控时停训；降低后的训练 loss 不能覆盖这些失败。

学习率、rank 和训练轮数应在实现正确之后调整。现有 `2e-6` 已低于官方原 SFT 默认值；这不是正确性证明，也不是下一轮一定还要降低的依据。

### P2：用 Qwen 专用参考改善零样本音色

项目已经设置 `x_vector_only_mode: false`，并传入参考音及转写；不能把“开启完整参考”当作尚未执行的新修复。

提交的零样本长文脚本使用 d8b9，和 Vox 实验所用参考相同。参考被另一套声纹评分或 Vox 试听选中，不保证对 Qwen 最好。

先做有限对照：3–5 条干净、同说话人且表达匹配的完整 reference，逐字校对转写，使用相同少量新台词和固定候选预算。分别评价共鸣、厚薄、气声、鼻音、尾音和节奏，不只比较声纹 cosine。

参考长度应遵守实际版本支持及本地预算；更长不必然更像。长文另行检查跨块身份，不把切成 100 字加固定间隔的效果当作完整长篇生成能力。

### P3：确有退化再做模块消融或能力保持

Qwen 这条路径主要是 Talker 与 CodePredictor，不能套用 Vox 的 DiT/LM 名称和结论。

可比较 Talker 内明确指定的模块适配与 Talker+CodePredictor 联合适配；同步记录真正训练的参数。泛用 `q_proj`、`gate_proj` 等名称可能覆盖两部分，不等于“只训练主模型”。

冻结子模型也不自动消除错误残差损失：损失仍可能通过条件隐藏状态反传到主模型。必须先纠正监督，再谈冻结和分模块强度。

若实现、数据、reference、早期权重均验证后仍发生较小的质感与习惯取舍，再研究数据覆盖、保持性目标和底座边界。此时才能给出“本底座、这批数据与当前预算下”的限制，而不是宣布开源 TTS 的整体上限。

## 8. 对前两份报告的补充

- 用户已试过 Index、Vox、Fish、Qwen。优先利用已有资产与结果，不将 Fish 家族重复列为必需新增；具体 Fish 版本尚未确认。
- 前两份报告以 VoxCPM 的音质与习惯取舍为主，其双参考、早期 checkpoint 等建议不能替代本次 Qwen 训练对齐检查。
- Qwen 的严重失真应优先按本报告 P0 核查。旧记录中的“学坏了声学分布”保留为当时解释，不能作为已经排除实现故障的结论。
- 本次只新增核查报告并纠正调研优先级，没有修改运行代码，也没有声称本地模型已修好。

## 9. 来源

| 编号 | 原始来源 | 用途与边界 |
|---|---|---|
| S1 | [Qwen issue #371](https://github.com/QwenLM/Qwen3-TTS/issues/371) | 三项对齐问题、真实 batch 与停止失败；第一手用户报告 |
| S2 | [Qwen issue #337](https://github.com/QwenLM/Qwen3-TTS/issues/337) | 子模型隐藏状态与生成契约 |
| S3 | [Qwen PR #278](https://github.com/QwenLM/Qwen3-TTS/pull/278) | 修复思路、开放状态、不同用户的复现与补丁自身问题 |
| S4 | [Transformers 4.57.3 loss_utils](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/loss/loss_utils.py)；[modeling_utils](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/modeling_utils.py) | 自动标签移动与默认损失分派，已读取源码 |
| S5 | [官方训练脚本](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/sft_12hz.py)；[模型实现](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py)；[dataset](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/dataset.py) | 本次克隆并直接核查的固定版本 |
| S6 | [社区 LoRA patch](https://github.com/instavar/qwen3-tts-lora-finetuning/blob/079db4ba844037f970b488dc96b32e5c7490642f/patches/0001-qwen3-tts-lora.patch) | 本次核查的部分修复样本，未确认项目采用 |
| S7 | [官方微调文档](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/README.md) | ref_audio 数据及 CustomVoice 推理契约 |

项目自身证据：`docs/pipelines/lora-training.md`、`workers/qwen3_tts_worker.py`、`configs/backends/qwen3_tts.yaml`、`scripts/eval/gen_qwen3_train_txt.py`、`scripts/bootstrap/setup_qwen3_tts.ps1`、`.gitignore`，读取基线见文首。

[S1]: https://github.com/QwenLM/Qwen3-TTS/issues/371
[S2]: https://github.com/QwenLM/Qwen3-TTS/issues/337
[S3]: https://github.com/QwenLM/Qwen3-TTS/pull/278
[S4]: https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/loss/loss_utils.py
[S5]: https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py
[S6]: https://github.com/instavar/qwen3-tts-lora-finetuning/blob/079db4ba844037f970b488dc96b32e5c7490642f/patches/0001-qwen3-tts-lora.patch
[S7]: https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/README.md
