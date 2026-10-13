# TTS 统一自检流程：音色、音质、韵律与内容的计算评分

- 调研日期：2026-10-13，中国标准时间。
- 项目：`NiHaiMa123/tts`；读取基线 `ccad07752f7b260e7637b718bbc622d7b5fe5c52`。
- 目标：让 Index、Vox、Fish、Qwen 及其 adapter 共用评价输入、指标定义、校准和结果格式，发现音色不够像、磨砂、破损、节奏异常和停止失控。
- 相关报告：[项目音质审查](tts-quality-review-2026-10-13.md)、[开源能力边界](open-source-tts-capability-boundaries-2026-10-13.md)、[Qwen 训练破损专项](qwen3-tts-training-distortion-audit-2026-10-13.md)。
- 本次交付：研究及可执行的实施规范。已读取论文、官方代码和项目评价代码；**没有运行音频评分模型、没有计算本地 WAV 的真实分数。** 同日已补充实现统一试听模板、本地保存和 Agent 读取流程，见 [统一试听页](../pipelines/listening-review.md)。下文拟新增的自动评分程序和接口尚不存在，需要 Agent 实施。

## 1. 要建立什么样的自检

**可以自动计算分数并筛查失败，但不能把声纹 cosine 或一个预测 MOS 当作完整人耳评价。**

统一输出五个维度：

| 维度 | 回答的问题 | 默认计算证据 |
|---|---|---|
| 声音身份 | 听起来是不是目标说话人 | 多参考 CAM++；扩展 WavLM 说话人模型与感知相似度模型 |
| 声学质量 | 是否新增磨砂、破损、噪声、压缩感 | UTMOSv2、SCOREQ-synthetic；局部诊断及可选 NISQA |
| 韵律与习惯 | 重音、停顿、语速、尾音及气息是否合理 | 同文本对齐后的音高/时长/能量；无同文本原声时只做分布和异常分析 |
| 内容 | 是否漏字、重复、错读、截断 | 中文 CER、编辑位置、第二 ASR 复核、关键字检查 |
| 稳定与有效性 | 是否正常生成、结束，长文是否漂移 | 文件检查、生成元数据、时长和逐块/逐窗结果 |

第一版保存**原始分数、相对固定基线的变化、局部时间位置、计算状态**；经项目人工标签校准后再增加“自动通过/失败”及可解释的 0–100 分。

一个候选声音可能“更像但更破损”，因此不设置一个允许音色收益抵消破损的总分。硬性失败先判，其余维度分别列出；通过候选再比较收益。

## 2. 近期学术证据与方法选择

### 2.1 2026 年研究对流程的直接影响

| 研究 | 方法或结果 | 对本项目的用途 |
|---|---|---|
| Rethinking Automated Voice Similarity，2026-09-27，arXiv:2609.33999 | 说话人验证 EER 不可靠地代表人耳音色相似；训练目标与 embedding 几何更关键 | 保留声纹，但用角色原声与人工判断校准；不能因某模型验证 EER 更低就认定更适合角色音色评价。[S1] |
| VoiceMOS Challenge 2026，2026-09，arXiv:2609.13792 | 分别预测自然度、情绪、说话人及口音相似；公开官方 baseline | 其中声纹加感知评分头可作为新的音色评价候选；训练域主要是有口音的英语，中文角色须独立验证。[S2][S3] |
| The Limits of Reference-Free Speech Quality Metrics，2026，arXiv:2609.13150 | 单一无参考评分在两条均无明显损伤的现代 TTS 上难以稳定预测偏好；校准组合更好，优化单个分数可被利用 | 采用多个互补信号和独立验证，不以单一 MOS 排名或直接作为唯一训练奖励。[S4] |
| Investigating Human-Model Discrepancies，Interspeech 2026，arXiv:2606.19951 | 多个 MOS 模型对声学损伤敏感，却可能漏掉韵律错误和说话速度、F0 变化 | 韵律必须单独计算，MOS 高不能覆盖“口型不对、语调顶上去、变化机械”等失败。[S5] |
| DS-WED / ProsodyEval，ICASSP 2026；预印本 2025-09 | 离散语音 token 的加权编辑距离，用于同文本多次生成的韵律多样性 | 可检测“每次都一样”或变化量异常；**多样性不是目标说话人的习惯相似度，也不是越大越自然。**[S6][S7] |
| CAL-MOS，2026-09-14，arXiv:2609.14956 | 用层间 adapter 校准多层语音基础模型特征；最佳层依模型及数据而变 | 后续评分器候选。本次未核实与论文一致的可部署权重，第一版不依赖它。[S8] |

以上包含近期预印本和已发表会议工作，不能统一称为“已成熟的最优标准”。对部分论文本次取得的是原始摘要而非完整正文，未复现其实验；不照搬跨域相关系数为本项目准确率。

### 2.2 可部署评分器与具体用法

| 方法 | 输入与输出 | 版本及采用理由 | 不能推出什么 |
|---|---|---|---|
| CAM++ 中文声纹 | 16k 单声道；单位向量 cosine，越高身份越接近 | 项目已用 `iic/speech_campplus_sv_zh-cn_16k-common`；复用资产并固定实际 revision/hash。[S9] | 0.9 不表示“90% 像”；不能衡量质感、漏字和完整演技 |
| WavLM 说话人模型 | 16k 音频，需有说话人 pooling/head 的 checkpoint；cosine | 用 Microsoft UniSpeech 的 speaker-verification 实现作为独立复核，第一阶段可后置。[S10] | 原始 WavLM 特征平均不自动等于正式说话人 embedding |
| VoiceMOS 2026 Track 3 感知评分 | 音频对；ECAPA 编码器、projection 和相似度回归头，训练量表 1–5 | 官方仓库有说话人评分 checkpoint；本次确认两个约 89MB 的权重文件为实际文件而非 LFS 指针。属于实验性扩展。[S3] | 英语域结果不保证适用于中文、气声角色或游戏表演 |
| UTMOSv2 | 无参考音频；预测自然度 MOS，通常按 1–5 量表训练，越高越好 | SLT 2024 的成熟公开实现；用于发现质量退化，不能包装为 2026 新模型。[S11] | 预测 MOS 不是实际听众 MOS，也不能证明习惯正确 |
| SCOREQ | `data_domain=synthetic, mode=nr` 输出预测质量；越高越好 | NeurIPS 2024；官方目前支持 ONNX 推理，适合隔离部署。[S12] | 与 UTMOS 分数不能直接等权相加；自然/合成域的不同权重不能直接互减 |
| SCOREQ reference mode | 输出 embedding 距离，**越低越好** | 可用固定干净的非匹配参考做质量辅助；先验证中文角色有效性。[S12] | 不是声纹相似度，也不是 1–5 MOS；不能把距离越大当作质量越高 |
| NISQA-TTS | `nisqa_tts.tar`，预测合成语音自然度 | 可选的另一套质量证据，较早的方法。[S13] | 没有四个细分损伤头 |
| NISQA v2 多维模型 | `nisqa.tar`，质量及 Noisiness / Coloration / Discontinuity / Loudness | 对应噪声、音染、断续、响度的辅助证据；原训练重点为通信失真，在本项目校准后才用于判定。[S13] | 不能把这四个分数写成“已证明的 TTS 磨砂/压嗓检测器” |
| Qwen3-ASR / SenseVoice | 转写；计算中文 CER 与插入、删除、替换 | 项目已有环境；Qwen3-ForcedAligner 可提供逐字/词时间戳。[S14] | 对齐器会把给定文本往音频上对齐，因此不能用对齐成功证明文字确实说对 |
| Praat/同类 F0、CPP/CPPS、HNR | 音高、谐波及倒谱特征，配合音素/浊音掩码 | 作为局部变化定位；参数与版本固定。[S15] | 不套用临床声嗓阈值；自然气声、喉化、情绪发声不是自动应剔除的缺陷 |

音频预测分数偶尔可能超出训练量表。保留原始值并记录越界，不能偷偷截断后作为正常结果。使用评测模型的公开预训练权重不等于已完成本项目校准。

### 2.3 本次核对过的实现版本

| 实现 | 已核查提交 |
|---|---|
| UTMOSv2 | `cc2700db57bb83ee13dc31ebe1b868c254e15d09` |
| SCOREQ | `0cb0b168d0f7ec1419475d1e7b7ea699d8cd599e` |
| VoiceMOS 2026 baselines | `f9482fbb81d0429e090b491dd7aa7870a4b040f0` |

实现时还需记录实际权重文件 SHA256、依赖版本和有效推理配置。上述提交用于复核，不代表 Windows / RTX 5080 已通过运行测试。

## 3. 项目已有基础与需要补齐的地方

| 现有代码 | 可复用内容 | 需要补齐 |
|---|---|---|
| `src/character_tts/diagnostics/metrics.py` | RMS、LUFS、近似 true peak、共用频段能量、谱统计和变化量 | 它们是描述值，不是音质评分器；增加局部定位和明确的异常判定 |
| `scripts/ingest/speaker_check.py` | CAM++、多参考均值及 cosine | 接入生成结果；固定独立真实参考，保存模型 hash 与窗口分数 |
| `scripts/asr_check.py` | SenseVoice、Whisper、Qwen3-ASR 与 CER | 输出编辑路径、错误词位置、逐字时间戳和关键字；规范化留证 |
| `src/character_tts/evaluation/` | manifest、试听和已有实验报告 | 共用评分入口、统一状态、校准版本和失败原因 |
| 既有盲听结果 | clarity、grit、likeness、naturalness 等维度 | 只使用确实有对应 WAV/hash 的记录；疲劳或不确定不能转成正常标签 |

两个实现细节必须处理：

1. 当前 `voiced_spectral_stats` 的掩码实际主要按相对 RMS 选帧，不是声带浊音检测。不能把它当作纯元音/浊音分析；建议保留旧值为 RMS-active 描述，新 CPP/HNR/F0 分析使用明确的 VAD 与浊音掩码。
2. 声纹脚本支持 self-centroid。数据入库探索可以使用，但**评价目标人物相似度不能把待评价生成物加入自己的参考中心**；否则一组同样跑偏的声音也可能获得高分。

现有 `_norm` 主要删除标签、标点等。中文数字、英文大小写、繁简、单位、多音字需有明确的规范化规则；同时保留原始文本、规范化文本与规则版本。不能简单删除数字或把不同关键词统一，以掩盖错读。

## 4. 固定评价资产，防止指标随实验漂移

### 4.1 每个角色的四组独立资产

| 资产 | 建议起步规模 | 作用 |
|---|---:|---|
| speaker_gallery | 5–8 条干净原声，覆盖适度表达变化 | 目标身份参考；不包含待评分生成物 |
| calibration | 24–40 对已有好坏对照，分小批标注 | 选择有效指标、暂定阈值和错误类型；初期仅支持筛查 |
| frozen_eval | 20 条目标文本，两个固定 seed | 最终比较；其中约 8 条有同文本留出原声，另 12 条为新文本 |
| regression_badcases | 已确认的磨砂、破损、漏字、拖长、声线漂移 | 每次评价代码或模型变更必跑，检查故障是否被漏掉 |

规模是本项目的工程起点，不是论文保证足够的样本量。最初可用 12 条文本、一个 seed 做快速筛查；正式比较扩到固定集合。

**划分规则**：

- 生成 prompt/reference 与对应目标原声分开；不能把要复刻的留出台词原声同时作为该台词的克隆提示。
- 训练、reference、calibration、最终评价明确分离；按源片段分组，避免同一台词裁剪版本跨集合。
- speaker_gallery 的素材不因某个候选表现差就临时换掉。
- 最终评价样本不参与选择阈值或参数；反复用来调参后应转成开发集，另补真正留出集合。
- 原声不够干净时先标记限制。不能用有损坏的素材定一个偏低的质量标准，再宣布模型通过。

### 4.2 要覆盖的场景

中性短句、普通叙述、问句、停顿/尾音、轻声/气声、情绪变化、姓名/数字/单位，以及至少一段生产实际长度的长文。不同表达要分别统计，防止安静中性句掩盖轻声或情绪句失败。

同一个 seed 在不同底座上不代表同一种噪声。公平条件是文本与参考资产固定、每个方案候选预算一致；模型自身参数遵循其有效推理方式，不能强行让不同模型共享同一个 CFG 数值。

## 5. 每次修改后的统一运行步骤

| 步骤 | 输入与操作 | 输出 / 失败处理 |
|---|---|---|
| 0. 留证 | 记录 git/model/tokenizer/adapter/数据清单 hash、真实生成参数、参考、预处理、seed、重试 | `run.json`；关键身份缺失则 BLOCKED |
| 1. 计算契约 | 对新训练路线检查训练/推理对齐、零更新 adapter、保存加载、合并等 | `implementation_checks.json`；失败则停止升级，不用音频分数覆盖 |
| 2. 生成或收集 | 基线与候选共用生产 worker 路径；优先复用身份与输入完全一致的缓存 | 原始 WAV、生成 sidecar、manifest；每次重试都保留 |
| 3. 基础健康 | 解码、长度、NaN/Inf、全零、声道、采样率、PCM 饱和/平台段、停止和预算 | 明确的硬故障 FAIL；疑似削波/拖长输出 REVIEW 与时间点 |
| 4. 内容 | 主 ASR、CER、编辑序列、关键词；疑似错误用第二 ASR | `content.json`；按已校准规则判断，识别器分歧 REVIEW |
| 5. 身份 | gallery 与窗口 CAM++；按 profile 运行第二编码器/感知评分 | `speaker.json`；短音频或失效 embedding 不填零装作真实分数 |
| 6. 音质 | UTMOSv2、SCOREQ；固定窗口及可选 NISQA/局部特征 | `quality.json`；整体与局部结果同时保留 |
| 7. 韵律 | 内容有效后对齐；同文本分析曲线，否则分析习惯分布/异常 | `prosody.json`；没有原声时不输出虚假的曲线相似分数 |
| 8. 校准及汇总 | 匹配校准 profile，比较固定基线与置信区间 | `verdict.json`、逐条 CSV 与 MD/HTML 报告 |
| 9. 有限复核 | 呈现最差局部、指标冲突与域外样本的小型 A/B | 新标签作为下一版开发证据，不回头污染当前最终测试 |

基础错误对所有底座通用。训练契约检查则是每条后端的特定前置项；Qwen 的四项训练不一致须先按专项报告核对，不能凭 UTMOS 高分宣布训练实现正确。

### 5.1 人工复核网页与 Agent 继续

所有实验调用同一试听模板，复用播放、评分、备注、盲听顺序与疲劳交互。选择及备注经本地服务直接写入当前实验的评分 JSON，无导出或额外提交步骤。

用户看到已保存后，在对话中说“继续”；Agent 用 `scripts/eval/read_listening.py --bundle <当前实验目录>` 读取真实文件，核对实验、音频 SHA 和已评/未评状态，再继续自检或既定迭代。未评价项保持 pending，疲劳不算平局；写盘失败、旧实验或变化的音频不能作当前人耳证据。

旧页面仅需重新构建 HTML，不为更新模板重新生成 WAV。实施命令及保存协议见 [统一试听页](../pipelines/listening-review.md)。

## 6. 各维度如何计算

### 6.1 声音身份：多参考、独立中心、长文漂移

将每条真实参考的 embedding 单位化，形成固定中心：

$$
e_i=\frac{f(r_i)}{\|f(r_i)\|},\quad c=\frac{\sum_i e_i}{\|\sum_i e_i\|},\quad SIM(x)=\frac{f(x)}{\|f(x)\|}\cdot c.
$$

至少同时报告：

- `sim_centroid`：中心 cosine。
- `sim_refs_median`：对各固定原声的相似度中位数。
- `sim_refs_min` 或低分位：是否只与某一条参考像。
- `sim_window_q10` 与窗口明细：长文的身份漂移；窗口必须有足够有效语音。
- 相对同文本、同参考、同候选预算基线的变化。

对 gallery 的真实原声做 leave-one-out：评分某条原声时，中心不能包括它自身。这提供同说话人在本条件下的自然波动；加入少量其他说话人只作负面对照，不能将二分类的验证阈值直接等同于人耳“足够像”。

WavLM 与 CAM++ 原始 cosine 各自展示和校准，不直接平均。若两个模型对候选变化方向冲突，则增加 REVIEW 线索；两者一致也不自动证明声音质感相同。

VoiceMOS 感知评分头可用于试验，但要在中文好坏对照上证明优于或补充 CAM++，再纳入必需项。不应随意对本角色少量参考做 PCA 后，把低维 cosine 宣称为复现了新论文的感知模型。

### 6.2 音质：两个主评分器与局部诊断

默认同时运行：

1. **UTMOSv2**：固定 config、data-domain ID、fold、随机 seed 和重复次数。官方 quick prediction 与全 fold/重复推理有区别；不能在一次实验用 quick、下一次用 ensemble 后直接比较。[S11]
2. **SCOREQ synthetic/no-reference**：所有基线与候选使用同一个 synthetic checkpoint。不要对原声用 natural 模型、对生成用 synthetic 模型，再将两个分数互减。[S12]

原声分数作为上下文与校准控制；模型给原声低于合成音的预测值，不证明合成音超过真人。质量保持主要比较**已确认可接受的固定生成基线**与候选，同时检查绝对可接受性。

长音频使用固定连续窗口：例如 10 秒窗口、5 秒步长，保留尾部的完整短句窗口；SCOREQ 官方建议较长音频取 10–15 秒段。最终输出整体/窗口中位数、低分位与最差片段，不能只报均值。

不把过短片段强行补静音后当成标准评分；短句按模型有效输入与短句校准 profile 处理。评分器不支持的长度记 `unsupported_length`，由健康/内容等结果补充，不能将缺失分数设成 5。

**磨砂、破损的辅助定位**：

- NISQA 多维变化：记录音染、噪声、断续维度，先确认与已知坏例的关系。
- 在对齐的元音/浊音区域比较 CPP/CPPS、HNR、谱倾斜和频段能量；在辅音处另看爆裂和噪声。
- 保存波形平台、突变、局部谱差的时间位置；差分或高频能量只作定位，不单独判定音质。
- 气声、喉化和情绪声与正常角色发声比较，避免把自然表达删除后获得“更干净”的假改善。

CPP/HNR、频段能量、抖动等不保证数值越大或越小越好；标注“诊断特征”，不写成未经校准的质量分数。

### 6.3 内容：中文 CER 与关键错误

$$CER=(S+D+I)/N$$

其中 S、D、I 是字符替换、删除、插入；保留具体字符和位置。CER 可以大于 1，不应截成 100% 后丢掉严重重复的信息。

- 同时输出原始和规范化后的 CER；空参考文本不得默认为正常。
- 名字、数字、单位、否定词等独立标记；整句 CER 低也可能漏掉重要词。
- 有同文本原声时，给原声也运行 ASR 作为识别器误差控制；不直接减去原声 CER 当作“修正后的真实错误率”。
- 第二 ASR 只处理疑似错读、重复或识别器分歧；必要时给出局部试听，不能将识别器共识视为绝对真值。
- 先 ASR 再强制对齐。强制对齐器可能在漏字音频上仍给出时间戳，时间戳有效性需检查。[S14]

### 6.4 韵律与习惯：必须看曲线和时间结构

**有同文本留出原声时**：

1. 对原声及生成独立对齐，匹配相同字符/音节；需要音素时增加明确的中文音素对齐工具，不能将逐字时间戳称为精确音素边界。
2. 提取带浊音/置信度掩码的 F0、能量、持续时间、停顿；设定合理的角色音高搜索范围，排查八度错误。
3. F0 转换成半音单位分析，既报告绝对声线差，又报告去中心后的局部轮廓；不能只去中心而掩盖声音整体过高/过低。
4. 每个音节内部可作有界重采样或受约束 DTW；独立保留真实时长、停顿与对齐路径。不能让无限时间拉伸把错误节奏对齐成“很好”。

推荐字段：

| 字段 | 含义 |
|---|---|
| `f0_absolute_rmse_st` | 有效匹配浊音区域的绝对音高差，单位半音 |
| `f0_contour_rmse_st` / `f0_contour_corr` | 去中心后轮廓的差与相关；近常数曲线的相关记 unavailable |
| `f0_delta_error`、`f0_curvature_error` | 斜率、转折、回收等曲线结构差；结合语音边界，不单独硬判 |
| `voicing_agreement` | 浊音/非浊音结构是否对应 |
| `syllable_duration_log_ratio` | 音节时长变化；与曲线指标分开 |
| `pause_boundary_f1`、`pause_duration_error` | 停顿位置与持续时间；容差按语速校准 |
| `tail_f0_slope`、`tail_duration`、`tail_energy_decay` | 尾音的音高、拉长与能量回收 |
| `energy_contour_distance` | 相同文本的重音与能量形状，固定响度处理 |

原声只有一次表演，不是同一句唯一正确的韵律。新生成存在合理变化时，应看目标场景与人工标签，不要求所有曲线逐点一致。

**没有同文本原声时**：

- 只比较该角色在相似场景下的语速、停顿、F0 范围/变化、尾音和气息分布；报告 `style_distribution_match`，不冒充逐句习惯相似度。
- 相同文本多 seed 可运行 DS-WED，衡量变化量；仅在内容均正确时使用，跨不同台词的 token 距离会混入内容差异。[S6][S7]
- F0 标准差更大、DS-WED 更大，均不代表更好。破损、乱停顿也会“更有变化”。
- 情绪分类可作可选证据；“识别为开心”不等于学会了她的开心说法。第一版不依赖通用音频大模型自由打分。

### 6.5 稳定性：不要让平均值掩盖坏块

长文同时计算每个生成块、拼接成品和连续窗口。检查块间声音身份、响度、语速及静音衔接；保留生产 chunk 文本、逐块 seed、adapter 状态和真实重试次数。

- NaN/Inf、空文件、全零且目标有文本：可作确定性 FAIL。
- 达到 generation token 上限、没有明确结束、异常拖长：需真实生成日志支持；不能从 WAV 单独臆造 EOS 状态。
- true peak 接近/超过 0dBTP、单个采样峰值高，不单独证明已经削波。结合输出格式、满刻度平台和试听校准。
- 所有候选及重试都计入失败率。不能仅保存 retry 选中的一条，把不稳定系统报告成 100% 成功。

## 7. 音频预处理与计算可比性

保留三种明确用途的视图：

1. **raw**：原始交付 WAV，不做降噪或响度修复；基础损伤、峰值和整体评分默认取此视图。
2. **analysis**：按每个指标规范重采样/声道处理的副本，保留转换 hash；必要的 VAD/边缘静音处理固定，原始停顿仍单独保存。
3. **listening**：仅用于公平试听的响度匹配副本，记录增益且避免削波；不覆盖 raw 的质量结果。

24k/48k 文件不能直接比较所有高频能量。频谱分析使用本轮固定的共用带宽；若某输入只有 16k，8–12k 指标为不适用，不能将零值视为高频差。升采样不能创造缺失频段。

UTMOS、SCOREQ、声纹等常用 16k 输入，因此 8k 以上的高频损伤可能被忽略。保留项目现有 24k 共用频谱分析和原采样率上下文；只跑 16k 神经评分器不足以证明全频段清澈。

普通 PESQ/STOI、波形误差、SI-SDR 等需要适当的匹配参考；它们不适合拿任意人物参考音直接比较新台词。codec roundtrip 可在处理延迟与对齐正确时使用匹配指标；自然 TTS 表演差异下的 MCD/DTW 也不能直接作为最终音质门槛。

## 8. 分数、校准和自动判定

### 8.1 第一阶段不制造“看起来科学”的分数

展示 native score 与基线差值：

```text
CAM++ cosine：原始值 + 与基线差值 + gallery 波动
UTMOSv2：预测 MOS + delta + fold/重复配置
SCOREQ-NR：预测质量 + delta
SCOREQ-REF：距离 + delta（低为好，单独显示）
韵律：曲线/时长/停顿各项 + 时间位置
内容：CER + S/D/I + 关键字
稳定：失败计数 / 总尝试数 + 重试分布
```

不能凭空设置“cosine≥0.85、UTMOS≥4.2、HNR≥20dB 即合格”。不同编码器、参考、语言、情绪、长度及声学条件会改变这些数字。

### 8.2 校准实际的人耳目标

同文本 A/B 的标签分别记录：

- `quality_regression`：是否有可听见的质量退步；另记是否本身可接受。
- `speaker_likeness`：人物声音是否足够像，候选是否改善。
- `habit_expression`：目标场景中的节奏、尾音、重音和气息是否合理。
- `content_failure`：漏字、重复、错读、截断。
- `uncertain` / `fatigue`：不进入确定的训练或合格标签。

每次 3–5 对，顺序随机，不显示底座或 adapter；加入少量重复样本检查一致性。现有已确认的坏例优先复用，避免让用户重复听大量相近样本。

先检查各指标的坏例召回、假警报、成对方向准确率、Spearman 相关及置信区间。某指标在已知磨砂/破损上反向或完全无响应时，不能作为该类失败的通过依据。

样本足够后，每个维度使用少量明确特征训练带正则的简单校准器，例如质量退步分类器；按文本/源片段分组交叉验证，并留出底座或训练阶段检查迁移。不要用每个音频窗口当成独立样本扩充样本数。

24–40 对足以起步筛查，不证明可靠自动放行。举例：30 个独立坏例全部检出，零漏检的单侧 95% 漏检率上限仍约 9.5%（二项模型假设下）；“这批全检出”不能写成“实际不会漏检”。因此保存样本量、类型覆盖和不确定区间。

### 8.3 若用户需要 0–100 分

校准后可以输出：

$$S_{quality}=100[1-\widehat P(\text{候选相对可接受基线出现可听退步})].$$

它表示**本校准域中的质量保持估计**，不是声音达到真人的百分比；还需独立检查候选绝对可接受性。音色和表达可各有对应的人工目标与校准器，未校准时填 null。

只将 MOS 线性缩放或 cosine 乘 100，并不会获得这个概率。计算器版本、特征、标注定义、校准划分和误差须写入 `calibration.json`；新角色/表达/底座超出覆盖时转入 REVIEW。

### 8.4 四种统一状态

| 状态 | 条件 | Agent 的动作 |
|---|---|---|
| FAIL | 确定性故障，或已验证的本地失败规则命中 | 不升级；记录具体原因、片段和最后一个正常权重；停止或回退该试验分支 |
| PASS_AUTO | 必需指标成功、身份与数据可追溯、校准覆盖有效，各项通过且没有未解释的冲突 | 可以进入下一轮验证；并不自动替换生产默认 |
| REVIEW | 指标冲突、靠近门槛、新表达、疑似损伤、短音频不可靠，或还未校准 | 生成有限的局部 A/B 包，保留不确定，不假判正常 |
| BLOCKED | 文件/权重/依赖缺失，必需模型未执行，关键 hash 或配置缺失 | 补齐运行条件；不能将跳过检查写成 PASS |

某个可选模型未运行可以记录 `skipped_optional`；必需模型失败不能同样跳过。状态是工程决策，报告必须保留支撑证据，不能声称自动结果等于人耳真值。

**没有校准时允许确定性 FAIL 和计算原始分数，其余质量判断为 REVIEW。**这只是首次建设阶段；完成校准后，覆盖内的常规样本应能自动判定，复核集中在异常及冲突样本。

## 9. 基线比较与阶段预算

默认先比较同一底座的“固定零样本/正常权重”与候选，再比较不同底座。

| 模式 | 生成预算 | 评分预算 |
|---|---|---|
| quick | 12 条文本 × 1 seed × 2 方案 = 24 条；相同基线可缓存 | CAM++、主 ASR、固定 quick UTMOSv2、SCOREQ、基础诊断 |
| qualification | 20 条 × 2 seeds × 2 方案 = 80 条；与 quick 重叠的可核验复用 | 完整指标、固定 UTMOS ensemble/重复策略、第二声纹/必要复核、长文 |
| diagnostic | 不扩大生成搜索，优先重用失败 WAV | 局部谱/F0、NISQA、codec roundtrip 或感知评分候选 |

quick 与 qualification 是不同 profile，原始分数不能混作同一评分设置。比较方案数或候选数变化要显式记账；不能某模型生成十次挑最好，另一个只生成一次。

对同文本的差值做**按文本分组的成对 bootstrap**，保留两个 seed；报告平均、低分位、最差 case、失败率与区间。参考有实际重复录音时额外计算原声变异；没有重复录音时不假设唯一理想表演。

先约束“没有超过预设可听退化裕量”，再比较音色/习惯收益。退化裕量由 calibration 定义，不能从最终测试反向挑选使某个 checkpoint 过关的数值。

## 10. RTX 5080 16GB / Windows 的落地方式

- 新建隔离的 evaluation 环境，避免改变正在工作的各 TTS 后端依赖。
- 先完成生成并卸载 TTS，再依次评分；batch 1 起步，CAM++、ASR、UTMOSv2、SCOREQ 串行驻留，不同时占用全部模型。
- SCOREQ 优先验证官方 ONNX 推理；GPU 不兼容时可用 CPU，但需记录 execution provider 和数值一致性，不能静默换计算路径。
- UTMOSv2 quick 用固定 fold/重复；正式评价按固定 ensemble 配方逐 fold 加载，保存重复采样的配置与波动。
- 对 raw WAV hash、预处理 hash、评分器代码/权重/config/hash 缓存；任何身份或配置改变均使对应缓存失效。
- 下载按项目偏好优先镜像，失败再走已有 7897 代理；校验权重 hash，记录实际来源。不能把无法下载的模型在报告中标成已执行。
- 记录评分时延、显存峰值、失败重试；以上仅为运行策略，本次没有测出这些模型在 5080 上的实际占用。

## 11. 给执行 Agent 的实施任务

### 阶段 A：统一输入和原始评分

拟新增：

- `scripts/eval/run_selfcheck.py`：统一入口，支持现有 manifest 及 `quick/qualification/diagnostic` profile。
- `src/character_tts/evaluation/selfcheck/`：评分适配器、资产检查、汇总与规则；不将新依赖直接塞进已有生成 worker。
- `configs/evaluations/selfcheck_v1.yaml`：必需/可选指标、数据集、预处理、评分模式与预算。
- `outputs/selfcheck/<run_id>/`：逐条/逐窗 JSONL、CSV、摘要、失败局部试听。

入口命令是实施后的目标，当前还不能运行：

```bash
python scripts/eval/run_selfcheck.py --manifest <manifest.json> --profile quick --out <out_dir>
```

先复用已有 CAM++ 与 ASR，加入 UTMOSv2、SCOREQ-NR。全部指标通过固定接口返回：`status/value/unit/direction/model_revision/weights_sha256/config_sha256/audio_view/windows/error`。只生成原始评价，未校准阶段不宣布音色或音质 PASS。

### 阶段 B：定位和项目校准

- 补齐 ASR 编辑路径、对齐时间与关键字。
- 同文本原声的 F0/时长/停顿分析；跨句身份漂移。
- 导入有 WAV/hash 的既有好坏标签；按第8节构建校准和有限 A/B。
- 依据留出结果启用明确的自动判定范围。

### 阶段 C：新方法是否值得升级

用完全相同的 calibration/eval 分别检查 WavLM、VoiceMOS 感知评分头、DS-WED 与 NISQA；评估是否减少真实坏例漏检或假警报，以及时延成本。新方法只有名字更新、不增加有效证据，就不升级默认。

CAL-MOS 等新预印本继续跟踪可部署代码与权重；不先投入训练大型评分器。小样本阶段优先校准已有表征和特征组合。

### 阶段 D：接入训练与发布评价

- 每个保存点优先跑少量固定回归 case；前期密集检查，避免最后一轮才发现破损。
- checkpoint 排序使用受约束的多维证据；loss、声纹或 MOS 不单独决定最优。
- 发布候选运行 qualification；保留真实失败率和未解决 REVIEW。
- 不将本评价器直接作为唯一可优化训练奖励；否则需要独立评分器与人工留出验证，防止把分数优化好而听感变差。[S4]

## 12. 统一结果示例

以下只展示结构，没有本地测量值：

```json
{
  "schema_version": "tts-selfcheck-v1",
  "illustrative_only": true,
  "run_id": "example",
  "case_id": "heldout_01",
  "backend": "qwen3_tts",
  "audio_sha256": null,
  "baseline_audio_sha256": null,
  "calibration_id": null,
  "scores": {
    "speaker_camplus_cosine": null,
    "speaker_wavlm_cosine": null,
    "speaker_perceptual_mos": null,
    "quality_utmosv2": null,
    "quality_scoreq_synthetic_nr": null,
    "quality_preservation_0_100": null,
    "cer": null,
    "f0_contour_rmse_st": null,
    "pause_boundary_f1": null
  },
  "metric_status": {"all": "not_run"},
  "events": [],
  "verdict": "BLOCKED",
  "reasons": ["EXAMPLE_NOT_MEASURED"]
}
```

实际 `events` 必须带 `start_sec/end_sec/type/evidence`，例如疑似音染、ASR 删除、异常尾音或身份漂移；没有定位能力时声明 `utterance_level`，不能编造时间点。

## 13. 实施验收与停止条件

必须验证：

1. 所有 WAV 都有原始、预处理及基线身份；相同输入/配置可复算，随机评分的波动已记录。
2. 零字节、全零、NaN/Inf、损坏 WAV 与必需模型缺失不会误报通过。
3. 故意漏字、重复、长静音、声线替换和合成削波样例能触发相应检查；人工构造坏例只验证机制，不能替代真实模型坏例。
4. 同一原声在 leave-one-out 对照中表现合理；待评分音频未被加入参考中心。
5. 参考模式 SCOREQ 的距离方向正确；NISQA-TTS 与多维模型未混淆。
6. 已确认的本项目磨砂/破损样例若未被评分发现，必须保留漏检并阻止将该指标写成对应故障的充分通过条件。
7. 报告同时展示质量、身份、表达、内容与失败率；没有均值遮蔽坏块、重试掩盖失败、缺失值装作正常。
8. calibration 与最终测试无源片段泄漏；提交评分代码、配置、版本和文本型报告，大 WAV 与权重仍按项目原方式存放。

达到这些条件后，可以用自检支持日常 Agent 迭代，把人工试听集中到新问题和难以区分的高质量候选。若仍需每条都人工复核，应说明是校准未覆盖、指标无效还是程序尚未完成，不能仅增加更多未经验证的评分器。

## 14. 来源与证据范围

| 编号 | 原始来源 | 本次取得的证据 |
|---|---|---|
| S1 | [声纹几何与人耳相似度，2026](https://arxiv.org/abs/2609.33999) | 摘要及 HTML；未复现实验，预印本 |
| S2 | [VoiceMOS 2026 论文](https://arxiv.org/html/2609.13792v1)；[赛事官网](https://sites.google.com/view/voicemos-challenge/voicemos-challenge-2026) | 不同评价任务、域和 baseline |
| S3 | [VoiceMOS 2026 官方 Track 3](https://github.com/voicemos-challenge/vmc2026-baselines/tree/f9482fbb81d0429e090b491dd7aa7870a4b040f0/track3) | 克隆、README/模型源码及实际 checkpoint 文件；未运行 |
| S4 | [现代 TTS 无参考指标的局限](https://arxiv.org/abs/2609.13150) | 原始摘要及作者出版页；未取得完整正文 |
| S5 | [人类与 MOS 模型对声学/韵律扰动的差异](https://arxiv.org/abs/2606.19951)；[作者出版页](https://www.sp.nitech.ac.jp/~takagi.masato/) | 原始摘要及 Interspeech 2026 出版记录；未复现实验 |
| S6 | [DS-WED / ProsodyEval 论文](https://arxiv.org/html/2509.19928v1) | 原始摘要、HTML及 ICASSP 2026 记录；英语验证范围 |
| S7 | [DS-WED 作者代码](https://github.com/yfyeung/DS-WED) | API、token 化与距离定义；未运行 |
| S8 | [CAL-MOS](https://arxiv.org/abs/2609.14956) | 原始摘要；未核实权重、未复现 |
| S9 | [CAM++ / 3D-Speaker 官方代码](https://github.com/modelscope/3D-Speaker) | 官方模型/训练路线及项目实际 CAM++ 接线 |
| S10 | [Microsoft 说话人验证实现](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification) | 官方模型及 pooling/head 路线 |
| S11 | [UTMOSv2 官方代码](https://github.com/sarulab-speech/UTMOSv2/tree/cc2700db57bb83ee13dc31ebe1b868c254e15d09)；[推理说明](https://github.com/sarulab-speech/UTMOSv2/blob/cc2700db57bb83ee13dc31ebe1b868c254e15d09/docs/inference.md) | 克隆并核对 API、domain、fold、重复策略 |
| S12 | [SCOREQ 官方代码](https://github.com/alessandroragano/scoreq/tree/0cb0b168d0f7ec1419475d1e7b7ea699d8cd599e)；[NeurIPS 2024 论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bece7e02455a628b770e49fcfa791147-Abstract-Conference.html) | 克隆并核对 ONNX、domain、NR/REF、距离及输入长度 |
| S13 | [NISQA 作者代码](https://github.com/gabrielmittag/NISQA) | 官方 README，权重用途及维度区别 |
| S14 | [Qwen3-ASR / ForcedAligner 官方代码](https://github.com/QwenLM/Qwen3-ASR) | 官方接口及逐字/词时间戳能力 |
| S15 | [Praat CPP 说明](https://praat.org/manual/PowerCepstrum__Get_peak_prominence___.html)；[CPPS 说明](https://praat.org/manual/PowerCepstrogram__Get_CPPS___.html) | 官方倒谱峰突出度定义与参数；未在本角色音频上运行 |

[S1]: https://arxiv.org/abs/2609.33999
[S2]: https://arxiv.org/html/2609.13792v1
[S3]: https://github.com/voicemos-challenge/vmc2026-baselines/tree/f9482fbb81d0429e090b491dd7aa7870a4b040f0/track3
[S4]: https://arxiv.org/abs/2609.13150
[S5]: https://arxiv.org/abs/2606.19951
[S6]: https://arxiv.org/html/2509.19928v1
[S7]: https://github.com/yfyeung/DS-WED
[S8]: https://arxiv.org/abs/2609.14956
[S9]: https://github.com/modelscope/3D-Speaker
[S10]: https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification
[S11]: https://github.com/sarulab-speech/UTMOSv2
[S12]: https://github.com/alessandroragano/scoreq
[S13]: https://github.com/gabrielmittag/NISQA
[S14]: https://github.com/QwenLM/Qwen3-ASR
[S15]: https://praat.org/manual/PowerCepstrogram__Get_CPPS___.html
