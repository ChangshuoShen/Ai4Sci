## Starting Point

### Antigen-Specific Antibody Design via Direct Energy-based Preference Optimization
基于能量偏好优化的抗原特异性抗体设计方法
* <a href="./papers/Antibody_DPO.pdf">查看PDF</a>
* <a href="https://arxiv.org/abs/2403.16576">链接</a>

#### 主要内容:

1. 研究目标
    * 通过优化问题解决抗原特异性抗体的序列结构共同设计问题
    * 使用Equivariant Neural Networks(等变神经网络)
        * f(T(x)) = T(f(x))，平移旋转以及SE(3)等变性
2. 核心方法：
    * 利用预训练的条件扩散模型，联合建模抗体的序列和结构
    * 提出直接基于能量的偏好优化方法(direct energy-based preference optimization)
    * 使用残基级能量分解进行模型微调
    * 采用梯度手术来处理不同类型能量之间的冲突

3. 实验结果：
    * RAbD基准测试
    * 能够同时实现低总能量和高结合亲和力的抗体设计
    * 证明了该方法的优越性


### Aligning Target-Aware Molecule Diffusion Models with Exact Energy Optimization
* <a href="./papers/aligning-target-aware-molecule-diffusion.pdf">查看PDF</a>
* <a href="https://arxiv.org/abs/2407.01648">链接</a>
* <a href="https://github.com/MinkaiXu/AliDiff">codes</a>

#### 主要内容
1. 研究问题
    * 解决基于结构的药物设计问题——生成能与特定蛋白质靶点结合的配体分子
    * 旨在改进现有模型，引导分子生成朝向更好的结合亲和力和结构特性方向发展
2. 主要贡献——ALIFF框架
    * 提出了一个新的预训练目标感知和扩散模型的对齐框架
    * 将目标条件化的化学分布转向：
        * 更高的结合亲和力
        * 更好的结构合理性
    * 使用精确能量优化(E2PO)的偏好优化算法
3. 技术创新
    * 开发改进的E2PO算法
    * 提供收敛分布的闭式表达
    * 解决偏好优化中常见的过拟合问题
    * 结合了离散化学类型和3D坐标特征
4. 细节
    * 对原子类型(离散)和3D坐标(连续)特征都使用扩散模型
    * 整合用户定义的奖励函数

### Enhancing Protein Mutation Effect Prediction through a Retrieval-Augmented Framework
* <a href="./papers/Protein_Mutation_prediction_RAG.pdf">查看PDf</a>
* <a href="https://neurips.cc/virtual/2024/poster/95577">链接</a>

#### 主要内容
1. 创新：
    * 开发结构基序嵌入数据库(SMEDB)，包含了蛋白质数据库(PDB)中所有结构
    * 提出多重结构基序对齐(MSMA)方法来检索局部协同进化基序
    * 设计多结构基序不变点注意力(MSM-IPA)模型来预测蛋白质结构适应度
2. 技术特点：
    * 使用ESM-IF模型作为结构编码器
    * 采用CUHNSW进行GPU加速的k近邻搜索
    * 关注局部微环境而不是全局蛋白质表示
3. 优势：
    * 能从整个蛋白质数据库中提取原子级信息
    * 检索结果与传统多序列比对(MSA)方法互补
    * 在多个基准数据集(S669、cDNA、SKEMPI)上性能优于现有方法
4. 应用价值：
    * 制药行业、药物设计、生物燃料生产等领域
    * 提供一个可扩展的框架来研究`蛋白质突变效应`


### Rag2Mol: Structure-based drug design based on Retrieval Augmented Generation
* <a href="./papers/Rag2M生成的分子在数据库中搜索相似的可购买分子
        * Rag2Mol-R: 从数据库中创建能够适配3D口袋的新分子
3. 研究结果：
    * Rag2Mol方法能持续产生具有更优结合亲和力和药物性的候选药物
    * Rag2Mol-R相比传统虚拟筛选模型提供了:
        * 更广泛的化学空间覆盖
        * 更精确的靶向能力
    * 两种工作流都成功识别出了针对具有挑战性靶点PTPN2的潜在抑制剂
4. 研究意义：
    * 提供了一个可扩展的框架，可以整合多种SBDD方法

## RAG
RAG在药物发现的一些相关文章

### Retrieval-Augmented Language Model for Knowledge-aware Protein Encoding
用于知识感知蛋白质编码的检索增强语言模型
* <a href="https://openreview.net/forum?id=0MVWOHwHDb">ICLR链接</a>
* <a href="./papers/7571_Retrieval_Augmented_Langu.pdf">查看PDF</a>
#### 摘要提取
* 问题背景：蛋白质语言模型（PLM，如ESM、ProtineinBert、ProtBert将氨基酸视作token）在捕捉蛋白质序列中编码的生物功能方面存在困难，原因在于缺乏事实性知识（如基因描述）。现有方法使用蛋白质知识图谱（PKG，提供蛋白质与基因本体GO之间的生物学关系）作为辅助编码目标，但存在不足：
    * 隐式嵌入知识，难以适应知识更新。
    * 忽略PKG中的结构信息，如蛋白质之间的高阶连接
    * 预训练和微调阶段的知识建模不一致，导致知识灾难性遗忘问题。
* 核心方法Kara：
    * 目标：
        * 实现PKG与蛋白质语言模型的统一建模，解决知识注入和高效利用问题
    * 核心组件
        * 知识检索器
            * 能够预测新蛋白质的潜在基因描述，生成与PKG对齐的知识
            * 检索过程
                * 根据新蛋白质的氨基酸序列，生成查询嵌入
                * 遍历PKG中`关系-实体`组合，生成候选知识，使用TransE得分函数对候选知识排序
            * 训练策略
                * 使用边际损失(margin loss)优化有效知识和无效知识之间的得分差异
                * 引入跨模态匹配损失，统一文本和氨基酸模态的嵌入空间
        * 上下文虚拟token
            * 将蛋白质的知识和结构信息分别构造成`知识虚拟token`和`结构虚拟token`，与氨基酸序列拼接，实现逐token的多模态信息融合
        * 结构正则化
            * 融入PKG中蛋白质的高阶连接信息，将功能相似性纳入表示学习中
            * 在预训练阶段和微调阶段统一优化，避免知识灾难性遗忘
    * 预训练阶段
        * 采用知识引导的掩码建模(MLM)任务，逐层学习氨基酸与虚拟token的交互
        * 使用边际损失约束高阶连接蛋白质的表示相似性
    * 微调阶段
        * 使用知识检索对新蛋白质生成潜在知识，与PKG对齐
        * 保持虚拟token和结构正则化的使用，统一优化目标
* 实验与分析：
    * 实验设计：
        * 在6个代表性任务上评估Kara，包括氨基酸接触预测（短程、中程、长程）、蛋白质相互作用预测（PPI）、蛋白质同源性检测等。
    * 实验结果：
        * Kara的性能全面优于现有方法（如KeAP、ESM-2）。
        * 示例：在长程接触预测任务上，Kara比KeAP提升11.6%，在蛋白质同源性检测任务上提升10.3%。
    * 消融实验：
        * 验证`知识检索器`、`虚拟token`以及`结构正则化`的独立贡献。


### ProtEx: A Retrieval-Augmented Approach for Protein Function Prediction
ProtEx：一种用于蛋白质功能预测的检索增强方法
* <a href="https://openreview.net/forum?id=ZxZabvtLwV">ICLR链接</a>
* <a href="./papers/11043_ProtEx_A_Retrieval_Augme.pdf">查看PDF</a>

#### 主要内容
* 研究问题
    * 将蛋白质序列映射到其生物学功能是一个关键问题，具有广泛的生物学、医学和化学应用。
    * 挑战
        * 分布外泛化问题：标签或序列空间的泛化能力不足
        * 稀有类别与“暗物质”蛋白：很多蛋白质的序列与已知分类类别差距较大，传统方法难以处理
    * 研究目标
        * 准确性、鲁棒性和泛化能力

* 方法：提出了ProtEx，一种基于检索增强的蛋白质功能预测方法，利用数据库中的样例（exemplars）来提高准确性、鲁棒性，并支持对未见类别的泛化。
    * 检索增强策略
        * 使用BLAST检索与候选标签相关的正负样例
        * 将这些样例与查询序列一起输入模型进行预测
    * 模型架构
        * 使用Transformer结构(T5)，支持多序列条件化输入
        * 模型针对每个候选标签进行二分类预测
    * 训练策略
        * 预训练：及基于无标签的序列对，设计一个多序列预训练任务，同时预测掩码残基和序列相似性分数
        * 微调：在有标注的数据集中，结合探索到的样例进行训练
    * 推理过程
        1. 索引相似序列
        2. 提取候选标签
        3. 选择正负样例
        4. 结合查询序列、候选标签和样例输入模型
        5. 聚合预测结果
* 实验与分析
    * 任务与数据集
        * EC 编号、GO 术语、Pfam 家族预测任务。
        * 数据集包含随机和聚类划分（基于序列相似性），以及不同的测试设置（如时间分割、新类别等）。
    * 主要实验结果
        * EC 和 GO 预测：
            * ProtEx 在随机和聚类分割上均超越基线（如 BLAST 与 ProteInfer）。
            * 在稀有类别和低相似性序列上改进显著。
        * 特殊测试集（如 NEW-392 和 Price-149）：
            * ProtEx 在时间分割和实验验证的 EC 数据集上表现最好。
        * Pfam 预测：
            * 在 Pfam 数据集上，ProtEx 提升了对低序列相似性测试集的预测能力。

    * 消融实验
        * 移除样例输入后性能显著下降，表明样例条件化预测是 ProtEx 的核心优势。
        * 不同样例选择策略、超参数的影响均进行了详细分析。
* 结论
    * ProtEx 提供了一种有效的检索增强方法，特别适用于稀有类别和未见类别的蛋白质功能预测。
    * 核心贡献：
        * 提出结合正负样例的检索增强预测方法。
        * 设计了一种多序列比较的预训练任务。
        * 验证了方法在多种数据集和任务上的泛化能力和性能优势。

### Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization
用于结构知情抗体设计和优化的检索增强扩散模型
* <a href="https://openreview.net/forum?id=a6U41REOa5">ICLR链接</a>
* <a href="./papers/5690_Retrieval_Augmented_Diffu.pdf">查看PDF</a>

#### 主要内容
* 背景与问题：
    * 抗体是免疫系统中的关键蛋白质，可特异性识别病原体的抗原分子。
    * 当前的抗体生成模型多从零开始设计，不考虑结构模板，容易导致模型优化困难以及产生不自然的序列。
    * 抗体设计的核心是优化`互补决定区（CDRs）`，尤其是其与抗原的结合能力。
    * 当前实验方法费时费力，需要动物免疫或筛选大型抗体库，难以高效开发目标导向的抗体。
    
    * 深度生成模型的主要问题包括：
        * 生成的序列未能充分利用结构约束。
        * 数据集规模有限，导致模型泛化能力不足。

* 相关工作
    * 抗体设计分为传统优化方法和机器学习方法。
        * 传统方法：基于`序列相似性`和`能量函数优化`（如 Rosetta）。
        * 机器学习方法：包括语言模型（如 ESM）和逆折叠模型（如 ProteinMPNN）。
        * 当前方法大多从头生成序列，缺乏明确的`结构约束`，导致功能性抗体设计面临挑战。
    * Diffusion Model 扩散生成模型：
        * 扩散模型擅长生成任务，具灵活性和可控性，已在文本生成、分子生成等领域取得成功。
    * RAG 检索增强生成模型：
        * 结合检索机制，可通过引入额外数据库，使生成模型更具多样性和现实性。
        * 本文首次将检索增强机制用于抗体设计。
* 方法（RADAb 框架）
    * 总体框架：
        * RADAb 包括结构检索模块与扩散生成模块。
        * 检索模块基于目标抗体的结构，提取相似的 CDR 片段（motifs）。
        * 扩散生成模块结合检索结果与抗体上下文信息，逐步优化抗体序列。
    * 结构检索：
        * 用 MASTER 工具在 PDB 数据库中检索与目标 CDR 匹配的片段，依据 RMSD（均方根偏差）排序。
        * 检索得到的片段包含结构与进化信息，有助于生成类似功能的抗体序列。
    * 模型架构：
        * 全局几何上下文分支：
            * 编码抗体与抗原的几何信息（如残基类型、原子坐标等）。
            * 利用进化信息编码器（如 ESM2）提取抗体序列的进化关系。
            * 通过结构感知网络生成全局条件概率分布。
        * 局部 CDR 聚焦分支：
            * 处理检索到的 CDR-like 片段，通过轴向注意力机制捕获局部相似性。
            * 结合检索片段的条件概率分布，生成最终的 CDR 序列。
    * 训练与推理：
        * 训练目标是最小化预测分布与真实分布的 KL 散度。
        * 推理时，通过扩散过程逐步去噪，生成符合结构和功能约束的抗体序列。

4. 实验

    实验设置：
        数据集：使用 SAbDab 数据库，并构建包含 CDR-like 片段的数据库。
        任务：抗体 CDR 逆折叠与优化。
        基线方法：包括 Grafting、ProteinMPNN、ESM-IF1、Diffab-fix 等。

    任务 1：抗体 CDR 逆折叠：
        RADAb 在所有 CDR 区域的氨基酸恢复率（AAR）上均优于其他方法。
        与传统方法相比，RADAb 更能生成结构合理且功能相关的序列。

    任务 2：抗体优化：
        在功能优化任务中，RADAb 显著改善了 ∆∆G（自由能变化），生成的抗体序列更加稳定和功能性。

    消融实验：
        去掉检索模块或进化信息模块后，模型性能显著下降，验证了检索增强机制的重要性。

5. 结论

    RADAb 提供了一种新颖的检索增强扩散框架，可结合结构和进化信息，高效设计抗体。
    实验表明 RADAb 的性能超越了现有方法，为抗体设计提供了新的视角。
    未来方向：
        扩展检索数据库，涵盖更多功能片段。
        进一步优化扩散模型的条件生成能力。




* 方法：
    * 提出一种检索增强扩散框架（Retrieval-Augmented Diffusion Framework），简称 RADAb。
    * 该方法利用结构同源的片段（motifs），结合输入骨架，通过一个双分支去噪模块，整合结构和进化信息，引导生成模型优化抗体序列。
    * 开发了一个条件扩散模型，结合全局上下文和局部进化条件，迭代优化抗体设计。
* 创新点：
    * 设计了一个基于结构的检索机制，结合模板片段与抗体骨架。
    * 提出了一个生成模型无关的框架，支持多种生成模型。
    * 通过实验证明，在抗体逆折叠与优化任务上，RADAb 超越了现有方法，性能达到了最先进水平。
* 实验结果：
    * RADAb 在多项抗体设计任务中表现优异，例如在长 CDRH3 区域逆折叠任务中，AAR 提升了 8.08%，在功能优化任务中，∆∆G 改善了 7 cal/mol。



### Retrieval Augmented Zero-Shot Enzyme Generation for Specified Substrate
针对特定底物的 Retrieval Augmented Zero-Shot 酶生成
* <a href="https://openreview.net/forum?id=T7lQGq73Lm">ICLR链接</a>
* <a href="./papers/674_Retrieval_Augmented_Zero_S.pdf">查看PDF</a>



### RAG-Diffusion
* <a href="https://arxiv.org/abs/2204.11824v1">Arxiv链接</a>
* <a href="./papers/2204.11824v1.pdf">查看PDF</a>
* <a href="https://github.com/CompVis/retrieval-augmented-diffusion-models">代码</a>

