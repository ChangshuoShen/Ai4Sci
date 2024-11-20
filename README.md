# Starting Point

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
* <a href="./papers/Rag2Mol.pdf">查看PDf</a>
* <a href="https://www.biorxiv.org/content/10.1101/2024.10.20.619266v1">链接</a>
* <a href="https://github.com/CQ-zhang-2016/Rag2Mol">codes</a>

#### 主要内容
1. 背景
    * 找到具有最佳`理化性质`和`药理性质`的先导化合物是一大挑战
    * 基于结构的药物设计(SBDD)成为一个很有前景的范式\
2. 主要创新：
    * 提出了两种基于检索增强生成(RAG)的方法：Rag2Mol-G和Rag2Mol-R
    * 这两种方法分别用于:
        * Rag2Mol-G: 基于生成的分子在数据库中搜索相似的可购买分子
        * Rag2Mol-R: 从数据库中创建能够适配3D口袋的新分子
3. 研究结果：
    * Rag2Mol方法能持续产生具有更优结合亲和力和药物性的候选药物
    * Rag2Mol-R相比传统虚拟筛选模型提供了:
        * 更广泛的化学空间覆盖
        * 更精确的靶向能力
    * 两种工作流都成功识别出了针对具有挑战性靶点PTPN2的潜在抑制剂
4. 研究意义：
    * 提供了一个可扩展的框架，可以整合多种SBDD方法