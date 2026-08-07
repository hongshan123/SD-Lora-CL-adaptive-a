# Citation Support Bank

> 引用 ID 规则：CIT-<编号>。已验证来源标 [verified]，需要补全/核实的标 [TODO_CITATION]。

| CIT | 文献 | 作用 | 状态 |
| --- | --- | --- | --- |
| CIT-01 | SD-LoRA（ICLR 2025, "Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning", OpenReview 5u1rlpx68a） | 本工作的骨干基线：幅度-方向解耦、逐任务 B bank | [verified] |
| CIT-02 | SA-LoRA（J. King Saud Univ. Comput. Inf. Sci. 2026, s44443-026-00925-x） | 共享 A + 每任务 B 非对称共享；无原型漂移补偿 | [verified]（method_revision_sd.md 碰撞审计） |
| CIT-03 | CL-LoRA（CVPR 2025, arXiv:2505.24816） | 训练期原型余弦分类器、双适配器；无递归原型 transport | [verified ID；页面待复核 TODO_CITATION] |
| CIT-04 | RanPAC（NeurIPS 2023, arXiv:2307.02251） | 冻结骨干+随机投影+类原型+去相关 | [verified ID；页面待复核 TODO_CITATION] |
| CIT-05 | LDC（ECCV 2024, arXiv:2407.08536） | 学习前向投影网络补偿旧原型 | [verified ID；页面待复核 TODO_CITATION] |
| CIT-06 | InfLoRA（CVPR 2024, arXiv:2404.00228） | 注入参数重参数化固定子空间 | [verified ID；页面待复核 TODO_CITATION] |
| CIT-07 | FM-LoRA（CVPR 2025 Workshop DG-EBF, arXiv:2504.08823） | 共享基+任务系数、动态 rank | [verified ID；页面待复核 TODO_CITATION] |
| CIT-08 | C-LoRA（arXiv:2502.17920） | 单一 LoRA + 可学习路由 A·R·B | [verified ID；页面待复核 TODO_CITATION] |
| CIT-09 | EASE（CVPR 2024, "Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning", arXiv:2403.12030） | 每任务适配器子空间 + 语义原型补全 | [verified] |
| CIT-10 | E2-LoRA（ICML 2026, "Energy-Structured Low-Rank Adaptation for Continual Learning", arXiv:2605.27482） | output feature drift 低秩结构与能量排序 | [verified] |
| CIT-11 | Janus-LoRA（ICML 2026, "Janus-LoRA: A Balanced Low-Rank Adaptation for Continual Learning", arXiv:2605.28495） | closed-form LoRA gradient rectification | [verified] |
| CIT-12 | Balanced LoRA（ICML 2026, "Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence", BaLoRA） | LoRA 因子表示不唯一与 balanced manifold | [verified venue；arXiv ID 待补] |
| CIT-13 | LoRA-DRS（CVPR 2025, "LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning", arXiv:2503.18985） | LoRA 减法构建漂移抵抗空间 | [verified] |
| CIT-14 | DGS（CVPR 2026, "DGS: Dual Gradient and Semantic-Shift Guided Low-Rank Adaptation for Class Incremental Learning"） | LoRA gradient 与 semantic-shift prototype alignment | [verified venue；arXiv ID 待补] |
| CIT-15 | ICLR 2026 Two-Way Alignment（"Two-Way Is Better Than One: Bidirectional Alignment with Cycle Consistency...", BiCyc, arXiv:2606.05675） | 旧/新特征双向映射、cycle consistency | [verified] |
| CIT-16 | CT-Merging（arXiv:2607.20561） | 模型合并系数幅度显式处理 | [已在前文记录；需复核编号 TODO_CITATION] |
| CIT-17 | Subspace-Boosted Model Merging（arXiv:2506.16506） | 合并随专家数 rank collapse | [已验证于前文记录；页面待复核 TODO_CITATION] |
