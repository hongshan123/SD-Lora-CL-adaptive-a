# Writing Rationale Matrix

| Row ID | Manuscript Unit | Planned Function | Research-Spine Link | Literature/Exemplar Pattern | Evidence/Citation Anchor | Planned Text Move | Claim Boundary | Final Check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R01 | Abstract | 压缩全文主张 | 问题→方法→结果 | CIT-01-05 的摘要句式 | C1-C5 | 先给问题与数字，再给方法一句话 | 仅声称状态效率与统计等价 | 与正文数字一致 |
| R02 | Intro ¶1 | 建立问题 | 无回放 CIL + LoRA | CIT-01 | E-INR-BASE | 基线数字 | 不泛化到所有 CIL | 数字准确 |
| R03 | Intro ¶2 | 综述分类 | 三类现有方法 | CIT-03-14 | — | 分组叙述 | 不穷举 | 引文库复核 |
| R04 | Intro ¶3 | Gap | 累计代数结构未用 | CIT-02,05,10 | C1-C2 | 直接指出 | 只在 Shared-A 设定 | — |
| R05 | Method 3.2 | 等价性 | C1 | — | E-AUDIT-UNIT | 公式+误差界 | 固定 A 口径 | 测试覆盖 |
| R06 | Method 3.3 | 对齐 | C2 | Balanced LoRA canonical 讨论 | E-AUDIT-UNIT、E-INR-DIAG | 闭式解推导 | 行空间内保持 | 残差表 |
| R07 | Experiments 4.2 | 主结果 | C5 | 标准 CIL 表 | E-MULTI/E-PAIR | 配对表+p | n=4 | 显著性诚实 |
| R08 | Experiments 4.3 | 效率 | C3/C7 | 参数-T 曲线 | E-PARAMS/E-EFF | 表+图 | 训练临时态另计 | 测量口径 |
| R09 | Experiments 4.5 | 外推 | C6 | 额外数据集 | E-CUB | 对比表 | 单 seed | 标注 |
| R10 | Discussion | 机制解释 | C2/C9 | E2-LoRA 漂移结构 | E-INR-DIAG/E-CORR | 诊断→机制 | 非因果证明 | 相关性弱如实 |
| R11 | Limitations | 边界 | 全部主张 | — | — | 列未做事项 | — | 与计划一致 |
