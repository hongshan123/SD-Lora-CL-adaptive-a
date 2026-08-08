# Final Artifact Manifest

## Deliverables (paper_output)

| Artifact | Status |
| --- | --- |
| `confirmed_research_spine.md` | Done |
| `evidence_bank.md` | Done |
| `claim_register.md` | Done |
| `citation_support_bank.md` | Done（5 条待核实） |
| `section_blueprints.md` | Done |
| `writing_rationale_matrix.md` | Done |
| `first_draft/main.md` | Done（首稿，含 TODO 标记） |
| `first_draft/main.tex` | 可选，未生成（后续按需） |

## Supporting evidence (repo)

- 实验记录：`experiment_sd.md`（EXP-000~027）、`experiment_note_sd.md`、`plan_sd.md`、`method_revision_sd.md`、`goal_live_a_sd.md`、`live_a_aggregate_b_modification_sd.md`。
- 代码：`backbone/sa_lora.py`（v2/v4 状态、累计折叠、gauge 纯函数、Live-A Aggregate）、`models/sa_sdlora.py`、`utils/inc_net.py`、`backbone/linears.py`（prototype/multi-prototype head）、迁移/测量/统计/诊断脚本。
- 测试：`tests/test_sa_cumulative.py`、`tests/test_live_a_aggregate_backbone.py` 等（81 passed）。
- 配置与日志：`exps/sa_cumulative_*.json`、`exps/live_a_*.json`、`sa_cumulative_*.log`、`sa_sdlora_proto_*seed*.log`、`live_a_*seed*.log`、`sdlora_*paired_rerun*.log`。
- 产物目录（gitignore）：`ImageNetR_SA_CUMULATIVE_*`、`CF100_SA_CUMULATIVE_*`、`CUB_SA_CUMULATIVE_*`、`ImageNetR_SA_SDLORA_PROTO_*`、`CF100_SA_SDLORA_PROTO_*`、`CUB_SA_SDLORA_PROTO_*`、`ImageNetR_LIVE_A_*`、`CF100_LIVE_A_*`。
- 统计与诊断：`scripts/stage_b_stats.sh`（seed 内配对统计）、`stage_b_stats_output.txt`、`stage_b_dual_stats_output.txt`、`scripts/diagnose_live_a_dual_head.py`。

## Open items before submission

- CIT 表核实（5+ 条 `TODO_CITATION`，2 条 `TODO_VERIFY`）。
- 训练峰值显存/吞吐测量表（`TODO_EVIDENCE`）。
- CUB 多 seed、ImageNet-A/DomainNet（可选）。
- 外部强基线复现（可选/limitations）。
- 图（参数-T 曲线、诊断曲线、CIL 曲线）与 LaTeX 版式。
- Live-A Aggregate-B 已按停止条件记录为负结果/消融（EXP-023~027，E-LIVE-A），不进入主方法表。
