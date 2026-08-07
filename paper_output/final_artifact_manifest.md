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

- 实验记录：`experiment_sd.md`（EXP-000~019）、`experiment_note_sd.md`、`plan_sd.md`、`method_revision_sd.md`。
- 代码：`backbone/sa_lora.py`（v2 状态、累计折叠、canonical/gauge 纯函数）、`models/sa_sdlora.py`、`utils/inc_net.py`、迁移/测量/统计/相关性脚本。
- 测试：`tests/test_sa_cumulative.py` 等（45 passed）。
- 配置与日志：`exps/sa_cumulative_*.json`、`sa_cumulative_*.log`、`sa_sdlora_proto_*seed*.log`。
- 产物目录（gitignore）：`ImageNetR_SA_CUMULATIVE_*`、`CF100_SA_CUMULATIVE_*`、`CUB_SA_CUMULATIVE_*`、`ImageNetR_SA_SDLORA_PROTO_*`、`CF100_SA_SDLORA_PROTO_*`、`CUB_SA_SDLORA_PROTO_*`。

## Open items before submission

- CIT 表核实（5+ 条 `TODO_CITATION`，2 条 `TODO_VERIFY`）。
- 训练峰值显存/吞吐测量表（`TODO_EVIDENCE`）。
- CUB 多 seed、ImageNet-A/DomainNet（可选）。
- 外部强基线复现（可选/limitations）。
- 图（参数-T 曲线、诊断曲线、CIL 曲线）与 LaTeX 版式。
