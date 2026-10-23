# Additional-seed Frozen controls

Activate the four Frozen-A configurations already prepared for CUB seeds 2/3
and ImageNet-R seeds 1996/1997. The unchanged main-family training worker and
source are reused. Source, configuration, existing numerical gates and input
signatures are verified before freezing fresh execution inputs. Only the seed
fields and operational output identifiers differ from the initial controls.

The scheduler is the original main scheduler with only its smoke phase omitted
for these unchanged controls. It uses the same environment and inherited GPU
lease descriptors, refuses existing outputs and never silently retries a failed
or apparently stale process. GPU6 is used after checking current occupancy.
No main, B-metric, A-step or allocation training process is stopped.

Run preparation once in a fresh directory, then start `run_queue.py` with the
same training interpreter. Runtime records are local and excluded from GitHub.
