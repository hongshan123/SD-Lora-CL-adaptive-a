from scripts.wait_early_a_pair import gpu_pair_idle


GPUS = "0, gpu-a, 3, 0\n1, gpu-b, 3, 0\n2, gpu-c, 24235, 100\n"


def test_free_pair_ignores_jobs_on_other_cards():
    assert gpu_pair_idle(GPUS, "gpu-c, 123\n", (0, 1))


def test_compute_process_on_selected_card_blocks_launch():
    assert not gpu_pair_idle(GPUS, "gpu-a, 123\n", (0, 1))


def test_memory_and_utilization_both_must_be_idle():
    assert not gpu_pair_idle(GPUS.replace("gpu-a, 3", "gpu-a, 1681"), "", (0, 1))
    assert not gpu_pair_idle(GPUS.replace("gpu-b, 3, 0", "gpu-b, 3, 8"), "", (0, 1))


def test_missing_or_unreadable_gpu_information_fails_closed():
    assert not gpu_pair_idle(GPUS, "", (0, 3))
    assert not gpu_pair_idle("driver error", "", (0, 1))
    assert not gpu_pair_idle(GPUS, "driver error", (0, 1))
