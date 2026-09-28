"""Streaming activation and output-gradient statistics for Q/V branches."""

import torch


class QKVFunctionStatistics:
    def __init__(self, wrappers):
        self.wrappers = list(wrappers)
        if not self.wrappers:
            raise ValueError("at least one QKV wrapper is required")
        self._handles = []
        self._outputs = []
        self._input_sums = [None] * len(self.wrappers)
        self._output_sums = [None] * (2 * len(self.wrappers))
        self._input_counts = [0] * len(self.wrappers)
        self._output_counts = [0] * (2 * len(self.wrappers))

    def __enter__(self):
        if self._handles:
            raise RuntimeError("QKV statistics hooks are already active")
        for index, wrapper in enumerate(self.wrappers):
            self._handles.append(
                wrapper.register_forward_pre_hook(
                    lambda _module, inputs, index=index: self._capture_input(index, inputs)
                )
            )
            self._handles.append(
                wrapper.register_forward_hook(
                    lambda _module, _inputs, output, index=index: self._capture_output(
                        index, output
                    )
                )
            )
        return self

    def __exit__(self, _type, _value, _traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._outputs.clear()

    def _capture_input(self, index, inputs):
        if len(inputs) != 1 or inputs[0].shape[-1] != self.wrappers[index].dim:
            raise ValueError("QKV input shape is invalid")
        flat = inputs[0].detach().reshape(-1, self.wrappers[index].dim).double()
        squares = flat.square().sum(dim=0)
        if self._input_sums[index] is None:
            self._input_sums[index] = squares
        else:
            self._input_sums[index].add_(squares)
        self._input_counts[index] += flat.shape[0]

    def _capture_output(self, index, output):
        dim = self.wrappers[index].dim
        if not torch.is_tensor(output) or output.shape[-1] != 3 * dim:
            raise ValueError("QKV output must have three projection channels")
        if not output.requires_grad:
            output.requires_grad_(True)
        self._outputs.append((index, output))
        return output

    def begin_batch(self):
        if self._outputs:
            raise RuntimeError("the previous QKV batch was not accumulated")

    def accumulate(self, loss, batch_size):
        if loss.ndim != 0 or int(batch_size) <= 0:
            raise ValueError("loss must be scalar and batch_size positive")
        if len(self._outputs) != len(self.wrappers):
            raise RuntimeError("each QKV wrapper must run exactly once per batch")
        gradients = torch.autograd.grad(
            loss, [output for _, output in self._outputs]
        )
        for (index, _), gradient in zip(self._outputs, gradients):
            dim = self.wrappers[index].dim
            flat = gradient.detach().reshape(-1, 3 * dim).double()
            for offset, part in ((0, flat[:, :dim]), (1, flat[:, -dim:])):
                branch = 2 * index + offset
                squares = (part * int(batch_size)).square().sum(dim=0)
                if self._output_sums[branch] is None:
                    self._output_sums[branch] = squares
                else:
                    self._output_sums[branch].add_(squares)
                self._output_counts[branch] += flat.shape[0]
        self._outputs.clear()

    def summary(self):
        if self._outputs:
            raise RuntimeError("cannot summarize a batch before accumulating it")
        if any(count <= 0 for count in self._input_counts + self._output_counts):
            raise ValueError("QKV calibration has no complete samples")
        return {
            "input_moments": [
                (total / count).float()
                for total, count in zip(self._input_sums, self._input_counts)
            ],
            "output_sensitivities": [
                (total / count).float()
                for total, count in zip(self._output_sums, self._output_counts)
            ],
            "input_counts": list(self._input_counts),
            "output_counts": list(self._output_counts),
        }
