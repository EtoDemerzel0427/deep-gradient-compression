import math
import random

import torch

import horovod.torch as hvd
from horovod.torch.mpi_ops import Average
from horovod.torch.mpi_ops import allreduce_async_
from horovod.torch.mpi_ops import allgather_async as allgather_async_
from horovod.torch.mpi_ops import synchronize as synchronize_

from dgc.memory import Memory

__all__ = ['RawTopKCompressor']


class RawTopKCompressor:
    def __init__(self, compress_ratio, memory=None,
                 fp16_values=False, int32_indices=False,
                 warmup_epochs=-1, warmup_coeff=None):
        self.world_size = hvd.size()
        self.op = Average
        self.fp16_values = fp16_values
        self.int32_indices = int32_indices

        self.base_compress_ratio = self.compress_ratio = \
            compress_ratio if compress_ratio <= 1.0 else 1.0 / compress_ratio
        self.memory = Memory if memory is None else memory
        self.warmup_epochs = warmup_epochs
        if self.warmup_epochs > 0:
            if warmup_coeff is None:
                self.warmup_coeff = self.base_compress_ratio \
                                    ** (1. / (self.warmup_epochs + 1))
            else:
                if isinstance(warmup_coeff, (tuple, list)):
                    assert len(warmup_coeff) >= self.warmup_epochs
                    for wc in warmup_coeff:
                        assert 0 < wc <= 1
                else:
                    assert 0 < warmup_coeff <= 1
                self.warmup_coeff = warmup_coeff
        else:
            self.warmup_coeff = 1


        self.attributes = {}

    def initialize(self, named_parameters):
        if hvd.rank() == 0:
            print("=> initializing dgc compressor")
        for name, param in named_parameters:
            if torch.is_tensor(param):
                numel = param.numel()
                shape = list(param.size())
            else:
                assert isinstance(param, (list, tuple))
                numel, shape = param[0], param[1]

            self.attributes[name] = (numel, shape)

    def warmup_compress_ratio(self, epoch):
        if self.warmup_epochs > 0:
            if epoch < self.warmup_epochs:
                if isinstance(self.warmup_coeff, (tuple, list)):
                    compress_ratio = self.warmup_coeff[epoch]
                else:
                    compress_ratio = max(self.warmup_coeff ** (epoch + 1),
                                         self.base_compress_ratio)
            else:
                compress_ratio = self.base_compress_ratio
        else:
            compress_ratio = self.base_compress_ratio
        if compress_ratio != self.compress_ratio:
            if hvd.rank() == 0:
                print(f'update compress ratio: {compress_ratio}')
            self.compress_ratio = compress_ratio
            self.initialize(self.attributes.items())

    def _sparsify(self, tensor, name):
        tensor = tensor.view(-1)
        numel, shape = self.attributes[name]

        importance = tensor.abs()
        indices = torch.topk(importance, int(importance.numel() * self.compress_ratio), 0, largest=True, sorted=False)[1]
        num_indices = indices.numel()

        values = tensor[indices]
        return values, indices, numel, shape, num_indices

    def compress(self, tensor, name):
        if self.compress_ratio < 1.0 and name in self.attributes:
            # compress, getting indices and values
            tensor_compensated = self.memory.compensate(
                tensor, name, accumulate=True)
            values, indices, numel, shape, num_selects = \
                self._sparsify(tensor_compensated, name)
            self.memory.update(name, (indices,))
            indices = indices.view(-1, 1)
            values = values.view(-1, 1)

            ctx = (name, numel, shape, values.dtype, indices.dtype,
                   tensor.data.view(numel))

            if self.fp16_values and values.dtype.is_floating_point:
                values = values.type(torch.float16)
            if self.int32_indices and not indices.dtype.is_floating_point:
                indices = indices.type(torch.int32)
            return (values, indices), ctx
        else:
            # keep the original tensor and it will do all reduce
            ctx = (name, None, None, tensor.dtype, None, None)
            if self.fp16_values and tensor.dtype.is_floating_point:
                tensor = tensor.type(torch.float16)
            return tensor, ctx

    def decompress(self, tensor, ctx):
        name, numel, shape, vdtype, idtype, grad = ctx
        if self.compress_ratio < 1.0 and name in self.attributes:
            # decompress
            assert isinstance(tensor, (list, tuple))
            values, indices = tensor
            values = values.view(-1)
            indices = indices.view(-1)
            if self.fp16_values and vdtype.is_floating_point:
                values = values.type(vdtype)
            if self.int32_indices and not idtype.is_floating_point:
                indices = indices.type(idtype)
            grad.zero_().index_put_([indices], values, accumulate=True)
            if self.op == Average:
                grad.mul_(1. / self.world_size)
            return grad.view(shape)
        else:
            if self.fp16_values and vdtype.is_floating_point:
                tensor = tensor.type(vdtype)
            return self.memory.compensate(tensor, name, accumulate=False)

    def communicate(self, tensor_compressed, name, op):
        self.op = op
        if self.compress_ratio < 1.0 and name in self.attributes:
            return [allgather_async_(t, name=f'{name}.t{e}')
                    for e, t in enumerate(tensor_compressed)]
        else:
            return allreduce_async_(tensor_compressed, name=name, op=op)

    def synchronize(self, handle):
        if isinstance(handle, (tuple, list)):
            return [synchronize_(h) for h in handle]
        else:
            return synchronize_(handle)
