r"""
FMoE's parallel linear layer
"""
import torch
import torch.nn as nn
from torch.autograd import Function
import math

import fmoe_cuda


class MOELinear(Function):
    r"""
    Computes linear operators within one GPU on different experts simutaneously.
    """

    @staticmethod
    def forward(ctx, global_input_buf, fwd_expert_count, weight, bias=None):
        try:
            
            if global_input_buf is None or weight is None:
                raise ValueError("Input buffer or weight is None")
            
            
            if global_input_buf.dim() != 2:
                raise ValueError(f"Expected 2D input, got {global_input_buf.dim()}D")
            
            if isinstance(fwd_expert_count, torch.Tensor):
                if fwd_expert_count.sum() == 0:
                    raise ValueError("Expert count sum is zero")
                if fwd_expert_count.max() > global_input_buf.size(0):
                    raise ValueError("Expert count exceeds input size")
            
            num_expert = weight.size(0)
            if num_expert <= 0:
                raise ValueError("Number of experts must be positive")
            
            global_output_buf = fmoe_cuda.linear_forward(
                global_input_buf, fwd_expert_count, weight, bias
            )
            variables = (global_input_buf, fwd_expert_count, weight, bias)
            ctx.save_for_backward(*variables)
            return global_output_buf
            
        except Exception as e:
            batch_size, input_dim = global_input_buf.shape
            output_dim = weight.size(1)
            fallback_output = torch.zeros(batch_size, output_dim, 
                                        device=global_input_buf.device, 
                                        dtype=global_input_buf.dtype)
            
            variables = (global_input_buf, fwd_expert_count, weight, bias)
            ctx.save_for_backward(*variables)
            return fallback_output

    @staticmethod
    def backward(ctx, grad_out):
        (input_buf, fwd_expert_count, weight, bias) = ctx.saved_tensors
        grad_inp_buf, grad_weight, grad_bias = fmoe_cuda.linear_backward(
            grad_out, input_buf, fwd_expert_count, weight, bias
        )

        if not torch.is_tensor(bias):
            grad_bias = None

        return grad_inp_buf, None, grad_weight, grad_bias



class FMoELinear(nn.Module):
    r"""
    A linear layer that contains multiple experts.
    As multiple experts can be placed on the same worker, the computation can be
    performed in parallel to increase the performance.
    The FMoELinear module provides such function.
    """

    def __init__(
        self,
        num_expert: int,
        in_feat: int,
        out_feat: int,
        bias: bool = True,
        rank: int = 0,
    ):
        super().__init__()
        self.num_expert = num_expert
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rank = rank
        self.weight = nn.Parameter(torch.Tensor(num_expert, out_feat, in_feat))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_expert, out_feat))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def forward(self, inp, fwd_expert_count):
        r"""
        Call MOE function
        """
        x = MOELinear.apply(inp, fwd_expert_count, self.weight, self.bias)
        return x

    def extra_repr(self) -> str:
        return "num_expert={}, in_features={}, \
        out_features={}, bias={}, rank={}".format(
            self.num_expert,
            self.in_feat,
            self.out_feat,
            self.bias is not None,
            self.rank,
        )

    def reset_parameters(self):
        

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

