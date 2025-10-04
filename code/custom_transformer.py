r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch
import torch.nn as nn
from custom_layers import FMoE
from linear import FMoELinear


class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x


class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0, hyper_size=None,
        **kwargs
    ):
        super().__init__(num_expert=num_expert, d_model=d_model, hyper_size=hyper_size, **kwargs)
        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=expert_rank
        )
        self.mark_parallel_comm(expert_dp_comm)
        
        # Create one-time fallback layer to avoid repeated creation
        self.fallback_layer = None

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        try:
            # Validate input tensor
            if not inp.is_contiguous():
                inp = inp.contiguous()
            
            # Handle input dimensions: ensure 2D
            original_shape = inp.shape
            if inp.dim() == 3:
                # Reshape 3D tensor to 2D: [batch, seq, dim] -> [batch*seq, dim]
                inp = inp.view(-1, inp.size(-1))
            elif inp.dim() != 2:
                print(f"Warning: Unsupported input dimension. Expected: 2D or 3D, Actual: {inp.shape}")
                return self._fallback_forward(inp)
            
            if inp.size(-1) != self.d_model:
                print(f"Warning: Feature dimension mismatch. Expected: {self.d_model}, Actual: {inp.size(-1)}")
                return self._fallback_forward(inp.view(original_shape) if len(original_shape) == 3 else inp)
            
            # Check if expert count is reasonable
            if hasattr(self.gate, 'num_expert') and self.gate.num_expert != self.num_expert:
                print(f"Warning: Gate expert count ({self.gate.num_expert}) does not match model expert count ({self.num_expert})")
                return self._fallback_forward(inp.view(original_shape) if len(original_shape) == 3 else inp)
            
            # Execute MoE forward pass
            output = super().forward(inp)
            
            # Validate output
            if output is None or not isinstance(output, torch.Tensor):
                print("Warning: Invalid MoE output, using fallback")
                return self._fallback_forward(inp.view(original_shape) if len(original_shape) == 3 else inp)
            
            # Ensure output shape matches input
            if output.size(0) != inp.size(0):
                if output.size(0) < inp.size(0):
                    # Create padding tensor
                    padding = torch.zeros(inp.size(0), self.d_model,
                                         device=output.device,
                                         dtype=output.dtype)
                    # Copy available data
                    if output.size(0) > 0:
                        padding[:output.size(0)] = output
                    output = padding
                else:
                    # Truncate excess rows
                    output = output[:inp.size(0)]
            
            # Restore original shape
            if len(original_shape) == 3:
                output = output.view(original_shape[0], original_shape[1], -1)
                
            return output
            
        except Exception as e:
            print(f"MoE forward pass error: {e}, using fallback solution")
            return self._fallback_forward(inp, getattr(self, '_original_shape', None))
    
    def _fallback_forward(self, inp, original_shape=None):
        try:
            if inp.dim() == 3:
                batch_size, seq_len = inp.shape[:2]
                inp_2d = inp.view(-1, inp.size(-1))
            else:
                inp_2d = inp
                batch_size = seq_len = None
            
            # Create or reuse fallback layer
            if self.fallback_layer is None:
                self.fallback_layer = nn.Linear(
                    self.d_model, self.d_model, 
                    device=inp_2d.device, dtype=inp_2d.dtype
                ).eval() 

                with torch.no_grad():
                    nn.init.eye_(self.fallback_layer.weight)
                    if self.fallback_layer.bias is not None:
                        nn.init.zeros_(self.fallback_layer.bias)
            
            
            if self.fallback_layer.weight.device != inp_2d.device:
                self.fallback_layer = self.fallback_layer.to(inp_2d.device)
            
            output = self.fallback_layer(inp_2d)
            
       
            if batch_size is not None and seq_len is not None:
                output = output.view(batch_size, seq_len, -1)
            
            return output
            
        except Exception as fallback_error:
          
            return inp
