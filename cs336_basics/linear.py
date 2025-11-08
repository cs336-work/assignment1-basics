import math
from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn, empty

class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        # initialize weights (W)
        self.W: Float[Tensor, " d_in d_out"] = nn.Parameter(empty((in_features, out_features), device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3.0*std, b=3.0*std)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.W, "... i, i o -> ... o")