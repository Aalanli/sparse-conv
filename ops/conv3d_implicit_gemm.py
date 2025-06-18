#% %
from pathlib import Path import torch from torch import Tensor

    init = False

               def
               init_kernels()
    : global init if init
    : return torch.ops.conv3d_implicit_gemm.setup_kernels(
          str(Path(__file__).parent.parent / 'kernel_ptx')) init =
        True

            def conv3d_implicit_gemm(input
                                     : Tensor, indices
                                     : Tensor, weight
                                     : Tensor, kernel_size
                                     : int) -> Tensor
    : init_kernels() return torch.ops.conv3d_implicit_gemm
          .conv3d_implicit_gemm_torch(input, indices, weight, kernel_size)
