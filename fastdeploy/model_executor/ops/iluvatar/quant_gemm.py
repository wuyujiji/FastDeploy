from typing import Optional

import paddle

try:
    from fastdeploy.model_executor.ops.iluvatar import weight_only_gemm
except ImportError:
    weight_only_gemm = None

def weight_only_linear(
    input: paddle.Tensor,
    weight: paddle.Tensor,
    bias: Optional[paddle.Tensor],
    weight_scale: paddle.Tensor,
    weight_dtype="int8",
    group_size=-1,
    arch=-1,
):
    input_shape= input.shape
    assert input.ndim < 4 # TODO: remove
    if input.ndim == 3:
        input = input.view([-1, input_shape[-1]])
    if input_shape[-1] == weight.shape[-1]:
        # TN
        transpose_format = "TN"
    elif input_shape[-1] == weight.shape[0]:
        # NN
        transpose_format = "NN"
    else:
        raise NotImplementedError(f"Unspport transpose format: {input.shape} vs {weight.shape}")
    output =  weight_only_gemm(
        input, weight, weight_scale, bias,
        weight_dtype, transpose_format, group_size
    )
    if len(input_shape) == 3:
        output = output.view([input_shape[0], input_shape[1], output.shape[-1]])
    return output
    
    