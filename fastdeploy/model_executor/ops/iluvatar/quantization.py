from typing import Optional

import paddle
from .utils import wint4_quant_algo

try:
    from fastdeploy.model_executor.ops.iluvatar import (
        w8a16_gemm,
        wu4a16_gemm,
        wi4a16_gemm,
    )
except ImportError:
    w8a16_gemm = None
    wu4a16_gemm = None
    wi4a16_gemm = None


def _call_w8a16_gemm(
    input: paddle.Tensor,
    weight: paddle.Tensor,
    weight_scale: paddle.Tensor,
    bias: Optional[paddle.Tensor],
    group_size: int = -1,
    act_type: str = "none",  
):
    # input: [m, k]
    # weight: int8: "NN": [k, n], "TN": [n, k]
    # weight_scale: [n]
    # bias: [n] if not None
    # group_size=-1
    # act_type: if bias=None, choices=("none", "silu")
    #           if bias!=None: choices=("none", "gelu", "silu", "relu")

    k = input.shape[1]
    if k == weight.shape[0]:
        # NN
        transpose_format = "NN"
    elif k == weight.shape[1]:
        # TN
        transpose_format = "TN"
        assert k % 4 == 0, \
            f"only support k % 4 == 0 in w8a16 TN format, but got {k}"
    else:
        raise NotImplementedError(
            f"Unspported transpose format: {input.shape} vs {weight.shape}"
        )

    output = w8a16_gemm(
        input, weight, weight_scale, bias,
        transpose_format, group_size, act_type
    )
    return output


def _call_wu4a16_gemm(
    input: paddle.Tensor,
    weight: paddle.Tensor,
    weight_scale: paddle.Tensor,
    weight_zeros: paddle.Tensor,
    bias: Optional[paddle.Tensor],
    group_size: int = -1,
    act_type: str = "none",
    quant_algo: str = "awq",
):
    # input: [m, k]
    # weight: only support NN, [k, n // 8]
    # weight_scale: group_size=-1: [1, n], group_size > 0: [k // group_size, n]
    # bias: [n] if not None
    # weight_zeros: 
    #   awq: group_size=-1: [1, n // 8], group_size > 0: [k // group_size, n // 8]
    #   dotsaddz: group_size=-1: [1, n], group_size > 0: [k // group_size, n]
    # group_size: -1, 32, 64, 128
    # act_type: if bias=None, choices=("none", "silu")
    #           if bias!=None: choices=("none", "gelu", "silu", "relu")
    # quant_algo: "awq" or "dotsaddz"

    k = input.shape[1]
    if k != weight.shape[0]:
        # only support NN
        raise NotImplementedError(
            f"Unspported transpose format: {input.shape} vs {weight.shape}"
        )

    output = wu4a16_gemm(
        input, weight, weight_scale, weight_zeros,
        bias, group_size, act_type, quant_algo
    )
    return output


def _call_wi4a16_gemm(
    input: paddle.Tensor,
    weight: paddle.Tensor,
    weight_scale: paddle.Tensor,
    weight_zeros: paddle.Tensor,
    bias: Optional[paddle.Tensor],
    group_size: int = 128,
    act_type: str = "none",
):
    # input: [m, k]
    # weight: only support TN, [n // 2, k]
    # weight_scale: [k // group_size, n]
    # bias: [n] if not None
    # weight_zeros: [k // group_size, n]
    # group_size: 128
    # act_type: if bias=None, choices=("none", "silu")
    #           if bias!=None: choices=("none", "gelu", "silu", "relu")

    k = input.shape[1]
    if k != weight.shape[1]:
        # only support TN
        raise NotImplementedError(
            f"Unspported transpose format: {input.shape} vs {weight.shape}"
        )

    output = wi4a16_gemm(
        input, weight, weight_scale, weight_zeros,
        bias, group_size, act_type
    )
    return output
    

def weight_only_linear(
    input: paddle.Tensor,
    weight: paddle.Tensor,
    bias: Optional[paddle.Tensor],
    weight_scale: paddle.Tensor,
    weight_zeros: Optional[paddle.Tensor] = None,
    group_size: int = -1,
    act_type: str = "none", 
):
    input_orig_shape = list(input.shape)
    if len(input_orig_shape) > 2:
        input = input.view([-1, input.shape[-1]])

    quant_type = "int8" if weight_zeros is None else "int4"
    if quant_type == "int8":
        output = _call_w8a16_gemm(
            input, weight, weight_scale, bias, group_size, act_type
        )
    else:
        if wint4_quant_algo == "wi4a16":
            output = _call_wi4a16_gemm(
                input, weight, weight_scale, weight_zeros,
                bias, group_size, act_type
            ) 
        elif wint4_quant_algo in ("wu4a16_awq", "wu4a16_dotsaddz"):
            output = _call_wu4a16_gemm(
                input, weight, weight_scale, weight_zeros, bias, group_size, act_type,
                quant_algo="awq" if "awq" in wint4_quant_algo else "dotsaddz",
            )
        else:
            raise NotImplementedError(
                "only support wu4a16_awq, wu4a16_dotsaddz or wi4a16 for wint4"
            )
    
    if len(input_orig_shape) > 2:
        input_orig_shape[-1] = output.shape[-1]
        output = output.view(input_orig_shape)
    return output
