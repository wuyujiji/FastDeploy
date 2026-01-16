import time

import paddle
from fastdeploy.model_executor.ops.iluvatar import weight_only_linear
from fastdeploy.model_executor.ops.iluvatar.utils import get_wint4_quant_func


def do_quant(weight, quant_method, format, group_size):
    if quant_method == "int8":
        wmax = weight.abs().max(axis=0)
        weight_scale = wmax / 127
        quant_weight = (
            paddle.round(weight.to(paddle.float32) / weight_scale)
            .clamp(-128, 127)
            .to(paddle.int8)
        )
        if format == "TN":
            quant_weight = quant_weight.transpose(0, 1).contiguous()
        weight_zeros = None
    else:
        quant_func = get_wint4_quant_func()
        quant_weight, weight_scale, weight_zeros = quant_func(
            weight.transpose(0, 1).contiguous(),
            group_size=group_size,
        )
    return quant_weight, weight_scale, weight_zeros


if __name__ == "__main__":
    # warmup
    test_count = 10
    act_type = "none"
    
    # int8
    quant_method = "int8"
    format = "NN" # or "TN"
    group_size = -1
    
    # wi4a16
    # FD_ILUVATAR_WINT4_QUANT_ALGO="wi4a16"
    # quant_method = "int4"
    # format = "TN"
    # group_size = 128

    # wu4a16_awq
    # FD_ILUVATAR_WINT4_QUANT_ALGO="wu4a16_awq"
    # quant_method = "int4"
    # format = "NN"
    # group_size = 32 # or -1, 64, 128
    
    # wu4a16_dotsaddz
    # FD_ILUVATAR_WINT4_QUANT_ALGO="wu4a16_dotsaddz"
    # quant_method = "int4"
    # format = "NN"
    # group_size = 32 # or -1, 64, 128

    print(f"quant_method={quant_method}, format={format}, group_size={group_size}, act_type={act_type}")

    #x = paddle.rand([8, 800, 1152], dtype="bfloat16")
    #weight = paddle.rand([1152, 1024], dtype="bfloat16")
    #bias = paddle.zeros([1024], dtype="bfloat16")

    #x = paddle.rand([8, 128], dtype="bfloat16")
    #weight = paddle.rand([128, 256], dtype="bfloat16") # only positive elements
    #weight = paddle.randn([128, 256], dtype="bfloat16") # contain negative elements

    x = paddle.rand([213, 2048], dtype="bfloat16")
    # x = paddle.rand([1, 2048], dtype="bfloat16")
    weight = paddle.rand([2048, 1024], dtype="bfloat16")
    bias = None

    quant_weight, weight_scale, weight_zeros = do_quant(weight, quant_method, format, group_size)
    print(f"x.shape={x.shape}")
    print(f"weight={weight}")
    print(f"quant_weight={quant_weight}")
    print(f"quant_weight.max()={quant_weight.to(paddle.int32).max()}, quant_weight.min()={quant_weight.to(paddle.int32).min()}")
    print(f"weight_scale.shape={weight_scale.shape}")
    print(f"weight_zeros={weight_zeros}")

    cublas_out = paddle.matmul(x, weight)
    cuinfer_out = weight_only_linear(
        x,
        quant_weight,
        bias,
        weight_scale,
        weight_zeros=weight_zeros,
        group_size=group_size,
        act_type=act_type
    )
    print(f"cublas_out={cublas_out}")
    print(f"cuinfer_out={cuinfer_out}")

    tol = 1e-02 if quant_method == "int8" else 5e-02
    assert paddle.allclose(
        cublas_out.to("cpu").to(paddle.float32),
        cuinfer_out.to("cpu").to(paddle.float32),
        rtol=tol,
        atol=tol)

    paddle.device.synchronize()
    cublas_time_start = time.time()
    for i in range(test_count):
        cublas_out = paddle.matmul(x, weight)
    paddle.device.synchronize()
    cublas_time_end = time.time()
    cublas_duration_time = (cublas_time_end - cublas_time_start) / test_count

    paddle.device.synchronize()
    cuinfer_time_start = time.time()
    for i in range(test_count):
        cuinfer_out = weight_only_linear(
            x,
            quant_weight,
            bias,
            weight_scale,
            weight_zeros=weight_zeros,
            group_size=group_size,
            act_type=act_type
        )
    paddle.device.synchronize()
    cuinfer_time_end = time.time()
    cuinfer_duration_time = (cuinfer_time_end - cuinfer_time_start) / test_count

    print(f"duration time: cublas_gemm={cublas_duration_time}, cuinfer_gemm={cuinfer_duration_time}")