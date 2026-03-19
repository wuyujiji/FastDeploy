import paddle
import os


# choices: ("wu4a16_awq", "wu4a16_dotsaddz", "wi4a16")
wint4_quant_algo = os.environ.get("FD_ILUVATAR_WINT4_QUANT_ALGO", "wu4a16_awq")


def get_wint4_quant_func():
    if  wint4_quant_algo == "wu4a16_awq":
        quant_func = wu4a16_awq_quantize
    elif wint4_quant_algo == "wu4a16_dotsaddz":
        quant_func = wu4a16_dotsadd_quantize
    elif wint4_quant_algo == "wi4a16":
        quant_func = wi4a16_quantize
    else:
        raise NotImplementedError(
            "only support wu4a16_awq, wu4a16_dotsaddz or wi4a16 for wint4"
        )
    return quant_func


def _get_weight_by_group_size(w, group_size):
    assert w.dim() == 2
    assert group_size in (-1, 32, 64, 128)
    if group_size == -1:
        quant_weight = w
    else:
        assert w.shape[-1] % group_size == 0
        quant_weight = w.reshape(-1, group_size)
    assert paddle.isnan(quant_weight).sum() == 0
    return quant_weight


def _pack_int4_to_int32(weight, n_bit, order_map):
    N, K = weight.shape
    pack_num = 32 // n_bit
    packed_K = K // pack_num

    pack_int_weight = paddle.zeros((N, packed_K), dtype=paddle.int32)

    for col in range(packed_K):
        for i in range(pack_num):
            qweight_col = weight[:, col * pack_num + order_map[i]]
            pack_int_weight[:, col] |= qweight_col << (i * n_bit)

    return pack_int_weight


def wu4a16_awq_quantize(w, group_size=-1, n_bit=4, order_map=None):
    # only support NN
    # [k, n] -> [n, k]
    w = w.T.contiguous()
    quant_weight = _get_weight_by_group_size(w, group_size)
    max_val = quant_weight.max(axis=1, keepdim=True)
    min_val = quant_weight.min(axis=1, keepdim=True)
    max_int = 2 ** n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    assert paddle.isnan(scales).sum() == 0

    zeros = (-paddle.round(min_val / scales)).clamp_(min_int, max_int)

    out = paddle.clamp(paddle.round(quant_weight / scales) + zeros, min_int, max_int)
    assert paddle.isnan(out).sum() == 0

    if order_map is None:
        order_map = [0, 2, 4, 6, 1, 3, 5, 7]

    out = _pack_int4_to_int32(
        out.to(dtype=paddle.int32).view(w.shape[0], -1).T.contiguous(),
        n_bit, order_map=order_map
    )

    scales = scales.view(w.shape[0], -1).T.contiguous()
    zeros = _pack_int4_to_int32(
        zeros.to(dtype=paddle.int32).view(w.shape[0], -1).T.contiguous(),
        n_bit, order_map=[0, 1, 2, 3, 4, 5, 6, 7]
    )

    return out, scales, zeros


def wu4a16_dotsadd_quantize(w, group_size=32, n_bit=4, order_map=None):
    # only support NN
    # [k, n] -> [n, k]
    w = w.T.contiguous()
    quant_weight = _get_weight_by_group_size(w, group_size)

    max_val = quant_weight.max(axis=1, keepdim=True)
    min_val = quant_weight.min(axis=1, keepdim=True)
    max_int = 2 ** n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    assert paddle.isnan(scales).sum() == 0

    zeros = min_val
    assert paddle.isnan(zeros).sum() == 0

    out = quant_weight.sub(min_val).div(scales).round().clamp_(min_int, max_int)
    assert paddle.isnan(out).sum() == 0

    if order_map is None:
        order_map = [0, 1, 2, 3, 4, 5, 6, 7]

    out = _pack_int4_to_int32(
        out.to(dtype=paddle.int32).view(w.shape[0], -1).T.contiguous(),
        n_bit, order_map=order_map
    )

    scales = scales.view(w.shape[0], -1).T.contiguous()
    zeros = zeros.view(w.shape[0], -1).T.contiguous()

    return out, scales, zeros


def _pack_int4_to_int8(weight):
    return ((weight[:, 1::2] & 0xF) << 4) | (weight[:, 0::2] & 0xF)


def wi4a16_quantize(w, group_size=128):
    # only support TN
    # [k, n] -> [n, k]
    w = w.T.contiguous()
    assert group_size == 128
    quant_weight = _get_weight_by_group_size(w, group_size)

    wmax = quant_weight.abs().max(axis=1, keepdim=True)
    scales = wmax / 7
    out = (
        paddle.round(quant_weight.to(paddle.float32) / scales)
        .clamp(-8, 7)
        .to(paddle.int8)
    )

    out = _pack_int4_to_int8(
        # NOTE: conver to numpy since paddle cannot support &
        out.view(w.shape[0], -1).T.contiguous().cpu().numpy(),
    )
    out = paddle.from_numpy(out).T.contiguous()

    scales = scales.view(w.shape[0], -1).T.contiguous()
    zeros = paddle.zeros_like(scales)
    return out, scales, zeros