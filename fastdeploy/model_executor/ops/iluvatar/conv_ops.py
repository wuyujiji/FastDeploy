# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Iluvatar convolution custom operators."""

from typing import Optional, Sequence, Union

import paddle

try:
    from fastdeploy.model_executor.ops.iluvatar import cuinfer_conv2d
except ImportError:
    cuinfer_conv2d = None


def _pair(value: Union[int, Sequence[int]], name: str) -> list:
    if isinstance(value, int):
        return [value, value]
    value = list(value)
    assert len(value) == 2, f"{name} must contain 2 values"
    return [int(value[0]), int(value[1])]


def conv2d(
    input: paddle.Tensor,
    weight: paddle.Tensor,
    bias: Optional[paddle.Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    groups: int = 1,
    channel_last: bool = False,
) -> paddle.Tensor:
    """Run cuinfer float16/bfloat16 conv2d on Iluvatar GPU.

    For NCHW, weight layout is OIHW. For NHWC, weight layout is OHWI.
    """
    if cuinfer_conv2d is None:
        raise RuntimeError("cuinfer_conv2d custom operator is not available")
    return cuinfer_conv2d(
        input,
        weight,
        bias,
        _pair(stride, "stride"),
        _pair(padding, "padding"),
        _pair(dilation, "dilation"),
        groups,
        channel_last,
    )
