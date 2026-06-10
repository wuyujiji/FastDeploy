// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "helper.h"
#include "iluvatar_context.h"

namespace {

void CheckInputTensor(const paddle::Tensor& x, const char* name) {
  PADDLE_ENFORCE_EQ(
      x.dtype() == paddle::DataType::FLOAT16 ||
          x.dtype() == paddle::DataType::BFLOAT16,
      true,
      common::errors::InvalidArgument(
          "%s must be a float16 or bfloat16 tensor", name));
  PADDLE_ENFORCE_EQ(
      x.is_contiguous(),
      true,
      common::errors::InvalidArgument("%s must be contiguous", name));
}

cuinferDataType_t GetCuinferDataType(const paddle::DataType& dtype) {
  if (dtype == paddle::DataType::FLOAT16) {
    return CUINFER_DATA_HALF;
  }
  if (dtype == paddle::DataType::BFLOAT16) {
    return CUINFER_DATA_BFLOAT16;
  }
  PD_THROW("conv2d only supports float16 and bfloat16 tensor now");
}

void CheckAttrPair(const std::vector<int>& attr, const char* name) {
  PADDLE_ENFORCE_EQ(
      attr.size(),
      2,
      common::errors::InvalidArgument("%s must contain 2 values", name));
}

int64_t ConvOutSize(int64_t in_size,
                    int64_t pad,
                    int64_t dilation,
                    int64_t kernel,
                    int64_t stride) {
  return (in_size + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

cuinferConvolutionFwdAlgo_t SelectConvAlgo(int n,
                                           int c_in,
                                           int h_in,
                                           int w_in,
                                           int c_out,
                                           int kernel_h,
                                           int kernel_w,
                                           int stride_h,
                                           int stride_w,
                                           int dilation_h,
                                           int dilation_w) {
  if (n % 16 == 0 && h_in * w_in * c_in * static_cast<int>(sizeof(__half)) < 1048576 &&
      c_in % 32 == 0 && kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 &&
      dilation_h == 1 && dilation_w == 1 && c_out % 2 == 0) {
    return CUINFER_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  }
  return CUINFER_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
}

}  // namespace

std::vector<paddle::Tensor> Conv2d(const paddle::Tensor& input,
                                   const paddle::Tensor& weight,
                                   const paddle::optional<paddle::Tensor>& bias,
                                   const std::vector<int>& stride,
                                   const std::vector<int>& padding,
                                   const std::vector<int>& dilation,
                                   int groups,
                                   bool channel_last) {
  CheckInputTensor(input, "conv2d input");
  CheckInputTensor(weight, "conv2d weight");
  CheckAttrPair(stride, "conv2d stride");
  CheckAttrPair(padding, "conv2d padding");
  CheckAttrPair(dilation, "conv2d dilation");

  PADDLE_ENFORCE_EQ(
      weight.dtype(),
      input.dtype(),
      common::errors::InvalidArgument(
          "conv2d input and weight must have the same dtype"));

  PADDLE_ENFORCE_EQ(input.dims().size(),
                    4,
                    common::errors::InvalidArgument(
                        "conv2d input must be a 4-D tensor"));
  PADDLE_ENFORCE_EQ(weight.dims().size(),
                    4,
                    common::errors::InvalidArgument(
                        "conv2d weight must be a 4-D tensor"));
  PADDLE_ENFORCE_GT(groups,
                    0,
                    common::errors::InvalidArgument(
                        "conv2d groups must be greater than 0"));

  const auto& input_dims = input.dims();
  const auto& weight_dims = weight.dims();
  const int n = input_dims[0];
  const int c_in = channel_last ? input_dims[3] : input_dims[1];
  const int h_in = channel_last ? input_dims[1] : input_dims[2];
  const int w_in = channel_last ? input_dims[2] : input_dims[3];
  const int c_out = weight_dims[0];
  const int kernel_h = channel_last ? weight_dims[1] : weight_dims[2];
  const int kernel_w = channel_last ? weight_dims[2] : weight_dims[3];
  const int weight_c_in = channel_last ? weight_dims[3] : weight_dims[1];

  PADDLE_ENFORCE_EQ(
      c_in % groups,
      0,
      common::errors::InvalidArgument("conv2d input channels must be divisible by groups"));
  PADDLE_ENFORCE_EQ(
      weight_c_in,
      c_in / groups,
      common::errors::InvalidArgument(
          "conv2d weight input channels must equal input channels / groups"));

  const int pad_h = padding[0];
  const int pad_w = padding[1];
  const int stride_h = stride[0];
  const int stride_w = stride[1];
  const int dilation_h = dilation[0];
  const int dilation_w = dilation[1];
  const int h_out = ConvOutSize(h_in, pad_h, dilation_h, kernel_h, stride_h);
  const int w_out = ConvOutSize(w_in, pad_w, dilation_w, kernel_w, stride_w);

  PADDLE_ENFORCE_GT(h_out,
                    0,
                    common::errors::InvalidArgument(
                        "conv2d output height must be greater than 0"));
  PADDLE_ENFORCE_GT(w_out,
                    0,
                    common::errors::InvalidArgument(
                        "conv2d output width must be greater than 0"));

  std::vector<int64_t> output_shape = channel_last
                                          ? std::vector<int64_t>{n, h_out, w_out, c_out}
                                          : std::vector<int64_t>{n, c_out, h_out, w_out};
  auto output = paddle::empty(output_shape, input.dtype(), input.place());

  cuinferHandle_t handle = iluvatar::getContextInstance()->getIxInferHandle();
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(input.place()));
  auto stream = static_cast<const cudaStream_t>(dev_ctx->stream());
  CUINFER_CHECK(cuinferSetStream(handle, stream));

  const auto layout = channel_last ? CUINFER_TENSOR_NHWC : CUINFER_TENSOR_NCHW;
  const auto data_type = GetCuinferDataType(input.dtype());
  const auto convolution_algorithm = SelectConvAlgo(n,
                                                    c_in,
                                                    h_in,
                                                    w_in,
                                                    c_out,
                                                    kernel_h,
                                                    kernel_w,
                                                    stride_h,
                                                    stride_w,
                                                    dilation_h,
                                                    dilation_w);

  cuinferTensorDescriptor_t input_descriptor;
  cuinferConvolutionDescriptor_t convolution_descriptor;
  cuinferFilterDescriptor_t kernel_descriptor;
  cuinferTensorDescriptor_t bias_descriptor;
  cuinferActivationDescriptor_t activation_descriptor;
  cuinferTensorDescriptor_t output_descriptor;

  CUINFER_CHECK(cuinferCreateTensorDescriptor(&input_descriptor));
  CUINFER_CHECK(cuinferCreateConvolutionDescriptor(&convolution_descriptor));
  CUINFER_CHECK(cuinferCreateFilterDescriptor(&kernel_descriptor));
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&bias_descriptor));
  CUINFER_CHECK(cuinferCreateActivationDescriptor(&activation_descriptor));
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&output_descriptor));

  CUINFER_CHECK(cuinferSetTensor4dDescriptor(input_descriptor,
                                             layout,
                                             data_type,
                                             n,
                                             c_in,
                                             h_in,
                                             w_in));
  CUINFER_CHECK(cuinferSetConvolution2dDescriptor(convolution_descriptor,
                                                  pad_h,
                                                  pad_w,
                                                  stride_h,
                                                  stride_w,
                                                  dilation_h,
                                                  dilation_w,
                                                  CUINFER_CROSS_CORRELATION,
                                                  CUINFER_DATA_FLOAT));
  CUINFER_CHECK(cuinferSetFilter4dDescriptor(kernel_descriptor,
                                             data_type,
                                             layout,
                                             c_out,
                                             c_in / groups,
                                             kernel_h,
                                             kernel_w));
  CUINFER_CHECK(cuinferSetTensor4dDescriptor(
      bias_descriptor, layout, CUINFER_DATA_FLOAT, 1, c_out, 1, 1));
  CUINFER_CHECK(cuinferSetActivationDescriptor(activation_descriptor,
                                               CUINFER_ACTIVATION_IDENTITY,
                                               CUINFER_NOT_PROPAGATE_NAN,
                                               0));
  CUINFER_CHECK(cuinferSetTensor4dDescriptor(output_descriptor,
                                             layout,
                                             data_type,
                                             n,
                                             c_out,
                                             h_out,
                                             w_out));
  CUINFER_CHECK(cuinferSetConvolutionGroupCount(convolution_descriptor, groups));

  size_t workspace_bytes = 0;
  CUINFER_CHECK(cuinferGetConvolutionForwardWorkspaceSize(handle,
                                                          input_descriptor,
                                                          kernel_descriptor,
                                                          convolution_descriptor,
                                                          output_descriptor,
                                                          convolution_algorithm,
                                                          &workspace_bytes));
  paddle::Tensor workspace =
      workspace_bytes == 0
          ? paddle::Tensor()
          : paddle::empty({static_cast<int64_t>(workspace_bytes)},
                          paddle::DataType::INT8,
                          input.place());

  const float* bias_ptr = nullptr;
  if (bias) {
    const auto& bias_tensor = bias.get();
    PADDLE_ENFORCE_EQ(
        bias_tensor.dtype(),
        paddle::DataType::FLOAT32,
        common::errors::InvalidArgument("conv2d bias must be a float32 tensor"));
    PADDLE_ENFORCE_EQ(
        bias_tensor.is_contiguous(),
        true,
        common::errors::InvalidArgument("conv2d bias must be contiguous"));
    PADDLE_ENFORCE_EQ(
        bias_tensor.dims().size(),
        1,
        common::errors::InvalidArgument("conv2d bias must be a 1-D tensor"));
    PADDLE_ENFORCE_EQ(
        bias_tensor.dims()[0],
        c_out,
        common::errors::InvalidArgument(
            "conv2d bias length must equal output channels"));
    bias_ptr = bias_tensor.data<float>();
  }

  float alpha1 = 1.0f;
  float alpha2 = 0.0f;
  float beta = 0.0f;
  float gamma = 0.0f;
  bool connection_before_activation = false;
  cuinferTensorConnectionMode_t connection_desc =
      static_cast<cuinferTensorConnectionMode_t>(0);
  int* workspace_ptr =
      workspace_bytes == 0
          ? nullptr
          : reinterpret_cast<int*>(const_cast<void*>(workspace.data()));
  __half* input_ptr = reinterpret_cast<__half*>(const_cast<void*>(input.data()));
  __half* weight_ptr =
      reinterpret_cast<__half*>(const_cast<void*>(weight.data()));
  __half* output_ptr = reinterpret_cast<__half*>(output.data());

  CUINFER_CHECK(cuinferHalfConvolution2dForward(
      handle,
      &alpha1,
      &beta,
      &gamma,
      input_descriptor,
      input_ptr,
      kernel_descriptor,
      weight_ptr,
      convolution_descriptor,
      convolution_algorithm,
      workspace_ptr,
      workspace_bytes,
      &alpha2,
      output_descriptor,
      output_ptr,
      bias_descriptor,
      const_cast<float*>(bias_ptr),
      activation_descriptor,
      connection_before_activation,
      connection_desc,
      output_descriptor,
      output_ptr));

  CUINFER_CHECK(cuinferDestroyTensorDescriptor(input_descriptor));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(output_descriptor));
  CUINFER_CHECK(cuinferDestroyConvolutionDescriptor(convolution_descriptor));
  CUINFER_CHECK(cuinferDestroyFilterDescriptor(kernel_descriptor));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(bias_descriptor));
  CUINFER_CHECK(cuinferDestroyActivationDescriptor(activation_descriptor));
  return {output};
}

std::vector<std::vector<int64_t>> Conv2dInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& weight_shape,
    const paddle::optional<std::vector<int64_t>>& bias_shape,
    const std::vector<int>& stride,
    const std::vector<int>& padding,
    const std::vector<int>& dilation,
    int groups,
    bool channel_last) {
  CheckAttrPair(stride, "conv2d stride");
  CheckAttrPair(padding, "conv2d padding");
  CheckAttrPair(dilation, "conv2d dilation");
  const int64_t n = input_shape[0];
  const int64_t c_out = weight_shape[0];
  const int64_t h_in = channel_last ? input_shape[1] : input_shape[2];
  const int64_t w_in = channel_last ? input_shape[2] : input_shape[3];
  const int64_t kernel_h = channel_last ? weight_shape[1] : weight_shape[2];
  const int64_t kernel_w = channel_last ? weight_shape[2] : weight_shape[3];
  const int64_t h_out = ConvOutSize(h_in, padding[0], dilation[0], kernel_h, stride[0]);
  const int64_t w_out = ConvOutSize(w_in, padding[1], dilation[1], kernel_w, stride[1]);
  if (channel_last) {
    return {{n, h_out, w_out, c_out}};
  }
  return {{n, c_out, h_out, w_out}};
}

std::vector<paddle::DataType> Conv2dInferDtype(
    const paddle::DataType& input_dtype,
    const paddle::DataType& weight_dtype,
    const paddle::optional<paddle::DataType>& bias_dtype) {
  return {input_dtype};
}

PD_BUILD_STATIC_OP(cuinfer_conv2d)
    .Inputs({"input", "weight", paddle::Optional("bias")})
    .Outputs({"output"})
    .Attrs({"stride: std::vector<int>",
            "padding: std::vector<int>",
            "dilation: std::vector<int>",
            "groups: int",
            "channel_last: bool"})
    .SetKernelFn(PD_KERNEL(Conv2d))
    .SetInferShapeFn(PD_INFER_SHAPE(Conv2dInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(Conv2dInferDtype));
