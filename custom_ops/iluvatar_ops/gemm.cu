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

#include "helper.h"
#include "iluvatar_context.h"

std::vector<paddle::Tensor> Gemm(
    const paddle::Tensor& x,
    const paddle::Tensor& weight,
    const paddle::optional<paddle::Tensor>& bias,
    const std::string& act_type) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));
  auto stream = static_cast<const cudaStream_t>(dev_ctx->stream());
  const auto& x_dims = x.dims();
  const auto& w_dims = weight.dims();

  // [..., k]
  PD_CHECK(x_dims.size() >= 2, "x should be larger than 2D");
  // [k, n]
  PD_CHECK(w_dims.size() == 2, "weight should be 2D");
  if (bias) {
     // [n]
     PD_CHECK(bias.get().dims().size() == 1, "weight should be 1D");
     PD_CHECK(bias.get().is_contiguous());
  }

  auto k = w_dims[0];
  auto n = w_dims[1];
  size_t m = 1;
  std::vector<int64_t> output_shape_vec;
  for (int i = 0; i < x_dims.size() - 1; ++i) {
      m *= x_dims[i];
      output_shape_vec.push_back(x_dims[i]);
  }
  output_shape_vec.push_back(n);
  common::DDim output_shape(output_shape_vec.data(), output_shape_vec.size());
  auto output = GetEmptyTensor(output_shape, x.dtype(), x.place());

  PD_CHECK(x_dims[x_dims.size()-1] == k);

  PD_CHECK(x.dtype() == paddle::DataType::BFLOAT16 ||
           x.dtype() == paddle::DataType::FLOAT16);
//   PD_CHECK(weight.dtype() == paddle::DataType::INT8);
  PD_CHECK(weight.dtype() == paddle::DataType::BFLOAT16 ||
           weight.dtype() == paddle::DataType::FLOAT16);
  PD_CHECK(x.dtype() == weight.dtype());
  PD_CHECK(x.is_contiguous());
  PD_CHECK(weight.is_contiguous());

  void* out_data = output.data();
  const void* x_data = x.data();
  const void* weight_data = weight.data();
  const void* bias_data = bias ? bias.get().data() : nullptr;

  cuinferHandle_t handle = iluvatar::getContextInstance()->getIxInferHandle();
  cuinferPointerMode_t cuinfer_ptr_mode = CUINFER_POINTER_MODE_HOST;
  cuinferOperation_t transa = CUINFER_OP_N;
  cuinferOperation_t transb = CUINFER_OP_N;
  cudaDataType_t Atype, Btype, Ctype;
  if (x.dtype() == paddle::DataType::FLOAT16) {
    Btype = CUDA_R_16F;
  } else if (x.dtype() == paddle::DataType::BFLOAT16) {
    Btype = CUDA_R_16BF;
  } else {
    PADDLE_THROW(common::errors::Unimplemented("Unsupported input dtype."));
  }
  Atype = Btype;
  Ctype = Btype;
  cudaDataType_t computeType = CUDA_R_32F;
  cudaDataType_t scaleType = CUDA_R_32F;
  cuinferGEMMCustomOption_t customOption;
  if (bias) {
      if (act_type == "gelu") {
          customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_GELU;
      } else if (act_type == "relu") {
          customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_RELU;
      } else if (act_type == "silu") {
          customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS_SILU;
      } else {
          customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS;
      }
  } else {
      // default CUINFER_BLAS_GEMM_CUSTOM_NONE
      customOption = CUINFER_BLAS_GEMM_CUSTOM_NONE;
      if (act_type == "silu") {
          customOption = CUINFER_BLAS_GEMM_CUSTOM_SILU;
      }
  }

  int lda = n;
  int ldb = k;
  int ldc = n;
  float beta = 0.f;
  float alpha = 1.f;
  int batch_count = 1;


  CUINFER_CHECK(cuinferCustomGemm(handle,
                                  stream,
                                  cuinfer_ptr_mode,
                                  transa,
                                  transb,
                                  n,
                                  m,
                                  k,
                                  &alpha,
                                  weight_data,
                                  Atype,
                                  lda,
                                  0, // lda
                                  x_data,
                                  Btype,
                                  ldb,
                                  0, // ldb
                                  &beta,
                                  out_data,
                                  Ctype,
                                  ldc,
                                  0, // ldc
                                  batch_count,
                                  computeType,
                                  scaleType,
                                  nullptr, // custom_params
                                  (void*)bias_data,
                                  customOption));
  return {output};
}

std::vector<std::vector<int64_t>> GemmInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& weight_shape) {
  return {{x_shape[0], weight_shape[1]}};
}
std::vector<paddle::DataType> GemmInferDtype(
    const paddle::DataType& input_dtype) {
  return {input_dtype};
}

PD_BUILD_STATIC_OP(gemm)
    .Inputs({"x",
             "weight",
             paddle::Optional("bias")})
    .Outputs({"output"})
    .Attrs({"act_type:std::string"})
    .SetKernelFn(PD_KERNEL(Gemm))
    .SetInferShapeFn(PD_INFER_SHAPE(GemmInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GemmInferDtype));

