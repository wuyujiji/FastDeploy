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

std::vector<paddle::Tensor> WeightOnlyGemm(
    const paddle::Tensor& x,
    const paddle::Tensor& weight,
    const paddle::Tensor& weight_scale,
    const paddle::optional<paddle::Tensor>& bias,
    const std::string& weight_dtype,
    const std::string& format,
    const int group_size) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));
  auto stream = static_cast<const cudaStream_t>(dev_ctx->stream());
  const auto& x_dims = x.dims();
  const auto& w_dims = weight.dims();
  const auto& ws_dims = weight_scale.dims();

  // [m, k]
  PD_CHECK(x_dims.size() == 2, "x should be 2D");
  // format="NN": [k, n]
  // format="TN": [n, k]
  PD_CHECK(w_dims.size() == 2, "weight should be 2D");
  // [1, n]
  PD_CHECK(ws_dims.size() == 2, "weight_scale should be 2D");
  if (bias) {
     // [n]
     PD_CHECK(bias.get().dims().size() == 1, "weight should be 1D");
     PD_CHECK(bias.get().is_contiguous());
  }

  PD_CHECK(x.dtype() == paddle::DataType::BFLOAT16 ||
           x.dtype() == paddle::DataType::FLOAT16);
  PD_CHECK(weight.dtype() == paddle::DataType::INT8);
  PD_CHECK(weight_scale.dtype() == x.dtype());
  PD_CHECK(x.is_contiguous());
  PD_CHECK(weight.is_contiguous());
  PD_CHECK(weight_scale.is_contiguous());

  int64_t m = x_dims[0];
  int64_t k = x_dims[1];
  int64_t n;
  cuinferOperation_t transa;
  int lda;
  if (format == "NN") {
      n = w_dims[1];
      PD_CHECK(w_dims[0] == k);
      transa = CUINFER_OP_N;
      lda = n;
  } else {
      // TN
      n = w_dims[0];
      PD_CHECK(w_dims[1] == k);
      transa = CUINFER_OP_T;
      lda = k;
  }
  cuinferOperation_t transb = CUINFER_OP_N;
  int ldb = k;
  int ldc = n;

  PD_CHECK(ws_dims[0] == 1);
  PD_CHECK(ws_dims[1] == n);
  
  auto output = GetEmptyTensor({m, n}, x.dtype(), x.place());

  void* out_data = output.data();
  const void* x_data = x.data();
  const void* weight_data = weight.data();
  const void* bias_data = bias ? bias.get().data() : nullptr;
  const void* weight_scale_data = weight_scale.data();

  cuinferHandle_t handle = iluvatar::getContextInstance()->getIxInferHandle();
  cuinferPointerMode_t cuinfer_ptr_mode = CUINFER_POINTER_MODE_HOST;
  cudaDataType_t Atype;
  if (weight_dtype == "int8") {
    Atype = CUDA_R_8I;
  } else if (weight_dtype == "int4") {
    Atype = CUDA_R_4I;
  } else {
    common::errors::InvalidArgument(
        "WeightOnlyGemm support int4 and int8 quant method now");
  }
  cudaDataType_t Btype, Ctype;
  if (x.dtype() == paddle::DataType::FLOAT16) {
    Btype = CUDA_R_16F;
  } else if (x.dtype() == paddle::DataType::BFLOAT16) {
    Btype = CUDA_R_16BF;
  } else {
    PADDLE_THROW(common::errors::Unimplemented("Unsupported input dtype."));
  }
  Ctype = Btype;
  cudaDataType_t computeType = CUDA_R_32F;
  cudaDataType_t scaleType = CUDA_R_32F;
  cuinferGEMMCustomOption_t customOption = CUINFER_BLAS_GEMM_CUSTOM_NONE;

  cuinferQuantGEMMHostParam cust_host_param;
  cust_host_param.size = sizeof(cuinferQuantGEMMHostParam);
  cust_host_param.persistent = 0;
  cust_host_param.groupSize = group_size;
  cuinferQuantGEMMDeviceParam cust_device_param;
  cust_device_param.bias = bias_data;
  cust_device_param.workspace = nullptr;
  cust_device_param.scale = weight_scale_data;

  float beta = 0.f;
  float alpha = 1.f;
  int batch_count = 1;

  size_t workspace_size = 0;
  CUINFER_CHECK(cuinferGetCustomGemmWorkspace(transa,
                                              transb,
                                              n,
                                              m,
                                              k,
                                              Atype,
                                              lda,
                                              0, // lda
                                              Btype,
                                              ldb,
                                              0, // ldb,
                                              Ctype,
                                              ldc,
                                              0, // ldc
                                              batch_count,
                                              computeType,
                                              scaleType,
                                              &workspace_size));

  auto* allocator = paddle::GetAllocator(x.place());
  phi::Allocator::AllocationPtr tmp_workspace;
  if (workspace_size > 0) {
      tmp_workspace = allocator->Allocate(workspace_size);
      cust_device_param.workspace = tmp_workspace->ptr();
  }

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
                                  &cust_host_param,
                                  &cust_device_param,
                                  customOption));
  return {output};
}

std::vector<std::vector<int64_t>> WeightOnlyGemmInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& weight_scale_shape,
    const paddle::optional<std::vector<int64_t>>& bias_shape,
    const std::string& weight_dtype,
    const std::string& format,
    const int group_size) {
  int64_t m, n;
  if (format == "NN") {
      // x: [m, k], weight: [k, n]
      m = x_shape[0];
      n = weight_shape[1];
  } else {
      // format == "TN"
      // x: [m, k], weight: [n, k]
      m = x_shape[0];
      n = weight_shape[0];
  }
  return {{m, n}};
}
std::vector<paddle::DataType> WeightOnlyGemmInferDtype(
    const paddle::DataType& input_dtype) {
  return {input_dtype};
}

PD_BUILD_STATIC_OP(weight_only_gemm)
    .Inputs({"x",
             "weight",
             "weight_scale",
             paddle::Optional("bias")})
    .Outputs({"output"})
    .Attrs({"weight_dtype:std::string", "format:std::string", "group_size:int"})
    .SetKernelFn(PD_KERNEL(WeightOnlyGemm))
    .SetInferShapeFn(PD_INFER_SHAPE(WeightOnlyGemmInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(WeightOnlyGemmInferDtype));
