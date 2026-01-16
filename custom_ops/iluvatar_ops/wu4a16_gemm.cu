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

std::vector<paddle::Tensor> WU4A16Gemm(
    const paddle::Tensor& x,
    const paddle::Tensor& weight,
    const paddle::Tensor& weight_scale,
    const paddle::Tensor& weight_zeros,
    const paddle::optional<paddle::Tensor>& bias,
    const int group_size,
    const std::string& act_type,
    const std::string& quant_algo) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));
  auto stream = static_cast<const cudaStream_t>(dev_ctx->stream());
  // [m, k]
  const auto& x_dims = x.dims();
  // weight: only support NN: [k, n // 8]
  const auto& w_dims = weight.dims();
  // weight_scale: group_size=-1: [1, n], group_size > 0: [k // group_size, n]
  const auto& ws_dims = weight_scale.dims();
  // bias: [n] if not None
  const auto& zeros_dims = weight_zeros.dims();
  // weight_zeros:
  //   awq: group_size=-1: [1, n // 8], group_size > 0: [k // group_size, n // 8]
  //   dotsaddz: group_size=-1: [1, n], group_size > 0: [k // group_size, n]

  PD_CHECK(x_dims.size() == 2);
  PD_CHECK(x.dtype() == paddle::DataType::BFLOAT16 ||
           x.dtype() == paddle::DataType::FLOAT16);
  PD_CHECK(x.is_contiguous());

  PD_CHECK(w_dims.size() == 2);
  PD_CHECK(weight.dtype() == paddle::DataType::INT32);
  PD_CHECK(weight.is_contiguous());

  PD_CHECK(ws_dims.size() == 2);
  PD_CHECK(weight_scale.dtype() == x.dtype());
  PD_CHECK(weight_scale.is_contiguous());

  PD_CHECK(zeros_dims.size() == 2);
  PD_CHECK(weight_zeros.is_contiguous());

  if (bias) {
     PD_CHECK(bias.get().dims().size() == 1, "bias should be 1D");
     PD_CHECK(bias.get().is_contiguous());
  }

  PD_CHECK(group_size == -1 || group_size == 32 || group_size == 64 || group_size == 128);
  PD_CHECK(quant_algo == "awq" || quant_algo == "dotsaddz");

  int64_t m = x_dims[0];
  int64_t k = x_dims[1];
  int64_t n = ws_dims[1];
  PD_CHECK(n % 8 == 0 && k % 2 == 0);

  PD_CHECK(w_dims[0] == k);
  PD_CHECK(w_dims[1] == n / 8);

  if (group_size == -1) {
    PD_CHECK(ws_dims[0] == 1);
    PD_CHECK(zeros_dims[0] == 1);
  } else {
    // group_size > 0
    PD_CHECK(k % group_size == 0);
    PD_CHECK(ws_dims[0] == k / group_size);
    PD_CHECK(zeros_dims[0] == k / group_size);
  }
  if (quant_algo == "awq") {
    PD_CHECK(zeros_dims[1] == n / 8);
    PD_CHECK(weight_zeros.dtype() == paddle::DataType::INT32);
  } else {
    // "dotsaddz"
    PD_CHECK(zeros_dims[1] == n);
    PD_CHECK(weight_zeros.dtype() == x.dtype());
  }

  cuinferOperation_t transa = CUINFER_OP_N;
  cuinferOperation_t transb = CUINFER_OP_N;
  int lda = n;
  int ldb = k;
  int ldc = n;
  
  auto output = GetEmptyTensor({m, n}, x.dtype(), x.place());

  void* out_data = output.data();
  const void* x_data = x.data();
  const void* weight_data = weight.data();
  const void* bias_data = bias ? bias.get().data() : nullptr;
  const void* weight_scale_data = weight_scale.data();
  const void* weight_zeros_data = weight_zeros.data();

  cuinferHandle_t handle = iluvatar::getContextInstance()->getIxInferHandle();
  cuinferPointerMode_t cuinfer_ptr_mode = CUINFER_POINTER_MODE_HOST;
  cudaDataType_t Atype = CUDA_R_4U;
  cudaDataType_t Btype, Ctype;
  if (x.dtype() == paddle::DataType::FLOAT16) {
    Btype = CUDA_R_16F;
  } else {
    // x.dtype() == paddle::DataType::BFLOAT16
    Btype = CUDA_R_16BF;
  }
  Ctype = Btype;
  cudaDataType_t computeType = CUDA_R_32F;
  cudaDataType_t scaleType = CUDA_R_32F;
  cuinferGEMMCustomOption_t customOption;

  if (bias_data != nullptr) {
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
    if (act_type == "silu") {
      customOption = CUINFER_BLAS_GEMM_CUSTOM_SILU;
    } else {
      customOption = CUINFER_BLAS_GEMM_CUSTOM_NONE;
    }
  }

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
                                              0,
                                              Btype,
                                              ldb,
                                              0,
                                              Ctype,
                                              ldc,
                                              0,
                                              batch_count,
                                              computeType,
                                              scaleType,
                                              &workspace_size));
  auto* allocator = paddle::GetAllocator(x.place());
  phi::Allocator::AllocationPtr tmp_workspace;
  tmp_workspace = allocator->Allocate(workspace_size);

  cuinferQuantGEMMHostParam cust_host_param;
  cuinferCustomGemmHostParamInit(&cust_host_param);
  cust_host_param.size = sizeof(cuinferQuantGEMMHostParam);
  cust_host_param.persistent = 0;
  cust_host_param.groupSize = group_size;
  // 0: awq, 1: gptq, 2: awq_group_gemm, 3: i4 variant, 4: dotsaddz
  if (quant_algo == "awq") {
    cust_host_param.type = 0;
  } else {
    // dotsaddz
    cust_host_param.type = 4;
  }

  cuinferQuantGEMMDeviceParam cust_device_param;
  cuinferCustomGemmDeviceParamInit(&cust_device_param);
  cust_device_param.bias = bias_data;
  cust_device_param.workspace = tmp_workspace->ptr();
  cust_device_param.scale = weight_scale_data;
  cust_device_param.zero = weight_zeros_data;

  if (m <= 1) {
    CUINFER_CHECK(cuinferCustomGemmEx(handle,
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
                                      0,
                                      x_data,
                                      Btype,
                                      ldb,
                                      0,
                                      &beta,
                                      out_data,
                                      Ctype,
                                      ldc,
                                      0,
                                      batch_count,
                                      computeType,
                                      scaleType,
                                      &cust_host_param,
                                      &cust_device_param,
                                      customOption,
                                      cust_device_param.workspace));
  } else {
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
                                    0,
                                    x_data,
                                    Btype,
                                    ldb,
                                    0,
                                    &beta,
                                    out_data,
                                    Ctype,
                                    ldc,
                                    0,
                                    batch_count,
                                    computeType,
                                    scaleType,
                                    &cust_host_param,
                                    &cust_device_param,
                                    customOption));
  }
  return {output};
}

std::vector<std::vector<int64_t>> WU4A16GemmInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& weight_scale_shape) {
   // x: [m, k], weight: [k, n // 8], weight_scale: [n]
  return {{x_shape[0], weight_scale_shape[0]}};
}
std::vector<paddle::DataType> WU4A16GemmInferDtype(
    const paddle::DataType& input_dtype) {
  return {input_dtype};
}

PD_BUILD_STATIC_OP(wu4a16_gemm)
    .Inputs({"x",
             "weight",
             "weight_scale",
             "weight_zeros",
             paddle::Optional("bias")})
    .Outputs({"output"})
    .Attrs({"group_size:int",
            "act_type:std::string",
            "quant_algo:std::string"})
    .SetKernelFn(PD_KERNEL(WU4A16Gemm))
    .SetInferShapeFn(PD_INFER_SHAPE(WU4A16GemmInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(WU4A16GemmInferDtype));