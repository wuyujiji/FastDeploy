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

std::vector<paddle::Tensor> PrefillFusedPagedAttnNoUpdateCache(
  const paddle::Tensor& qkv,
  const paddle::Tensor& cu_seqlens_qkv,
  const paddle::Tensor& rope_sin,
  const paddle::Tensor& rope_cos,
  int num_heads,
  int head_dim,
  int num_kv_heads,
  int max_seq_len,
  float scale,
  bool causal,
  bool q_rope,
  bool k_rope,
  bool v_rope,
  bool is_interleaved_rope_mode) {
  // check dtype and contiguous
  const auto& dtype = qkv.dtype();
  cuinferDataType_t data_type;
  if (dtype == paddle::DataType::FLOAT16) {
    data_type = CUINFER_DATA_HALF;

  } else if (dtype == paddle::DataType::BFLOAT16) {
    data_type = CUINFER_DATA_BFLOAT16;
  } else {
    common::errors::InvalidArgument(
        "paged_attention support half and bfloat16 now");
  }
  PADDLE_ENFORCE_EQ(
      cu_seqlens_qkv.dtype(),
      paddle::DataType::INT32,
      common::errors::InvalidArgument("cu_seqlens_qkv dtype must be int32"));
  PADDLE_ENFORCE_EQ(
      cu_seqlens_qkv.is_contiguous(),
      true,
      common::errors::InvalidArgument(
          "paged_attention expects cu_seqlens_qkv is contiguous"));

  const auto& qkv_dims = qkv.dims();
  PADDLE_ENFORCE_EQ(qkv_dims.size(),
                    2,
                    common::errors::InvalidArgument(
                        "paged_attn receive query dims is "
                        "[num_tokens, (num_heads+2*num_kv_heads)*head_dim]"));

  const auto& cu_seqlens_qkv_dims = cu_seqlens_qkv.dims();
  PADDLE_ENFORCE_EQ(
      cu_seqlens_qkv_dims.size(),
      1,
      common::errors::InvalidArgument(
          "paged_attn receive cu_seqlens_qkv dims is [batch_size]"));

  int batch_size = cu_seqlens_qkv_dims[0] - 1;
  int num_tokens = qkv_dims[0];
  int num_total_heads = num_heads + 2 * num_kv_heads;
  int qkv_stride = qkv.strides()[0];

  auto out = paddle::empty({num_tokens, num_heads * head_dim}, dtype, qkv.place());

  const float* rope_sin_ptr = rope_sin.data<float>();
  const float* rope_cos_ptr = rope_cos.data<float>();
  const auto& rope_dims = rope_sin.dims();
  std::vector<int> rope_shape_vec, rope_stride_vec;
  int rope_ndim;
  int batch_size_or_num_tokens = rope_dims[0];
  if (rope_dims.size() == 4) {
    // [batch_size, max_seq_len, 1, head_dim]
    PADDLE_ENFORCE_EQ(batch_size_or_num_tokens,
                      batch_size,
                      common::errors::InvalidArgument(
                          "rope_dims[0] must be equal to batch_size"));
    rope_shape_vec = std::vector<int>({batch_size, max_seq_len, head_dim});
    rope_stride_vec = std::vector<int>({max_seq_len * head_dim, head_dim, 1});
    rope_ndim = 3;
  } else if (rope_dims.size() == 3) {
    if (batch_size_or_num_tokens == max_seq_len) {
      // [max_seq_len, 1, head_dim]
      rope_shape_vec = std::vector<int>({max_seq_len, head_dim});
      rope_stride_vec = std::vector<int>({head_dim, 1});
      rope_ndim = 2;
    } else if (batch_size_or_num_tokens == num_tokens) {
      // [num_tokens, 1, head_dim]
      // NOTE: The reason for passing in the max_seq_len parameter is to allow
      // cuinfer to obtain the max_seq_len value; in actual operation, it still
      // follows the method of rope_ndim=2.
      rope_shape_vec = std::vector<int>({num_tokens, max_seq_len, head_dim});
      rope_stride_vec = std::vector<int>({max_seq_len * head_dim, head_dim, 1});
      rope_ndim = 3;
    } else {
      PD_THROW("Unsupported");
    }
  } else {
    PD_THROW("Unsupported rope_ndim = %d for Paged attn", rope_ndim);
  }

  cuinferHandle_t cuinfer_handle =
      iluvatar::getContextInstance()->getIxInferHandle();

  size_t workspace_size = 0;
  CUINFER_CHECK(cuinferGetFmhaFwdMergedFuseRopeWorkspaceSize(num_tokens,
                                                             num_heads,
                                                             num_kv_heads,
                                                             head_dim,
                                                             q_rope,
                                                             k_rope,
                                                             v_rope,
                                                             data_type,
                                                             data_type,
                                                             data_type,
                                                             &workspace_size));
  auto* allocator = paddle::GetAllocator(qkv.place());
  phi::Allocator::AllocationPtr tmp_workspace =
      allocator->Allocate(workspace_size);
  void* workspace_ptr = tmp_workspace->ptr();

  cuinferTensorDescriptor_t qkv_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&qkv_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(
      qkv_desc,
      data_type,
      3,
      std::vector<int>({num_tokens, num_total_heads, head_dim}).data(),
      std::vector<int>({num_total_heads * head_dim, head_dim, 1}).data()));

  cuinferTensorDescriptor_t qkv_seqlens_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&qkv_seqlens_desc));
  CUINFER_CHECK(
      cuinferSetTensorNdDescriptor(qkv_seqlens_desc,
                                   CUINFER_DATA_INT32,
                                   1,
                                   std::vector<int>({batch_size + 1}).data(),
                                   std::vector<int>({1}).data()));

  cuinferTensorDescriptor_t block_table_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&block_table_desc));

  cuinferTensorDescriptor_t o_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&o_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(
      o_desc,
      data_type,
      3,
      std::vector<int>({num_tokens, num_heads, head_dim}).data(),
      std::vector<int>({num_heads * head_dim, head_dim, 1}).data()));

  cuinferTensorDescriptor_t k_cache_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&k_cache_desc));
  cuinferTensorDescriptor_t v_cache_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&v_cache_desc));

  cuinferTensorDescriptor_t cos_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&cos_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(cos_desc,
                                             CUINFER_DATA_FLOAT,
                                             rope_ndim,
                                             rope_shape_vec.data(),
                                             rope_stride_vec.data()));

  cuinferTensorDescriptor_t sin_desc;
  CUINFER_CHECK(cuinferCreateTensorDescriptor(&sin_desc));
  CUINFER_CHECK(cuinferSetTensorNdDescriptor(sin_desc,
                                             CUINFER_DATA_FLOAT,
                                             rope_ndim,
                                             rope_shape_vec.data(),
                                             rope_stride_vec.data()));

  cuinferAttentionRopeMode_t rope_mode =
      is_interleaved_rope_mode ? CUINFER_ATTEN_NORMAL : CUINFER_ATTEN_OCRV1;
  CUINFER_CHECK(cuinferFmhaFwdMergedFuseRopeFunc(cuinfer_handle,
                                                 qkv_desc,
                                                 qkv.data(),
                                                 qkv_seqlens_desc,
                                                 cu_seqlens_qkv.data<int32_t>(),
                                                 block_table_desc,
                                                 nullptr,
                                                 o_desc,
                                                 out.data(),
                                                 k_cache_desc,
                                                 nullptr,
                                                 v_cache_desc,
                                                 nullptr,
                                                 workspace_ptr,
                                                 workspace_size,
                                                 cos_desc,
                                                 rope_cos_ptr,
                                                 sin_desc,
                                                 rope_sin_ptr,
                                                 batch_size,
                                                 num_heads,
                                                 num_kv_heads,
                                                 head_dim,
                                                 causal,
                                                 scale,
                                                 q_rope,
                                                 k_rope,
                                                 v_rope,
                                                 rope_mode));

  CUINFER_CHECK(cuinferDestroyTensorDescriptor(qkv_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(qkv_seqlens_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(block_table_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(o_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(k_cache_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(v_cache_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(cos_desc));
  CUINFER_CHECK(cuinferDestroyTensorDescriptor(sin_desc));

  return {out};
}

std::vector<std::vector<int64_t>> PrefillFusedPagedAttnNoUpdateCacheInferShape(
    const std::vector<int64_t>& qkv_shape,
    const std::vector<int64_t>& k_cache_shape,
    const std::vector<int64_t>& v_cache_shape,
    const std::vector<int64_t>& block_table_shape,
    const std::vector<int64_t>& cu_seqlens_qkv_shape,
    const std::vector<int64_t>& rope_sin_shape,
    const std::vector<int64_t>& rope_cos_shape,
    int num_heads,
    int head_dim,
    int num_kv_heads,
    int max_seq_len,
    float scale,
    bool causal,
    bool q_rope,
    bool k_rope,
    bool v_rope) {
  return {{qkv_shape[0], num_heads * head_dim}};
}

std::vector<paddle::DataType> PrefillFusedPagedAttnNoUpdateCacheInferDtype(
    const paddle::DataType& qkv_dtype) {
  return {qkv_dtype};
}

PD_BUILD_STATIC_OP(prefill_fused_paged_attn_without_update_cache)
    .Inputs({"qkv",
             "cu_seqlens_qkv",
             "rope_sin",
             "rope_cos"})
    .Outputs({"out"})
    .Attrs({"num_heads:int",
            "head_dim:int",
            "num_kv_heads:int",
            "max_seq_len:int",
            "scale:float",
            "causal:bool",
            "q_rope:bool",
            "k_rope:bool",
            "v_rope:bool",
            "is_interleaved_rope_mode:bool"})
    .SetKernelFn(PD_KERNEL(PrefillFusedPagedAttnNoUpdateCache))
    .SetInferShapeFn(PD_INFER_SHAPE(PrefillFusedPagedAttnNoUpdateCacheInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PrefillFusedPagedAttnNoUpdateCacheInferDtype));