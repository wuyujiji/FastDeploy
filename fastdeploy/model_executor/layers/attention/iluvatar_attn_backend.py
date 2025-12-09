"""
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
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING, Optional

import paddle

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.ops.iluvatar import (
    mixed_fused_paged_attention,
    paged_attention,
    prefill_fused_paged_attention,
)

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta


@dataclass
class IluvatarAttentionMetadata(AttentionMetadata):
    """
    IluvatarAttentionMetadata
    """

    alibi_slopes: Optional[paddle.Tensor] = None
    window_left: int = -1
    window_right: int = -1
    softcap: float = 0.0
    use_cuda_graph: bool = False
    use_sqrt_alibi: bool = False


# qk[seq, h, d], cos/sin [seq, 1, d]
def apply_rope(qk, cos, sin):
    #rotate_half = paddle.reshape(
    #    paddle.stack([-qk[..., 1::2], qk[..., 0::2]], axis=-1),
    #    paddle.shape(qk),
    #)

    head_dim = qk.shape[-1]
    x1 = qk[..., : head_dim // 2]
    x2 = qk[..., head_dim // 2 :]
    rotate_half = paddle.concat([-x2, x1], axis=-1)

    orig_dtype = qk.dtype
    qk = qk.astype("float32")

    out = paddle.add(paddle.multiply(qk, cos), paddle.multiply(rotate_half, sin))
    return paddle.cast(out, orig_dtype)
    #return paddle.cast(out, qk.dtype)



class IluvatarAttnBackend(AttentionBackend):
    """
    The backend class that uses paddle native attention implementation.
    Which is used only for testing purpose.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        kv_num_heads: int,
        num_heads: int,
        head_dim: int,
        encoder_block_shape_q: int = -1,
        decoder_block_shape_q: int = -1,
    ):
        super().__init__()
        self.attention_metadata = IluvatarAttentionMetadata()
        self.block_size = fd_config.cache_config.block_size
        assert self.block_size == 16, "Iluvatar paged attn requires block_size must be 16."
        self.max_context_len = fd_config.model_config.max_model_len
        self.causal = getattr(fd_config.model_config, "causal", True)
        self.speculate_method = getattr(fd_config.parallel_config, "speculate_method", None)
        self.use_speculate = self.speculate_method is not None
        self.num_kv_heads = kv_num_heads
        self.num_heads = num_heads
        self.total_num_heads = num_heads + 2 * kv_num_heads
        self.head_dim = head_dim
        self.hidden_dim = fd_config.model_config.hidden_size
        # note: scale need to change if using MLA
        self.scale = 1.0 / sqrt(head_dim)
        self.num_layers = fd_config.model_config.num_hidden_layers
        self.dtype = paddle.get_default_dtype()
        self.enable_mm = fd_config.model_config.enable_mm
        self.rope_batch_stride = self.max_context_len * self.head_dim if self.enable_mm else 0
        if "paddleocr" in fd_config.model_config.model_type:
            self.is_interleaved_rope_mode = False
        else:
            self.is_interleaved_rope_mode = True
        
        self.record_block_table_metadata = {}

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        self.prefill_info_dict = {}
        self.decode_info_dict = {}
        self.prefill_info_dict["batch_ids"] = paddle.where(forward_meta.seq_lens_encoder)[0]
        self.decode_info_dict["batch_ids"] = paddle.where(forward_meta.seq_lens_decoder)[0]
        self.prefill_len = len(self.prefill_info_dict["batch_ids"])
        self.decode_len = len(self.decode_info_dict["batch_ids"])
        if self.enable_mm:
            # the num_seqs dim of rotary_embs > 1 (e.g. ernie-vl and paddleocr-vl)
            num_seqs = self.prefill_len + self.decode_len
            self.rope_cos = forward_meta.rotary_embs[:num_seqs, 0, 0, :, :, :]
            self.rope_sin = forward_meta.rotary_embs[:num_seqs, 1, 0, :, :, :]
        else:
            #  the num_seqs dim of rotary_embs = 1 (e.g. ernie-text)
            self.rope_cos = forward_meta.rotary_embs[0, 0, :, :, :]
            self.rope_sin = forward_meta.rotary_embs[1, 0, :, :, :]
        # only prefill
        if self.decode_len == 0:
            cu_seq_ids = self.prefill_info_dict["batch_ids"] + 1
            self.prefill_info_dict["cu_seqlens_q"] = paddle.concat(
                [forward_meta.cu_seqlens_q[:1], forward_meta.cu_seqlens_q[cu_seq_ids]]
            )
            self.mixed = False
        # only decode
        elif self.prefill_len == 0:
            self.mixed = False
            
            cu_seq_ids = self.decode_info_dict["batch_ids"] + 1
            self.decode_info_dict["cu_seqlens_q"] = paddle.concat(
                [forward_meta.cu_seqlens_q[:1], forward_meta.cu_seqlens_q[cu_seq_ids]])
        # both prefill and decode
        else:
            self.mixed = True
            if self.enable_mm:
                self.prefill_rope_cos = self.rope_cos[self.prefill_info_dict["batch_ids"], :, :, :]
                self.prefill_rope_sin = self.rope_sin[self.prefill_info_dict["batch_ids"], :, :, :]
                self.decode_rope_cos = self.rope_cos[self.decode_info_dict["batch_ids"], :, :, :]
                self.decode_rope_sin = self.rope_sin[self.decode_info_dict["batch_ids"], :, :, :]
            else:
                self.prefill_rope_cos = self.decode_rope_cos = self.rope_cos
                self.prefill_rope_sin = self.decode_rope_sin = self.rope_sin
            self.prefill_num_tokens = paddle.sum(forward_meta.seq_lens_encoder).item()
            self.prefill_info_dict["cu_seqlens_q"] = paddle.zeros(
                [self.prefill_len + 1], dtype=forward_meta.cu_seqlens_q.dtype
            )
            self.prefill_info_dict["cu_seqlens_q"][1:] = forward_meta.seq_lens_encoder[
                self.prefill_info_dict["batch_ids"], 0
            ]
            # NOTE: The explicit dtype='int32' is required for Iluvatar hardware compatibility.
            self.prefill_info_dict["cu_seqlens_q"] = paddle.cumsum(
                self.prefill_info_dict["cu_seqlens_q"], dtype="int32"
            )
            self.decode_info_dict["cu_seqlens_q"] = paddle.arange(
                self.decode_len+1, dtype=forward_meta.cu_seqlens_q.dtype)

            self.tmp_buffer = paddle.zeros(
                [self.prefill_num_tokens + self.decode_len, self.hidden_dim], dtype=self.dtype
            )

            prefill_start, decode_start, start = 0, self.prefill_num_tokens, 0
            non_zeros_ids = paddle.where(forward_meta.seq_lens_this_time)[0]
            non_zeros_seq_lens = forward_meta.seq_lens_this_time[non_zeros_ids]
            end = non_zeros_seq_lens[0]
            if end > 1:
                last_stage = "prefill"
                prefill_end = end
                decode_end = decode_start
            else:
                last_stage = "decode"
                prefill_end = 0
                decode_end = decode_start + end

            self.id_group = []
            self.reverse_id_group = []
            for seq_len in non_zeros_seq_lens[1:]:
                if seq_len > 1:
                    if last_stage == "decode":
                        self.id_group.append((decode_start, decode_end))
                        self.reverse_id_group.append((start, end))
                        decode_start = decode_end
                        start = end
                        last_stage = "prefill"
                    prefill_end += seq_len
                    end += seq_len
                else:
                    if last_stage == "prefill":
                        self.id_group.append((prefill_start, prefill_end))
                        self.reverse_id_group.append((start, end))
                        prefill_start = prefill_end
                        start = end
                        last_stage = "decode"
                    decode_end += seq_len
                    end += seq_len

            if prefill_start < prefill_end:
                self.id_group.append((prefill_start, prefill_end))
                self.reverse_id_group.append((start, end))
            if decode_start < decode_end:
                self.id_group.append((decode_start, decode_end))
                self.reverse_id_group.append((start, end))

    def get_attntion_meta(self):
        """get_attntion_meta"""
        return self.attention_metadata

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
        kv_cache_quant_type: str = None,
    ):
        """
        Calculate kv cache shape
        """
        key_cache_shape = [max_num_blocks, self.num_kv_heads, self.block_size, self.head_dim]
        value_cache_shape = [max_num_blocks, self.num_kv_heads, self.block_size, self.head_dim]
        return key_cache_shape, value_cache_shape

    def transpose(self, hidden_states):
        for ids, reverse_ids in zip(self.id_group, self.reverse_id_group):
            self.tmp_buffer[ids[0] : ids[1], :] = hidden_states[reverse_ids[0] : reverse_ids[1], :]
        return self.tmp_buffer

    def reverse_transpose(self, hidden_states):
        for ids, reverse_ids in zip(self.id_group, self.reverse_id_group):
            self.tmp_buffer[reverse_ids[0] : reverse_ids[1], :] = hidden_states[ids[0] : ids[1], :]
        return self.tmp_buffer

    def get_splited_qkv(
        self, qkv: paddle.Tensor, forward_meta: ForwardMeta, cu_seqlens_q: paddle.Tensor, batch_ids
    ):
        q_end = self.num_heads * self.head_dim
        k_end = q_end + self.num_kv_heads * self.head_dim
        v_end = k_end + self.num_kv_heads * self.head_dim
        assert v_end == qkv.shape[-1], f"Shape mismatch: {v_end} vs {qkv.shape[-1]}"
        assert qkv.shape[0] == cu_seqlens_q[-1], f"Shape mismatch: {qkv.shape[0]} vs {cu_seqlens_q[-1]}"

        q = qkv[..., 0:q_end]
        k = qkv[..., q_end:k_end]
        v = qkv[..., k_end:v_end]
        q = q.view([-1, self.num_heads, self.head_dim])
        k = k.view([-1, self.num_kv_heads, self.head_dim])
        v = v.view([-1, self.num_kv_heads, self.head_dim])

        for idx in range(len(cu_seqlens_q) - 1):
            batch_idx = batch_ids[idx]
            seq_len_i = forward_meta.seq_lens_this_time[batch_idx]
            if seq_len_i == 0:
                continue
            cached_kv_len = forward_meta.seq_lens_decoder[batch_idx][0]
            cu_seq_start_q = cu_seqlens_q[idx]
            cu_seq_end_q = cu_seqlens_q[idx + 1]
            if forward_meta.rotary_embs.dim() == 6:
                # forward_meta.rotary_embs is [BS, 2, 1, S, 1, D]
                cos = forward_meta.rotary_embs[batch_idx, 0, 0, cached_kv_len : cached_kv_len + seq_len_i, :, :]
                sin = forward_meta.rotary_embs[batch_idx, 1, 0, cached_kv_len : cached_kv_len + seq_len_i, :, :]
            else:  # forward_meta.rotary_embs.dim() == 5
                # forward_meta.rotary_embs is [2, 1, S, 1, D]
                cos = forward_meta.rotary_embs[0, 0, cached_kv_len : cached_kv_len + seq_len_i, :, :]
                sin = forward_meta.rotary_embs[1, 0, cached_kv_len : cached_kv_len + seq_len_i, :, :]
            q[cu_seq_start_q:cu_seq_end_q] = apply_rope(q[cu_seq_start_q:cu_seq_end_q], cos, sin)
            k[cu_seq_start_q:cu_seq_end_q] = apply_rope(k[cu_seq_start_q:cu_seq_end_q], cos, sin)

        return q, k, v

    def prefill_update_kv_cache(
        self, k, v, k_cache_id: int, v_cache_id: int, layer_id: int, forward_meta: ForwardMeta, prefill_batch_ids: list
    ):
        # [num_tokens, num_kv_heads, head_dim] -> [num_kv_heads, num_tokens, head_dim]
        trans_k = k.transpose([1, 0, 2]).contiguous()
        trans_v = v.transpose([1, 0, 2]).contiguous()
        tensor_start = 0
        for batch_idx in prefill_batch_ids:
            seq_len = forward_meta.seq_lens_this_time[batch_idx]

            tensor_end = tensor_start + seq_len
            slice_trans_k = trans_k[:, tensor_start:tensor_end, :]
            slice_trans_v = trans_v[:, tensor_start:tensor_end, :]

            cur_block_tables = forward_meta.block_tables[batch_idx]
            cur_used_block_tables = cur_block_tables[cur_block_tables != -1]

            cache_start = 0
            cur_used_num_blocks = cur_used_block_tables.shape[0]
            for i, block_id in enumerate(cur_used_block_tables):
                # last block: seq_len - cache_start <= block_size
                if i == cur_used_num_blocks - 1:
                    cache_end = seq_len - cache_start
                    assert cache_end <= self.block_size
                    paddle.assign(
                        slice_trans_k[:, cache_start:seq_len, :],
                        output=forward_meta.caches[k_cache_id][block_id, :, 0:cache_end, :],
                    )
                    paddle.assign(
                        slice_trans_v[:, cache_start:seq_len, :],
                        output=forward_meta.caches[v_cache_id][block_id, :, 0:cache_end, :],
                    )
                    if layer_id == self.num_layers - 1:
                        self.record_block_table_metadata[batch_idx] = {
                            "block_id": block_id.item(),
                            "cache_end": cache_end.item(),
                        }
                # non last block: seq_lens_this_time > block_size
                else:
                    assert seq_len > self.block_size
                    cache_end = cache_start + self.block_size
                    paddle.assign(
                        slice_trans_k[:, cache_start:cache_end, :], output=forward_meta.caches[k_cache_id][block_id]
                    )
                    paddle.assign(
                        slice_trans_v[:, cache_start:cache_end, :], output=forward_meta.caches[v_cache_id][block_id]
                    )
                    cache_start += self.block_size

            tensor_start = tensor_end

    def forward_mixed(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: Attention,
        forward_meta: ForwardMeta,
    ):
        """
        forward_mixed
        """
        layer_id = layer.layer_id
        k_cache_id = layer_id * 2
        v_cache_id = k_cache_id + 1
        k_cache = forward_meta.caches[k_cache_id]
        v_cache = forward_meta.caches[v_cache_id]
        if self.decode_len == 0:
            if False:
                output = prefill_fused_paged_attention(
                    qkv,
                    k_cache,
                    v_cache,
                    block_tables=forward_meta.block_tables[self.prefill_info_dict["batch_ids"], :],
                    cu_seqlens_qkv=self.prefill_info_dict["cu_seqlens_q"],
                    rope_sin=self.rope_sin,
                    rope_cos=self.rope_cos,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    num_kv_heads=self.num_kv_heads,
                    block_size=self.block_size,
                    max_seq_len=self.max_context_len,
                    scale=self.scale,
                    causal=self.causal,
                    q_rope=True,
                    k_rope=True,
                    v_rope=False,
                    is_interleaved_rope_mode=self.is_interleaved_rope_mode
                )
            else:
                from paddle.nn.functional.flash_attention import flash_attn_unpadded
                
                prefill_q, prefill_k, prefill_v = self.get_splited_qkv(
                    qkv,
                    forward_meta,
                    self.prefill_info_dict["cu_seqlens_q"],
                    self.prefill_info_dict["batch_ids"],
                )

                output = flash_attn_unpadded(
                    prefill_q,
                    prefill_k,
                    prefill_v,
                    cu_seqlens_q=self.prefill_info_dict["cu_seqlens_q"],
                    cu_seqlens_k=self.prefill_info_dict["cu_seqlens_q"],
                    max_seqlen_q=self.max_context_len,
                    max_seqlen_k=self.max_context_len,
                    scale=self.scale,
                    dropout=0.0,
                    causal=self.causal,
                    return_softmax=False,
                )[0]
                self.prefill_update_kv_cache(
                    prefill_k, prefill_v, k_cache_id, v_cache_id, layer_id, forward_meta, self.prefill_info_dict["batch_ids"]
                )
                output = output.view([-1, self.num_heads * self.head_dim])

        elif self.prefill_len == 0:
            if False:
                output = paged_attention(
                    qkv,
                    k_cache,
                    v_cache,
                    block_tables=forward_meta.block_tables[self.decode_info_dict["batch_ids"], :],
                    seq_lens=forward_meta.seq_lens_decoder[self.decode_info_dict["batch_ids"], 0] + 1,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    num_kv_heads=self.num_kv_heads,
                    scale=self.scale,
                    block_size=self.block_size,
                    max_context_len=self.max_context_len,
                    alibi_slopes=self.attention_metadata.alibi_slopes,
                    causal=self.causal,
                    window_left=self.attention_metadata.window_left,
                    window_right=self.attention_metadata.window_right,
                    softcap=self.attention_metadata.softcap,
                    use_cuda_graph=self.attention_metadata.use_cuda_graph,
                    use_sqrt_alibi=self.attention_metadata.use_sqrt_alibi,
                    merged_qkv=True,
                    k=qkv,
                    v=qkv,
                    rope_sin=self.rope_sin,
                    rope_cos=self.rope_cos,
                    rope_batch_stride=self.rope_batch_stride,
                    is_interleaved_rope_mode=self.is_interleaved_rope_mode
                )
            else:
                decode_q, decode_k, decode_v = self.get_splited_qkv(
                    qkv,
                    forward_meta,
                    self.decode_info_dict["cu_seqlens_q"],
                    self.decode_info_dict["batch_ids"],
                )

                decode_q = decode_q.view([-1, self.num_heads * self.head_dim])
                decode_k = decode_k.view([-1, self.num_kv_heads * self.head_dim])
                decode_v = decode_v.view([-1, self.num_kv_heads * self.head_dim])

                output = paged_attention(
                    decode_q,
                    k_cache,
                    v_cache,
                    block_tables=forward_meta.block_tables[self.decode_info_dict["batch_ids"], :],
                    seq_lens=forward_meta.seq_lens_decoder[self.decode_info_dict["batch_ids"], 0] + 1,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    num_kv_heads=self.num_kv_heads,
                    scale=self.scale,
                    block_size=self.block_size,
                    max_context_len=self.max_context_len,
                    alibi_slopes=self.attention_metadata.alibi_slopes,
                    causal=self.causal,
                    window_left=self.attention_metadata.window_left,
                    window_right=self.attention_metadata.window_right,
                    softcap=self.attention_metadata.softcap,
                    use_cuda_graph=self.attention_metadata.use_cuda_graph,
                    use_sqrt_alibi=self.attention_metadata.use_sqrt_alibi,
                    merged_qkv=False,
                    k=decode_k,
                    v=decode_v,
                    rope_sin=None,
                    rope_cos=None,
                    rope_batch_stride=self.rope_batch_stride,
                    is_interleaved_rope_mode=self.is_interleaved_rope_mode
                )

        else:
            if False:
                output = mixed_fused_paged_attention(
                    qkv,
                    k_cache,
                    v_cache,
                    prefill_block_tables=forward_meta.block_tables[self.prefill_info_dict["batch_ids"], :],
                    decode_block_tables=forward_meta.block_tables[self.decode_info_dict["batch_ids"], :],
                    cu_seqlens_qkv=self.prefill_info_dict["cu_seqlens_q"],
                    seq_lens=forward_meta.seq_lens_decoder[self.decode_info_dict["batch_ids"], 0] + 1,
                    prefill_rope_sin=self.prefill_rope_sin,
                    prefill_rope_cos=self.prefill_rope_cos,
                    prefill_num_tokens=self.prefill_num_tokens,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    num_kv_heads=self.num_kv_heads,
                    block_size=self.block_size,
                    max_seq_len=self.max_context_len,
                    scale=self.scale,
                    causal=self.causal,
                    q_rope=True,
                    k_rope=True,
                    v_rope=False,
                    window_left=self.attention_metadata.window_left,
                    window_right=self.attention_metadata.window_right,
                    softcap=self.attention_metadata.softcap,
                    use_cuda_graph=self.attention_metadata.use_cuda_graph,
                    use_sqrt_alibi=self.attention_metadata.use_sqrt_alibi,
                    decode_rope_sin=self.decode_rope_sin,
                    decode_rope_cos=self.decode_rope_cos,
                    rope_batch_stride=self.rope_batch_stride,
                    is_interleaved_rope_mode=self.is_interleaved_rope_mode
                )
            else:
                
                prefill_qkv, decode_qkv = qkv[:self.prefill_num_tokens, :], qkv[self.prefill_num_tokens:, :]
                
                # prefill
                from paddle.nn.functional.flash_attention import flash_attn_unpadded
                prefill_q, prefill_k, prefill_v = self.get_splited_qkv(
                    prefill_qkv,
                    forward_meta,
                    self.prefill_info_dict["cu_seqlens_q"],
                    self.prefill_info_dict["batch_ids"],
                )

                prefill_output = flash_attn_unpadded(
                    prefill_q,
                    prefill_k,
                    prefill_v,
                    cu_seqlens_q=self.prefill_info_dict["cu_seqlens_q"],
                    cu_seqlens_k=self.prefill_info_dict["cu_seqlens_q"],
                    max_seqlen_q=self.max_context_len,
                    max_seqlen_k=self.max_context_len,
                    scale=self.scale,
                    dropout=0.0,
                    causal=self.causal,
                    return_softmax=False,
                )[0]
                self.prefill_update_kv_cache(
                    prefill_k, prefill_v, k_cache_id, v_cache_id, layer_id, forward_meta, self.prefill_info_dict["batch_ids"]
                )
                prefill_output = prefill_output.view([-1, self.num_heads * self.head_dim])

                # decode
                decode_q, decode_k, decode_v = self.get_splited_qkv(
                    decode_qkv,
                    forward_meta,
                    self.decode_info_dict["cu_seqlens_q"],
                    self.decode_info_dict["batch_ids"],
                )

                decode_q = decode_q.view([-1, self.num_heads * self.head_dim])
                decode_k = decode_k.view([-1, self.num_kv_heads * self.head_dim])
                decode_v = decode_v.view([-1, self.num_kv_heads * self.head_dim])

                decode_output = paged_attention(
                    decode_q,
                    k_cache,
                    v_cache,
                    block_tables=forward_meta.block_tables[self.decode_info_dict["batch_ids"], :],
                    seq_lens=forward_meta.seq_lens_decoder[self.decode_info_dict["batch_ids"], 0] + 1,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    num_kv_heads=self.num_kv_heads,
                    scale=self.scale,
                    block_size=self.block_size,
                    max_context_len=self.max_context_len,
                    alibi_slopes=self.attention_metadata.alibi_slopes,
                    causal=self.causal,
                    window_left=self.attention_metadata.window_left,
                    window_right=self.attention_metadata.window_right,
                    softcap=self.attention_metadata.softcap,
                    use_cuda_graph=self.attention_metadata.use_cuda_graph,
                    use_sqrt_alibi=self.attention_metadata.use_sqrt_alibi,
                    merged_qkv=False,
                    k=decode_k,
                    v=decode_v,
                    rope_sin=None,
                    rope_cos=None,
                    rope_batch_stride=self.rope_batch_stride,
                    is_interleaved_rope_mode=self.is_interleaved_rope_mode
                )
                
                output = paddle.concat([prefill_output, decode_output])


        return output
