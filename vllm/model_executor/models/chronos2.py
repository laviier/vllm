# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chronos-2: Encoder-only time series foundation model for zero-shot forecasting."""

from collections.abc import Iterable
from typing import Any
import copy

import torch
import torch.nn as nn
from einops import rearrange, repeat

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import Pooler
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.sequence import IntermediateTensors
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from .interfaces_base import VllmModelForPooling

logger = init_logger(__name__)


class Chronos2LayerNorm(nn.Module):
    """Chronos2 LayerNorm without bias and mean subtraction (T5-style)."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


class Chronos2MLP(nn.Module):
    """Chronos2 MLP layer using vLLM optimized linear layers when possible."""
    
    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        
        # Check if tensor parallelism is available
        try:
            from vllm.distributed.parallel_state import is_initialized
            use_parallel = is_initialized()
        except (ImportError, AttributeError):
            use_parallel = False
        
        if use_parallel:
            # Use vLLM's optimized parallel linear layers
            self.wi = ColumnParallelLinear(
                input_size=config.d_model,
                output_size=config.d_ff,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.wi",
            )
            self.wo = RowParallelLinear(
                input_size=config.d_ff,
                output_size=config.d_model,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.wo",
            )
            self.use_parallel = True
        else:
            # Fall back to regular linear layers
            self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
            self.use_parallel = False
        
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_parallel:
            hidden_states, _ = self.wi(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states, _ = self.wo(hidden_states)
        else:
            hidden_states = self.wi(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.wo(hidden_states)
        return hidden_states


class Chronos2FeedForward(nn.Module):
    """Chronos2 FeedForward with residual connection."""
    
    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        self.mlp = Chronos2MLP(config, quant_config, f"{prefix}.mlp")
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class Chronos2Attention(nn.Module):
    """Chronos2 Multi-head attention using vLLM optimized components when possible."""
    
    def __init__(self, config, use_rope: bool = True, cache_config=None, quant_config=None, prefix: str = ""):
        super().__init__()
        self.d_model = config.d_model
        self.kv_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.kv_proj_dim
        self.use_rope = use_rope
        self.scale = 1.0  # No scaling like original Chronos-2
        
        # Check if tensor parallelism is available
        try:
            from vllm.distributed.parallel_state import is_initialized
            use_parallel = is_initialized()
        except (ImportError, AttributeError):
            use_parallel = False
        
        if use_parallel:
            # Use vLLM's QKV parallel linear layer for efficiency
            self.qkv_proj = QKVParallelLinear(
                hidden_size=self.d_model,
                head_size=self.kv_proj_dim,
                total_num_heads=self.n_heads,
                total_num_kv_heads=self.n_heads,  # Chronos2 uses same heads for K,V as Q
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            
            self.o_proj = RowParallelLinear(
                input_size=self.inner_dim,
                output_size=self.d_model,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj",
            )
            self.use_parallel = True
        else:
            # Fall back to regular linear layers with manual QKV split
            self.q_proj = nn.Linear(self.d_model, self.inner_dim, bias=False)
            self.k_proj = nn.Linear(self.d_model, self.inner_dim, bias=False)
            self.v_proj = nn.Linear(self.d_model, self.inner_dim, bias=False)
            self.o_proj = nn.Linear(self.inner_dim, self.d_model, bias=False)
            self.use_parallel = False

        if use_rope:
            try:
                # Use vLLM's optimized RoPE implementation when available
                self.rotary_emb = get_rope(
                    head_size=self.kv_proj_dim,
                    max_position=getattr(config, "max_position_embeddings", 8192),
                    rope_parameters={"rope_theta": getattr(config, "rope_theta", 10000.0)},
                    is_neox_style=True,
                )
            except (AssertionError, RuntimeError) as e:
                # Fallback when vLLM config context is not available (e.g., in serving process)
                logger.debug(f"Could not initialize vLLM RoPE: {e}. Using simple RoPE fallback.")
                self.rotary_emb = self._create_simple_rope(config)
        else:
            self.rotary_emb = None

    def _create_simple_rope(self, config):
        """Create a simple RoPE implementation as fallback."""
        max_position = getattr(config, "max_position_embeddings", 8192)
        rope_theta = getattr(config, "rope_theta", 10000.0)
        
        # Simple RoPE implementation
        class SimpleRoPE(nn.Module):
            def __init__(self, dim, max_seq_len, theta):
                super().__init__()
                self.dim = dim
                self.max_seq_len = max_seq_len
                
                # Precompute frequency tensor
                inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
                self.register_buffer('inv_freq', inv_freq)
                
            def forward(self, positions, q, k):
                """Apply RoPE to query and key tensors."""
                if positions is None:
                    return q, k
                
                # Input tensors shape: (batch, n_heads, seq_len, head_dim)
                batch_size, n_heads, seq_len, head_dim = q.shape
                    
                # Get frequencies for positions
                freqs = torch.outer(positions.float().flatten(), self.inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                
                cos = emb.cos()
                sin = emb.sin()
                
                # Apply RoPE rotation
                def rotate_half(x):
                    x1, x2 = x.chunk(2, dim=-1)
                    return torch.cat((-x2, x1), dim=-1)
                
                # Reshape for broadcasting to (1, 1, seq_len, head_dim)
                cos = cos.view(1, 1, seq_len, head_dim)
                sin = sin.view(1, 1, seq_len, head_dim)
                
                # Apply rotation
                q_rot = q * cos + rotate_half(q) * sin
                k_rot = k * cos + rotate_half(k) * sin
                
                return q_rot, k_rot
        
        return SimpleRoPE(self.kv_proj_dim, max_position, rope_theta)

    def forward(self, positions, hidden_states, mask=None, output_attentions=False):
        """Forward pass with vLLM optimized attention when available."""
        seq_length = hidden_states.shape[1]
        
        if self.use_parallel:
            # Use QKV projection
            qkv, _ = self.qkv_proj(hidden_states)
            q_size = self.n_heads * self.kv_proj_dim
            kv_size = self.n_heads * self.kv_proj_dim
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        else:
            # Use separate projections
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        
        # Reshape for attention computation FIRST
        def shape_for_attention(x):
            return x.view(-1, seq_length, self.n_heads, self.kv_proj_dim).transpose(1, 2)
            
        q = shape_for_attention(q)
        k = shape_for_attention(k) 
        v = shape_for_attention(v)
        
        # Apply RoPE AFTER reshaping
        if self.use_rope and positions is not None:
            q, k = self.rotary_emb(positions, q, k)
        
        # Compute attention (no scaling)
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        if mask is not None:
            scores += mask
            
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous().view(-1, seq_length, self.inner_dim)
        
        # Output projection
        if self.use_parallel:
            attn_output, _ = self.o_proj(attn_output)
        else:
            attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights if output_attentions else None


class Chronos2TimeSelfAttention(nn.Module):
    """Chronos2 Time Self-Attention layer."""
    
    def __init__(self, config, cache_config=None, quant_config=None, prefix: str = ""):
        super().__init__()
        self.self_attention = Chronos2Attention(config, use_rope=True, cache_config=cache_config, 
                                              quant_config=quant_config, prefix=f"{prefix}.self_attention")
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, attention_mask, position_ids, output_attentions=False):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output, attn_weights = self.self_attention(
            positions=position_ids, hidden_states=normed_hidden_states, mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        return hidden_states, attn_weights


class Chronos2GroupSelfAttention(nn.Module):
    """Chronos2 Group Self-Attention layer (attention along batch axis)."""
    
    def __init__(self, config, cache_config=None, quant_config=None, prefix: str = ""):
        super().__init__()
        self.self_attention = Chronos2Attention(config, use_rope=False, cache_config=cache_config, 
                                              quant_config=quant_config, prefix=f"{prefix}.self_attention")
        self.layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, attention_mask, output_attentions=False):
        # Flip time and batch axes because attention operates along dim=-2
        hidden_states = rearrange(hidden_states, "batch time d -> time batch d")
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output, attn_weights = self.self_attention(
            positions=None, hidden_states=normed_hidden_states, mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        # Flip time and batch axes back to their original position
        hidden_states = rearrange(hidden_states, "time batch d -> batch time d")
        return hidden_states, attn_weights


class Chronos2ResidualBlock(nn.Module):
    """Optimized residual block using vLLM linear layers when possible."""
    
    def __init__(self, in_dim, h_dim, out_dim, act_fn_name, dropout_p=0.0, quant_config=None, prefix: str = ""):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        
        # Check if tensor parallelism is available
        try:
            from vllm.distributed.parallel_state import is_initialized
            use_parallel = is_initialized()
        except (ImportError, AttributeError):
            use_parallel = False
        
        if use_parallel:
            # Use vLLM parallel linear layers for better performance
            self.hidden_layer = ColumnParallelLinear(
                input_size=in_dim,
                output_size=h_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.hidden_layer",
            )
            self.output_layer = RowParallelLinear(
                input_size=h_dim,
                output_size=out_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.output_layer",
            )
            self.residual_layer = ColumnParallelLinear(
                input_size=in_dim,
                output_size=out_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.residual_layer",
            )
            self.use_parallel = True
        else:
            # Fall back to regular linear layers
            self.hidden_layer = nn.Linear(in_dim, h_dim, bias=True)
            self.output_layer = nn.Linear(h_dim, out_dim, bias=True)
            self.residual_layer = nn.Linear(in_dim, out_dim, bias=True)
            self.use_parallel = False
        
        self.act = get_act_fn(act_fn_name)

    def forward(self, x):
        if self.use_parallel:
            hid, _ = self.hidden_layer(x)
            hid = self.act(hid)
            out, _ = self.output_layer(self.dropout(hid))
            res, _ = self.residual_layer(x)
        else:
            hid = self.hidden_layer(x)
            hid = self.act(hid)
            out = self.output_layer(self.dropout(hid))
            res = self.residual_layer(x)
        return out + res


class Chronos2Pooler(Pooler):
    """Simple pooler for Chronos-2 time series model."""

    def get_supported_tasks(self) -> set[PoolingTask]:
        """Return supported pooling tasks."""
        return {"forecast"}  # Chronos-2 is a time series forecasting model

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> torch.Tensor:
        """
        Pool hidden states for Chronos-2.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            pooling_metadata: Pooling metadata

        Returns:
            Pooled tensor (batch, hidden_size)
        """
        # Mean pooling across sequence dimension
        return hidden_states.mean(dim=1)


class InstanceNorm(nn.Module):
    """Instance normalization (scaling) for time series."""
    
    def __init__(self, use_arcsinh: bool = False):
        super().__init__()
        self.use_arcsinh = use_arcsinh

    def forward(self, x: torch.Tensor, loc_scale=None):
        if loc_scale is not None:
            loc, scale = loc_scale
            return (x - loc) / scale, (loc, scale)
        
        # Compute location and scale
        if self.use_arcsinh:
            x_transformed = torch.asinh(x)
            loc = x_transformed.mean(dim=-1, keepdim=True)
            scale = x_transformed.std(dim=-1, keepdim=True) + 1e-6
            normalized = (x_transformed - loc) / scale
            return torch.sinh(normalized), (loc, scale)
        else:
            loc = x.mean(dim=-1, keepdim=True)
            scale = x.std(dim=-1, keepdim=True) + 1e-6
            return (x - loc) / scale, (loc, scale)

    def inverse(self, x: torch.Tensor, loc_scale):
        loc, scale = loc_scale
        if self.use_arcsinh:
            x_asinh = x * scale + loc
            return torch.sinh(x_asinh)
        else:
            return x * scale + loc


class Patch(nn.Module):
    """Patching layer for time series."""
    
    def __init__(self, patch_size: int, patch_stride: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        # Handle case where input is shorter than patch size
        if seq_len < self.patch_size:
            # Pad the input to at least patch_size length
            padding_length = self.patch_size - seq_len
            # Pad with zeros on the left (historical padding)
            padding = torch.zeros(batch_size, padding_length, device=x.device, dtype=x.dtype)
            x = torch.cat([padding, x], dim=-1)
            seq_len = x.shape[-1]
        
        # Calculate number of patches
        num_patches = (seq_len - self.patch_size) // self.patch_stride + 1
        
        # Ensure at least one patch
        if num_patches <= 0:
            num_patches = 1
        
        # Create patches
        patches = []
        for i in range(num_patches):
            start_idx = i * self.patch_stride
            end_idx = start_idx + self.patch_size
            
            # Handle case where end_idx exceeds sequence length
            if end_idx > seq_len:
                # Take the last patch_size elements
                start_idx = max(0, seq_len - self.patch_size)
                end_idx = seq_len
            
            patch = x[:, start_idx:end_idx]
            
            # Ensure patch has correct size
            if patch.shape[-1] < self.patch_size:
                padding_needed = self.patch_size - patch.shape[-1]
                patch_padding = torch.zeros(batch_size, padding_needed, device=x.device, dtype=x.dtype)
                patch = torch.cat([patch_padding, patch], dim=-1)
            
            patches.append(patch)
        
        # Stack patches: (batch_size, num_patches, patch_size)
        return torch.stack(patches, dim=1)


class Chronos2EncoderBlock(nn.Module):
    """Chronos2 encoder block with time and group attention."""
    
    def __init__(self, config, cache_config=None, quant_config=None, prefix: str = ""):
        super().__init__()
        self.time_self_attn = Chronos2TimeSelfAttention(
            config, cache_config, quant_config, f"{prefix}.time_self_attn"
        )
        self.group_self_attn = Chronos2GroupSelfAttention(
            config, cache_config, quant_config, f"{prefix}.group_self_attn"
        )
        self.feed_forward = Chronos2FeedForward(
            config, quant_config, f"{prefix}.feed_forward"
        )

    def forward(self, hidden_states, position_ids, attention_mask, group_time_mask, output_attentions=False):
        # Time self-attention
        hidden_states, time_attn_weights = self.time_self_attn(
            hidden_states, attention_mask, position_ids, output_attentions
        )
        
        # Group self-attention
        hidden_states, group_attn_weights = self.group_self_attn(
            hidden_states, group_time_mask, output_attentions
        )
        
        # Feed forward
        hidden_states = self.feed_forward(hidden_states)
        
        return hidden_states, time_attn_weights, group_attn_weights


class Chronos2Encoder(nn.Module):
    """Chronos2 encoder using vLLM layers."""
    
    def __init__(self, config, cache_config=None, quant_config=None, prefix: str = ""):
        super().__init__()
        self.blocks = nn.ModuleList([
            Chronos2EncoderBlock(config, cache_config, quant_config, f"{prefix}.blocks.{i}") 
            for i in range(config.num_layers)
        ])
        self.final_layer_norm = Chronos2LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    @staticmethod
    def _expand_and_invert_time_attention_mask(attention_mask, floating_type):
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=floating_type)
        attention_mask = (1.0 - attention_mask) * torch.finfo(floating_type).min
        return attention_mask

    @staticmethod
    def _construct_and_invert_group_time_mask(group_ids, attention_mask, floating_type):
        group_mask = group_ids[:, None] == group_ids[None, :]
        group_time_mask = torch.einsum("qb, bt -> qbt", group_mask, attention_mask)
        if torch.is_floating_point(group_time_mask):
            floating_type = group_time_mask.dtype
        group_time_mask = rearrange(group_time_mask, "q b t -> t 1 q b")
        group_time_mask = (1.0 - group_time_mask) * torch.finfo(floating_type).min
        return group_time_mask

    def forward(self, inputs_embeds, group_ids, attention_mask=None, position_ids=None, output_attentions=False):
        batch_size, seq_length = inputs_embeds.size()[:-1]

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=inputs_embeds.device, dtype=inputs_embeds.dtype)

        # Process attention masks
        extended_attention_mask = self._expand_and_invert_time_attention_mask(attention_mask, inputs_embeds.dtype)
        group_time_mask = self._construct_and_invert_group_time_mask(group_ids, attention_mask, inputs_embeds.dtype)

        hidden_states = self.dropout(inputs_embeds)
        
        all_time_attentions = ()
        all_group_attentions = ()

        for block in self.blocks:
            hidden_states, time_attn_weights, group_attn_weights = block(
                hidden_states, position_ids, extended_attention_mask, group_time_mask, output_attentions
            )
            
            if output_attentions:
                all_time_attentions += (time_attn_weights,)
                all_group_attentions += (group_attn_weights,)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states, all_time_attentions, all_group_attentions


class Chronos2ForForecasting(nn.Module, VllmModelForPooling):
    """
    Chronos-2 time series forecasting model using vLLM layers.

    This implementation replaces the pipeline dependency with direct use of vLLM
    optimized layers while preserving the Chronos2 architecture.
    """

    packed_modules_mapping = {}
    supported_lora_modules = []
    embedding_modules = {}
    embedding_padding_modules = []

    is_pooling_model = True

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # CRITICAL: Chronos-2 is encoder-only for time series forecasting
        config.is_encoder_decoder = False

        self.config = config
        self.model_dim = config.d_model

        # Chronos-2 specific config
        chronos_config = getattr(config, "chronos_config", {})
        self.chronos_config = chronos_config
        
        self.context_length = chronos_config.get("context_length", 8192)
        self.input_patch_size = chronos_config.get("input_patch_size", 16)
        self.output_patch_size = chronos_config.get("output_patch_size", 16)
        self.input_patch_stride = chronos_config.get("input_patch_stride", self.input_patch_size)
        self.max_output_patches = chronos_config.get("max_output_patches", 64)
        self.use_reg_token = chronos_config.get("use_reg_token", False)
        self.use_arcsinh = chronos_config.get("use_arcsinh", False)
        self.time_encoding_scale = chronos_config.get("time_encoding_scale", self.context_length)
        
        self.quantiles = chronos_config.get("quantiles", [0.1, 0.5, 0.9])
        self.num_quantiles = len(self.quantiles)
        
        # Register quantiles as buffer
        quantiles = torch.tensor(self.quantiles, dtype=torch.float32)
        self.register_buffer("quantiles_tensor", quantiles, persistent=False)

        # Token embeddings (for special tokens like REG if used)
        config.vocab_size = 2 if self.use_reg_token else 1
        if self.use_reg_token:
            config.reg_token_id = 1
        
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Input patch embedding layer
        self.input_patch_embedding = Chronos2ResidualBlock(
            in_dim=self.input_patch_size * 3,  # [time_embedding, patch, patch_mask]
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
            quant_config=quant_config,
        )

        # Patching and normalization layers
        self.patch = Patch(
            patch_size=self.input_patch_size,
            patch_stride=self.input_patch_stride
        )
        self.instance_norm = InstanceNorm(use_arcsinh=self.use_arcsinh)

        # Encoder
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        self.encoder = Chronos2Encoder(encoder_config, cache_config, quant_config)

        # Output patch embedding layer
        self.output_patch_embedding = Chronos2ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=self.num_quantiles * self.output_patch_size,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
            quant_config=quant_config,
        )

        # Initialize pooler module
        self.pooler = Chronos2Pooler()

        logger.info(
            "Initialized Chronos2ForForecasting with vLLM layers: d_model=%d, "
            "num_layers=%d, context_length=%d, patch_size=%d",
            config.d_model,
            config.num_layers,
            self.context_length,
            self.input_patch_size,
        )

    def _prepare_patched_context(self, context, context_mask=None):
        """Prepare patched context inputs."""
        context_mask = (
            context_mask.to(context.dtype)
            if context_mask is not None
            else torch.isnan(context).logical_not().to(context.dtype)
        )

        batch_size, context_length = context.shape
        
        # Truncate context if it's longer than model's context length
        if context_length > self.context_length:
            context = context[..., -self.context_length:]
            context_mask = context_mask[..., -self.context_length:]

        # Instance normalization (scaling)
        context, loc_scale = self.instance_norm(context)
        context = context.to(self.config.torch_dtype or torch.float32)
        context_mask = context_mask.to(context.dtype)

        # Patching
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(context_mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)

        # Attention mask: 1 if at least one item in the patch is observed
        attention_mask = patched_mask.sum(dim=-1) > 0  # (batch_size, num_patches)
        num_context_patches = attention_mask.shape[-1]

        # Context time encoding
        final_context_length = num_context_patches * self.input_patch_size
        context_time_enc = torch.arange(start=-final_context_length, end=0, device=context.device, dtype=torch.float32)
        context_time_enc = (
            repeat(
                context_time_enc,
                "(n p) -> b n p",
                b=batch_size,
                n=num_context_patches,
                p=self.input_patch_size,
            )
            .div(self.time_encoding_scale)
            .to(context.dtype)
        )

        # Concat time encoding, context and mask along the last (feature) dim
        patched_context = torch.cat([context_time_enc, patched_context, patched_mask], dim=-1)

        return patched_context, attention_mask, loc_scale

    def _prepare_patched_future(self, future_covariates, future_covariates_mask, loc_scale, num_output_patches, batch_size):
        """Prepare patched future inputs."""
        if future_covariates is not None:
            future_covariates, _ = self.instance_norm(future_covariates, loc_scale)
            future_covariates = future_covariates.to(self.config.torch_dtype or torch.float32)

            if future_covariates_mask is None:
                future_covariates_mask = torch.isnan(future_covariates).logical_not().to(future_covariates.dtype)

            future_covariates = torch.where(future_covariates_mask > 0.0, future_covariates, 0.0)

            # Add padding if needed
            if num_output_patches * self.output_patch_size > future_covariates.shape[-1]:
                padding_shape = (
                    *future_covariates.shape[:-1],
                    num_output_patches * self.output_patch_size - future_covariates.shape[-1],
                )
                future_covariates = torch.cat(
                    [future_covariates, torch.zeros(padding_shape, device=future_covariates.device, dtype=future_covariates.dtype)], dim=-1
                )
                future_covariates_mask = torch.cat(
                    [future_covariates_mask, torch.zeros(padding_shape, device=future_covariates_mask.device, dtype=future_covariates_mask.dtype)], dim=-1
                )

            patched_future_covariates = rearrange(
                future_covariates, "b (n p) -> b n p", n=num_output_patches, p=self.output_patch_size
            )
            patched_future_covariates_mask = rearrange(
                future_covariates_mask, "b (n p) -> b n p", n=num_output_patches, p=self.output_patch_size
            )
        else:
            patched_future_covariates = torch.zeros(
                batch_size, num_output_patches, self.output_patch_size, device=self.shared.weight.device, dtype=self.config.torch_dtype or torch.float32
            )
            patched_future_covariates_mask = torch.zeros(
                batch_size, num_output_patches, self.output_patch_size, device=self.shared.weight.device, dtype=self.config.torch_dtype or torch.float32
            )

        # Future time encoding
        final_future_length = num_output_patches * self.output_patch_size
        future_time_enc = torch.arange(start=0, end=final_future_length, device=self.shared.weight.device, dtype=torch.float32)
        future_time_enc = (
            repeat(
                future_time_enc,
                "(n p) -> b n p",
                b=batch_size,
                n=num_output_patches,
                p=self.output_patch_size,
            )
            .div(self.time_encoding_scale)
            .to(patched_future_covariates.dtype)
        )

        patched_future = torch.cat(
            [future_time_enc, patched_future_covariates, patched_future_covariates_mask], dim=-1
        )

        return patched_future, patched_future_covariates_mask

    def predict(
        self,
        inputs: list | torch.Tensor = None,
        context: torch.Tensor = None,
        num_output_patches: int = 1,
        context_mask: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        prediction_length: int = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate forecasts using vLLM layers directly.
        
        Compatible with both direct calls and vLLM serving interface.

        Args:
            inputs: Input data from serving interface (list of dicts or tensor)
            context: Input time series tensor of shape (batch_size, seq_len)  
            num_output_patches: Number of output patches to forecast
            prediction_length: Alternative way to specify forecast length
            context_mask: Optional mask for context
            future_covariates: Optional future covariates
            future_covariates_mask: Optional mask for future covariates
            group_ids: Optional group IDs for cross-series learning

        Returns:
            Quantile predictions tensor of shape (batch_size, num_quantiles, forecast_length)
        """
        # Handle serving interface inputs
        if inputs is not None and context is None:
            if isinstance(inputs, list):
                # Extract target data from serving format
                # Expected format: [{"target": [1,2,3,4,5,6,7,8,9,10]}]
                contexts = []
                for inp in inputs:
                    if isinstance(inp, dict) and "target" in inp:
                        target_data = torch.tensor(inp["target"], dtype=torch.float32)
                        contexts.append(target_data)
                    else:
                        raise ValueError(f"Expected input dict with 'target' key, got: {inp}")
                
                if contexts:
                    context = torch.stack(contexts)
                else:
                    raise ValueError("No valid target data found in inputs")
            else:
                context = inputs
        
        # Handle prediction_length parameter (from serving interface)
        if prediction_length is not None:
            # Convert prediction_length to num_output_patches
            # Each patch produces output_patch_size predictions
            num_output_patches = max(1, (prediction_length + self.output_patch_size - 1) // self.output_patch_size)
        
        # Ensure context is a tensor on the right device
        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context, dtype=torch.float32)
        
        # Move to model device if needed
        if hasattr(self.shared.weight, 'device'):
            context = context.to(self.shared.weight.device)
        batch_size = context.shape[0]
        
        # Prepare inputs
        patched_context, attention_mask, loc_scale = self._prepare_patched_context(context, context_mask)
        num_context_patches = attention_mask.shape[-1]

        # Get input embeddings
        input_embeds = self.input_patch_embedding(patched_context)
        
        # Append REG token embedding if needed
        if self.use_reg_token:
            reg_input_ids = torch.full((batch_size, 1), self.config.reg_token_id, device=input_embeds.device)
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [attention_mask.to(input_embeds.dtype), torch.ones_like(reg_input_ids, dtype=input_embeds.dtype)], dim=-1
            )

        # Prepare future inputs
        patched_future, _ = self._prepare_patched_future(
            future_covariates, future_covariates_mask, loc_scale, num_output_patches, batch_size
        )
        future_attention_mask = torch.ones(batch_size, num_output_patches, dtype=input_embeds.dtype, device=input_embeds.device)

        # Get future embeddings
        future_embeds = self.input_patch_embedding(patched_future)

        # Concatenate context and future embeddings and masks
        input_embeds = torch.cat([input_embeds, future_embeds], dim=-2)
        attention_mask = torch.cat([attention_mask, future_attention_mask], dim=-1)

        # Set default group IDs if not provided
        if group_ids is None:
            group_ids = torch.arange(batch_size, dtype=torch.long, device=input_embeds.device)

        # Run encoder
        hidden_states, _, _ = self.encoder(
            inputs_embeds=input_embeds,
            group_ids=group_ids,
            attention_mask=attention_mask,
        )

        # Get forecast embeddings (last num_output_patches hidden states)
        forecast_embeds = hidden_states[:, -num_output_patches:]
        
        # Generate quantile predictions
        quantile_preds = self.output_patch_embedding(forecast_embeds)
        quantile_preds = rearrange(
            quantile_preds,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=self.num_quantiles,
            p=self.output_patch_size,
        )

        # Unscale predictions
        quantile_preds = rearrange(quantile_preds, "b q h -> b (q h)", b=batch_size, q=self.num_quantiles)
        quantile_preds = self.instance_norm.inverse(quantile_preds, loc_scale)
        quantile_preds = rearrange(
            quantile_preds,
            "b (q h) -> b q h",
            q=self.num_quantiles,
            h=num_output_patches * self.output_patch_size,
        )

        # Trim to requested prediction_length if needed
        if prediction_length is not None:
            quantile_preds = quantile_preds[:, :, :prediction_length]

        # Check if this is being called from the serving interface
        # If inputs were provided as list (serving format), return formatted response
        if inputs is not None and isinstance(inputs, list):
            # Convert to proper serving response format
            from vllm.entrypoints.pooling.forecast.protocol import ForecastPrediction
            
            predictions = []
            for batch_idx in range(batch_size):
                # Extract predictions for this time series
                ts_predictions = quantile_preds[batch_idx]  # Shape: (num_quantiles, prediction_length)
                
                # Get input metadata
                input_data = inputs[batch_idx]
                item_id = input_data.get("item_id") if isinstance(input_data, dict) else None
                
                # Create prediction dict with median as mean and quantiles as dynamic fields
                pred_dict = {
                    "mean": ts_predictions[self.num_quantiles // 2].tolist(),  # Use median quantile
                    "item_id": item_id,
                }
                
                # Add quantile-specific predictions as dynamic fields  
                for q_idx, quantile in enumerate(self.quantiles):
                    pred_dict[str(quantile)] = ts_predictions[q_idx].tolist()
                
                predictions.append(ForecastPrediction(**pred_dict))
            
            return predictions
        else:
            # Direct call - return raw tensors
            return quantile_preds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for Chronos-2 model using vLLM layers.

        Args:
            input_ids: Time series input tensor (used as context for forecasting)
            positions: Position indices
            intermediate_tensors: Optional intermediate tensors
            inputs_embeds: Optional pre-computed embeddings

        Returns:
            Hidden states for pooling or forecasting
        """
        if inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
        elif input_ids is not None:
            # Handle both 1D (warmup) and 2D (normal) input shapes
            if input_ids.ndim == 1:
                batch_size = 1
                seq_len = input_ids.shape[0]
                input_ids = input_ids.unsqueeze(0)
            elif input_ids.ndim == 2:
                batch_size, seq_len = input_ids.shape
            else:
                raise ValueError(f"Expected 1D or 2D input_ids, got {input_ids.ndim}D")
            
            # For vLLM compatibility, treat input_ids as time series values
            # Convert to float and normalize for time series processing
            context = input_ids.float()
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # If inputs_embeds provided, use them directly
        if inputs_embeds is not None:
            return inputs_embeds

        # Use predict method for actual time series forecasting 
        # with minimal output patches for efficiency during forward pass
        try:
            with torch.no_grad():
                forecasts = self.predict(
                    context=context,
                    num_output_patches=1,  # Minimal for forward pass
                )
            
            # Return mean forecast as hidden states for pooling
            # Shape: (batch_size, forecast_length, 1) -> (batch_size, seq_len, d_model)
            forecast_mean = forecasts[:, self.num_quantiles // 2, :]  # Use median quantile
            
            # Expand to match expected hidden size
            if forecast_mean.shape[-1] < self.model_dim:
                padding_size = self.model_dim - forecast_mean.shape[-1]
                padding = torch.zeros(batch_size, padding_size, device=forecast_mean.device, dtype=forecast_mean.dtype)
                forecast_mean = torch.cat([forecast_mean, padding], dim=-1)
            elif forecast_mean.shape[-1] > self.model_dim:
                forecast_mean = forecast_mean[:, :self.model_dim]
            
            # Reshape to (batch_size, seq_len, d_model)
            if forecast_mean.shape[-1] != self.model_dim:
                forecast_mean = forecast_mean.unsqueeze(-1).expand(batch_size, seq_len, self.model_dim)
            else:
                forecast_mean = forecast_mean.unsqueeze(1).expand(batch_size, seq_len, self.model_dim)
            
            return forecast_mean
            
        except Exception as e:
            logger.warning(f"Forecast prediction failed during forward pass: {e}. Using dummy embeddings.")
            # Fallback to dummy embeddings for vLLM compatibility
            return torch.zeros(
                batch_size,
                seq_len,
                self.model_dim,
                device=context.device,
                dtype=torch.float32,
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert input IDs to embeddings.

        For Chronos-2, this is a placeholder since actual embedding
        happens in the Chronos2Pipeline.

        Args:
            input_ids: Input tensor

        Returns:
            Dummy embeddings
        """
        # Handle both 1D and 2D inputs
        if input_ids.ndim == 1:
            batch_size = 1
            seq_len = input_ids.shape[0]
        else:
            batch_size, seq_len = input_ids.shape

        return torch.zeros(
            batch_size,
            seq_len,
            self.d_model,
            device=input_ids.device,
            dtype=torch.float32,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights for Chronos-2 model using vLLM's weight loading system.

        This implementation loads weights directly from Chronos2 checkpoints and maps
        them to the corresponding vLLM layers, eliminating the need for the pipeline.

        Args:
            weights: Iterable of (parameter_name, tensor) pairs

        Returns:
            Set of loaded parameter names
        """
        params_dict = dict(weights)
        loaded_params = set()
        
        # Collect QKV weights for fusion
        qkv_weights = {}  # {layer_type}_{layer_idx}: {'q': tensor, 'k': tensor, 'v': tensor}
        
        # Debug: Print all checkpoint parameter names
        logger.info(f"Checkpoint contains {len(params_dict)} parameters:")
        for i, param_name in enumerate(sorted(params_dict.keys())):
            if i < 20:  # Show first 20 parameters
                logger.info(f"  {param_name}: {params_dict[param_name].shape}")
            elif i == 20:
                logger.info("  ... (truncated)")
                break
        
        # First pass: collect and organize weights
        for param_name, param_tensor in params_dict.items():
            loaded = False
            
            # Handle shared embedding
            if param_name == "shared.weight":
                self.shared.weight.data.copy_(param_tensor)
                loaded_params.add("shared.weight")
                loaded = True
                logger.debug(f"✓ Loaded shared embedding: {param_name}")
                
            # Handle final layer norm
            elif param_name == "encoder.final_layer_norm.weight":
                self.encoder.final_layer_norm.weight.data.copy_(param_tensor)
                loaded_params.add("encoder.final_layer_norm.weight")
                loaded = True
                logger.debug(f"✓ Loaded final layer norm: {param_name}")
                
            # Handle patch embeddings
            elif param_name.startswith(("input_patch_embedding.", "output_patch_embedding.")):
                try:
                    param_parts = param_name.split('.')
                    module = self
                    for part in param_parts[:-1]:
                        module = getattr(module, part)
                    param = getattr(module, param_parts[-1])
                    param.data.copy_(param_tensor)
                    loaded_params.add(param_name)
                    loaded = True
                    logger.debug(f"✓ Loaded patch embedding: {param_name}")
                except AttributeError:
                    logger.warning(f"Could not load parameter: {param_name}")
                    
            # Handle encoder blocks - extract layer index first
            elif param_name.startswith("encoder.block."):
                # Extract layer index: encoder.block.{layer_idx}.layer.{layer_num}.{component}
                parts = param_name.split(".")
                try:
                    layer_idx = int(parts[2])  # encoder.block.{layer_idx}
                    layer_num = int(parts[4])  # layer.{layer_num} 
                    
                    logger.debug(f"Processing layer {layer_idx}, sub-layer {layer_num}: {param_name}")
                    
                    # Determine attention type and component
                    if layer_num == 0 and "self_attention" in param_name:  # Time self-attention
                        attn_type = "time_self_attn"
                        key = f"{attn_type}_{layer_idx}"
                        
                        if param_name.endswith(".q.weight"):
                            if key not in qkv_weights:
                                qkv_weights[key] = {}
                            qkv_weights[key]['q'] = param_tensor
                            logger.debug(f"Collected Q weight for {key}")
                        elif param_name.endswith(".k.weight"):
                            if key not in qkv_weights:
                                qkv_weights[key] = {}
                            qkv_weights[key]['k'] = param_tensor
                            logger.debug(f"Collected K weight for {key}")
                        elif param_name.endswith(".v.weight"):
                            if key not in qkv_weights:
                                qkv_weights[key] = {}
                            qkv_weights[key]['v'] = param_tensor
                            logger.debug(f"Collected V weight for {key}")
                        elif param_name.endswith(".o.weight"):
                            # Load output projection directly
                            self.encoder.blocks[layer_idx].time_self_attn.self_attention.o_proj.weight.data.copy_(param_tensor)
                            vllm_param = f"encoder.blocks.{layer_idx}.time_self_attn.self_attention.o_proj.weight"
                            loaded_params.add(vllm_param)
                            loaded = True
                            logger.debug(f"✓ Loaded time attention output projection: {vllm_param}")
                            
                    elif layer_num == 1 and "self_attention" in param_name:  # Group self-attention
                        attn_type = "group_self_attn"
                        key = f"{attn_type}_{layer_idx}"
                        
                        if param_name.endswith(".q.weight"):
                            if key not in qkv_weights:
                                qkv_weights[key] = {}
                            qkv_weights[key]['q'] = param_tensor
                            logger.debug(f"Collected Q weight for {key}")
                        elif param_name.endswith(".k.weight"):
                            if key not in qkv_weights:
                                qkv_weights[key] = {}
                            qkv_weights[key]['k'] = param_tensor
                            logger.debug(f"Collected K weight for {key}")
                        elif param_name.endswith(".v.weight"):
                            if key not in qkv_weights:
                                qkv_weights[key] = {}
                            qkv_weights[key]['v'] = param_tensor
                            logger.debug(f"Collected V weight for {key}")
                        elif param_name.endswith(".o.weight"):
                            # Load output projection directly
                            self.encoder.blocks[layer_idx].group_self_attn.self_attention.o_proj.weight.data.copy_(param_tensor)
                            vllm_param = f"encoder.blocks.{layer_idx}.group_self_attn.self_attention.o_proj.weight"
                            loaded_params.add(vllm_param)
                            loaded = True
                            logger.debug(f"✓ Loaded group attention output projection: {vllm_param}")
                            
                    elif layer_num == 0 and param_name.endswith("layer_norm.weight"):
                        # Time attention layer norm
                        self.encoder.blocks[layer_idx].time_self_attn.layer_norm.weight.data.copy_(param_tensor)
                        vllm_param = f"encoder.blocks.{layer_idx}.time_self_attn.layer_norm.weight"
                        loaded_params.add(vllm_param)
                        loaded = True
                        logger.debug(f"✓ Loaded time attention layer norm: {vllm_param}")
                        
                    elif layer_num == 1 and param_name.endswith("layer_norm.weight"):
                        # Group attention layer norm
                        self.encoder.blocks[layer_idx].group_self_attn.layer_norm.weight.data.copy_(param_tensor)
                        vllm_param = f"encoder.blocks.{layer_idx}.group_self_attn.layer_norm.weight"
                        loaded_params.add(vllm_param)
                        loaded = True
                        logger.debug(f"✓ Loaded group attention layer norm: {vllm_param}")
                        
                    elif layer_num == 2 and ".mlp.wi.weight" in param_name:
                        # MLP input projection
                        self.encoder.blocks[layer_idx].feed_forward.mlp.wi.weight.data.copy_(param_tensor)
                        vllm_param = f"encoder.blocks.{layer_idx}.feed_forward.mlp.wi.weight"
                        loaded_params.add(vllm_param)
                        loaded = True
                        logger.debug(f"✓ Loaded MLP input projection: {vllm_param}")
                        
                    elif layer_num == 2 and ".mlp.wo.weight" in param_name:
                        # MLP output projection
                        self.encoder.blocks[layer_idx].feed_forward.mlp.wo.weight.data.copy_(param_tensor)
                        vllm_param = f"encoder.blocks.{layer_idx}.feed_forward.mlp.wo.weight"
                        loaded_params.add(vllm_param)
                        loaded = True
                        logger.debug(f"✓ Loaded MLP output projection: {vllm_param}")
                        
                    elif layer_num == 2 and param_name.endswith("layer_norm.weight"):
                        # Feed forward layer norm
                        self.encoder.blocks[layer_idx].feed_forward.layer_norm.weight.data.copy_(param_tensor)
                        vllm_param = f"encoder.blocks.{layer_idx}.feed_forward.layer_norm.weight"
                        loaded_params.add(vllm_param)
                        loaded = True
                        logger.debug(f"✓ Loaded feed forward layer norm: {vllm_param}")
                        
                except (IndexError, ValueError) as e:
                    logger.warning(f"Could not parse layer indices from {param_name}: {e}")
            
            if not loaded:
                logger.debug(f"Unhandled parameter: {param_name}")

        # Second pass: fuse QKV weights
        for key, qkv_dict in qkv_weights.items():
            if len(qkv_dict) == 3:  # Have all Q, K, V
                attn_type, layer_idx = key.rsplit('_', 1)
                layer_idx = int(layer_idx)
                
                # Concatenate Q, K, V weights for fused QKV projection
                q_weight = qkv_dict['q']
                k_weight = qkv_dict['k'] 
                v_weight = qkv_dict['v']
                
                # vLLM QKV expects [Q, K, V] concatenation
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                
                # Load into the appropriate attention module (handle both parallel and fallback cases)
                attention_module = None
                if attn_type == "time_self_attn":
                    attention_module = self.encoder.blocks[layer_idx].time_self_attn.self_attention
                    vllm_param_base = f"encoder.blocks.{layer_idx}.time_self_attn.self_attention"
                elif attn_type == "group_self_attn":
                    attention_module = self.encoder.blocks[layer_idx].group_self_attn.self_attention
                    vllm_param_base = f"encoder.blocks.{layer_idx}.group_self_attn.self_attention"
                
                if attention_module:
                    if hasattr(attention_module, 'qkv_proj'):
                        # Using vLLM parallel layers - load fused QKV weight
                        attention_module.qkv_proj.weight.data.copy_(qkv_weight)
                        vllm_param = f"{vllm_param_base}.qkv_proj.weight"
                        loaded_params.add(vllm_param)
                        logger.debug(f"Fused QKV weights for {attn_type} layer {layer_idx} (parallel)")
                    elif hasattr(attention_module, 'q_proj'):
                        # Using fallback layers - load separate Q, K, V weights
                        q_weight = qkv_dict['q']
                        k_weight = qkv_dict['k']
                        v_weight = qkv_dict['v']
                        
                        attention_module.q_proj.weight.data.copy_(q_weight)
                        attention_module.k_proj.weight.data.copy_(k_weight)
                        attention_module.v_proj.weight.data.copy_(v_weight)
                        
                        loaded_params.add(f"{vllm_param_base}.q_proj.weight")
                        loaded_params.add(f"{vllm_param_base}.k_proj.weight")
                        loaded_params.add(f"{vllm_param_base}.v_proj.weight")
                        logger.debug(f"Separate QKV weights for {attn_type} layer {layer_idx} (fallback)")
                    else:
                        logger.warning(f"Could not find QKV layers for {attn_type} layer {layer_idx}")

        # Log loading summary
        total_params = len(params_dict)
        loaded_count = len(loaded_params)
        logger.info(
            f"Loaded {loaded_count}/{total_params} parameters into Chronos2 vLLM model. "
            f"Using vLLM optimized layers instead of pipeline."
        )
        
        if loaded_count < total_params:
            unloaded_params = set(params_dict.keys()) - {name for name in params_dict.keys() 
                                                         if any(name in loaded_name for loaded_name in loaded_params)}
            logger.warning(f"Some parameters were not loaded: {len(unloaded_params)} missing")
            logger.debug(f"Missing parameters: {list(unloaded_params)[:10]}...")  # Show first 10

        return loaded_params
