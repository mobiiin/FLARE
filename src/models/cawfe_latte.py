"""New CAWFE-Latte encoder and fusion stem.

This is intentionally not a complete forecasting model yet. It implements only
atmosphere, wind, fire/fuel, flux/energy encoders and fire-query modality fusion.
"""

from __future__ import annotations

import contextlib
import math
import warnings
from types import SimpleNamespace
from collections.abc import Mapping, Sequence
from typing import Any

from src.models.mamba_backend import build_mamba_layer

try:
	import torch
	import torch.nn as nn
	import torch.nn.functional as F
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object)
	F = None


def _group_count(channels: int, preferred: int = 8) -> int:
	for groups in range(min(preferred, channels), 0, -1):
		if channels % groups == 0:
			return groups
	return 1


def _as_int_list(values: Sequence[Any] | None, default: Sequence[int]) -> list[int]:
	if values is None:
		return [int(value) for value in default]
	return [int(value) for value in values]


def _as_str_list(values: Sequence[Any] | None) -> list[str]:
	if values is None:
		return []
	return [str(value).lower() for value in values]


def _manual_multihead_attention(
	module: nn.MultiheadAttention,
	query: torch.Tensor,
	key: torch.Tensor,
	value: torch.Tensor,
	*,
	training: bool,
	need_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
	"""Small-sequence MHA using explicit matmul/softmax instead of SDPA kernels."""
	if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
		raise ValueError("Manual MHA expects batch-first B,L,D tensors.")
	if key.shape != value.shape:
		raise ValueError(f"Manual MHA key/value shapes differ: key={tuple(key.shape)} value={tuple(value.shape)}")
	embed_dim = int(module.embed_dim)
	num_heads = int(module.num_heads)
	head_dim = embed_dim // num_heads
	if embed_dim % num_heads != 0:
		raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}.")
	if int(query.shape[-1]) != embed_dim or int(key.shape[-1]) != embed_dim:
		raise ValueError(f"Manual MHA expected D={embed_dim}, got query={tuple(query.shape)} key={tuple(key.shape)}")
	weight = module.in_proj_weight
	bias = module.in_proj_bias
	if weight is None:
		raise ValueError("Manual MHA requires in_proj_weight.")
	q_bias = None if bias is None else bias[:embed_dim]
	k_bias = None if bias is None else bias[embed_dim : 2 * embed_dim]
	v_bias = None if bias is None else bias[2 * embed_dim :]
	q_proj = F.linear(query, weight[:embed_dim], q_bias)
	k_proj = F.linear(key, weight[embed_dim : 2 * embed_dim], k_bias)
	v_proj = F.linear(value, weight[2 * embed_dim :], v_bias)
	batch = int(query.shape[0])
	q_len = int(query.shape[1])
	k_len = int(key.shape[1])
	q_proj = q_proj.reshape(batch, q_len, num_heads, head_dim).transpose(1, 2)
	k_proj = k_proj.reshape(batch, k_len, num_heads, head_dim).transpose(1, 2)
	v_proj = v_proj.reshape(batch, k_len, num_heads, head_dim).transpose(1, 2)
	# These attentions have tiny sequence lengths (vertical: 8, fusion: 1x3).
	# Use broadcasted multiply/reduce instead of matmul/bmm so backward does not route
	# through Triton-backed bmm kernels on clusters where libcuda discovery is brittle.
	scores = (q_proj.unsqueeze(3) * k_proj.unsqueeze(2)).sum(dim=-1) / math.sqrt(float(head_dim))
	weights = torch.softmax(scores, dim=-1)
	weights_for_values = F.dropout(weights, p=float(module.dropout), training=training) if float(module.dropout) > 0.0 else weights
	out = (weights_for_values.unsqueeze(-1) * v_proj.unsqueeze(2)).sum(dim=3)
	out = out.transpose(1, 2).contiguous().reshape(batch, q_len, embed_dim)
	out = module.out_proj(out)
	if need_weights:
		return out, weights.mean(dim=1)
	return out, None


def select_channels_by_keywords(channel_names: Mapping[int, str] | None, keywords: Sequence[str], *, input_channels: int | None = None) -> list[int]:
	"""Return channel indices whose names contain any keyword."""

	if not channel_names or not keywords:
		return []
	lowered_keywords = [str(keyword).lower() for keyword in keywords]
	selected: list[int] = []
	for index, name in channel_names.items():
		channel_index = int(index)
		if input_channels is not None and not 0 <= channel_index < int(input_channels):
			continue
		lowered_name = str(name).lower()
		if any(keyword in lowered_name for keyword in lowered_keywords):
			selected.append(channel_index)
	return sorted(set(selected))


def _dedupe_indices(indices: Sequence[int], *, input_channels: int) -> list[int]:
	seen: set[int] = set()
	result: list[int] = []
	for index in indices:
		channel_index = int(index)
		if not 0 <= channel_index < int(input_channels):
			continue
		if channel_index not in seen:
			seen.add(channel_index)
			result.append(channel_index)
	return result


def _apply_per_timestep(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
	if x.ndim != 5:
		raise ValueError(f"Expected B x T x C x H x W tensor, got shape {tuple(x.shape)}.")
	batch, time_steps, channels, height, width = (int(value) for value in x.shape)
	y = module(x.reshape(batch * time_steps, channels, height, width))
	out_channels = int(y.shape[1])
	return y.reshape(batch, time_steps, out_channels, height, width)


class ConvBlock2d(nn.Module):
	"""Small channel-first Conv2d block."""

	def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 3) -> None:
		super().__init__()
		padding = int(kernel_size) // 2
		self.block = nn.Sequential(
			nn.Conv2d(int(in_channels), int(out_channels), kernel_size=int(kernel_size), padding=padding),
			nn.GroupNorm(_group_count(int(out_channels)), int(out_channels)),
			nn.SiLU(inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.block(x)


class TerrainEncoder(nn.Module):
	"""Encode static four-channel terrain features without a time axis."""
	def __init__(self, in_channels: int = 4, hidden_dim: int = 32, out_dim: int = 64, activation: str = "gelu", norm: str = "groupnorm") -> None:
		super().__init__(); self.out_dim=int(out_dim)
		act = nn.GELU() if str(activation).lower() == "gelu" else nn.SiLU()
		def block(cin, cout):
			return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(_group_count(cout), cout) if str(norm).lower() == "groupnorm" else nn.BatchNorm2d(cout), act)
		self.net=nn.Sequential(block(int(in_channels), int(hidden_dim)), block(int(hidden_dim), int(out_dim)), block(int(out_dim), int(out_dim)))
	def forward(self, terrain: torch.Tensor) -> torch.Tensor:
		if terrain.ndim != 4: raise ValueError(f"TerrainEncoder expects B x C x H x W, got {tuple(terrain.shape)}")
		return self.net(terrain)


class TerrainFiLMConditioner(nn.Module):
	"""Zero-initialized FiLM modulation for static terrain conditioning."""
	def __init__(self, dim: int = 64, use_scale: bool = True, use_shift: bool = True, scale_init_zero: bool = True, shift_init_zero: bool = True) -> None:
		super().__init__(); self.use_scale=bool(use_scale); self.use_shift=bool(use_shift)
		self.scale=nn.Conv2d(dim, dim, 1) if self.use_scale else None; self.shift=nn.Conv2d(dim, dim, 1) if self.use_shift else None
		if scale_init_zero and self.scale is not None: nn.init.zeros_(self.scale.weight); nn.init.zeros_(self.scale.bias)
		if shift_init_zero and self.shift is not None: nn.init.zeros_(self.shift.weight); nn.init.zeros_(self.shift.bias)
	def forward(self, z: torch.Tensor, terrain_embedding: torch.Tensor) -> torch.Tensor:
		if z.ndim != 5 or terrain_embedding.ndim != 4 or int(z.shape[2]) != int(terrain_embedding.shape[1]) or tuple(z.shape[-2:]) != tuple(terrain_embedding.shape[-2:]): raise ValueError("Terrain FiLM spatial/channel dimensions must match fused representation.")
		scale=self.scale(terrain_embedding) if self.scale is not None else torch.zeros_like(terrain_embedding)
		shift=self.shift(terrain_embedding) if self.shift is not None else torch.zeros_like(terrain_embedding)
		return z * (1 + scale[:, None]) + shift[:, None]


class ResidualBlock2d(nn.Module):
	"""Residual spatial refinement block for one timestep."""

	def __init__(self, channels: int) -> None:
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1),
			nn.GroupNorm(_group_count(int(channels)), int(channels)),
			nn.SiLU(inplace=True),
			nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1),
			nn.GroupNorm(_group_count(int(channels)), int(channels)),
		)
		self.activation = nn.SiLU(inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.activation(x + self.block(x))


class SequenceSpatialEncoder(nn.Module):
	"""Apply a 2D projection and residual blocks independently per timestep."""

	def __init__(self, in_channels: int, hidden_dim: int, out_dim: int, num_blocks: int) -> None:
		super().__init__()
		layers: list[nn.Module] = [ConvBlock2d(int(in_channels), int(hidden_dim), kernel_size=1)]
		for _ in range(max(0, int(num_blocks))):
			layers.append(ResidualBlock2d(int(hidden_dim)))
		if int(hidden_dim) != int(out_dim):
			layers.append(ConvBlock2d(int(hidden_dim), int(out_dim), kernel_size=1))
		self.net = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return _apply_per_timestep(x, self.net)


class TemporalConvBlock(nn.Module):
	"""Lightweight temporal mixing over B x T x C x H x W."""

	def __init__(self, channels: int) -> None:
		super().__init__()
		self.conv = nn.Conv3d(int(channels), int(channels), kernel_size=(3, 1, 1), padding=(1, 0, 0))
		self.norm = nn.GroupNorm(_group_count(int(channels)), int(channels))
		self.activation = nn.SiLU(inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"TemporalConvBlock expects B x T x C x H x W, got {tuple(x.shape)}.")
		y = x.permute(0, 2, 1, 3, 4).contiguous()
		y = self.activation(self.norm(self.conv(y)))
		return y.permute(0, 2, 1, 3, 4).contiguous()


class AtmosphereEncoder(nn.Module):
	"""Encode raw 8-level x 10-variable atmospheric structure."""

	def __init__(
		self,
		*,
		out_dim: int = 64,
		num_levels: int = 8,
		vars_per_level: int = 10,
		hidden_dim: int = 64,
		num_blocks: int = 2,
		vertical_mixing: str = "attention",
		vertical_attention_chunk_size: int | None = 32768,
		vertical_attention_force_math: bool = True,
	) -> None:
		super().__init__()
		self.out_dim = int(out_dim)
		self.num_levels = int(num_levels)
		self.vars_per_level = int(vars_per_level)
		self.hidden_dim = int(hidden_dim)
		self.required_channels = self.num_levels * self.vars_per_level
		self.vertical_mixing = str(vertical_mixing).lower()
		self.vertical_attention_chunk_size = None if vertical_attention_chunk_size in (None, 0, "", "null") else int(vertical_attention_chunk_size)
		self.vertical_attention_force_math = bool(vertical_attention_force_math)
		if self.vertical_mixing not in {"attention", "mlp"}:
			raise ValueError(f"vertical_mixing must be 'attention' or 'mlp', got {vertical_mixing!r}.")
		self.variable_projection = nn.Linear(self.vars_per_level, self.hidden_dim)
		if self.vertical_mixing == "attention":
			num_heads = max(1, min(4, self.hidden_dim // 16))
			while self.hidden_dim % num_heads != 0:
				num_heads -= 1
			self.vertical_attention = nn.MultiheadAttention(self.hidden_dim, num_heads, batch_first=True)
			self.pool_score = nn.Linear(self.hidden_dim, 1)
			self.vertical_mlp = None
		else:
			self.vertical_attention = None
			self.pool_score = None
			self.vertical_mlp = nn.Sequential(
				nn.Linear(self.num_levels * self.hidden_dim, self.hidden_dim),
				nn.SiLU(),
				nn.Linear(self.hidden_dim, self.hidden_dim),
			)
		self.refine = SequenceSpatialEncoder(self.hidden_dim, self.out_dim, self.out_dim, int(num_blocks))

	def _vertical_attention_context(self):
		if not self.vertical_attention_force_math or torch is None:
			return contextlib.nullcontext()
		try:
			from torch.nn.attention import SDPBackend, sdpa_kernel

			return sdpa_kernel(SDPBackend.MATH)
		except Exception:
			pass
		if not hasattr(torch.backends, "cuda"):
			return contextlib.nullcontext()
		try:
			return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
		except Exception:
			return contextlib.nullcontext()

	def _apply_vertical_attention(self, tokens: torch.Tensor) -> torch.Tensor:
		assert self.vertical_attention is not None
		chunk_size = self.vertical_attention_chunk_size
		if chunk_size is None or int(chunk_size) <= 0 or int(tokens.shape[0]) <= int(chunk_size):
			out, _ = _manual_multihead_attention(self.vertical_attention, tokens, tokens, tokens, training=self.training, need_weights=False)
			return out
		chunks = []
		for start in range(0, int(tokens.shape[0]), int(chunk_size)):
			chunk = tokens[start : start + int(chunk_size)]
			out, _ = _manual_multihead_attention(self.vertical_attention, chunk, chunk, chunk, training=self.training, need_weights=False)
			chunks.append(out)
		return torch.cat(chunks, dim=0)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"AtmosphereEncoder expects B x T x C x H x W, got {tuple(x.shape)}.")
		if int(x.shape[2]) < self.required_channels:
			raise ValueError(
				f"AtmosphereEncoder requires at least {self.required_channels} channels for "
				f"{self.num_levels} levels x {self.vars_per_level} variables, got {int(x.shape[2])}."
			)
		batch, time_steps, _, height, width = (int(value) for value in x.shape)
		x_atm = x[:, :, : self.required_channels].reshape(batch, time_steps, self.num_levels, self.vars_per_level, height, width)
		tokens = x_atm.permute(0, 1, 4, 5, 2, 3).contiguous().reshape(batch * time_steps * height * width, self.num_levels, self.vars_per_level)
		tokens = self.variable_projection(tokens)
		if self.vertical_mixing == "attention":
			assert self.vertical_attention is not None and self.pool_score is not None
			tokens = self._apply_vertical_attention(tokens)
			weights = torch.softmax(self.pool_score(tokens), dim=1)
			mixed = (tokens * weights).sum(dim=1)
		else:
			assert self.vertical_mlp is not None
			mixed = self.vertical_mlp(tokens.reshape(tokens.shape[0], self.num_levels * self.hidden_dim))
		maps = mixed.reshape(batch, time_steps, height, width, self.hidden_dim).permute(0, 1, 4, 2, 3).contiguous()
		return self.refine(maps)


class WindEncoder(nn.Module):
	"""Encode wind transport features derived from U/V/W and optional engineered channels."""

	def __init__(
		self,
		*,
		input_channels: int = 129,
		out_dim: int = 64,
		hidden_dim: int = 64,
		num_blocks: int = 2,
		num_levels: int = 8,
		vars_per_level: int = 10,
		u_var_index: int = 0,
		v_var_index: int = 1,
		w_var_index: int = 2,
		use_engineered_wind_channels: bool = True,
		engineered_channel_keywords: Sequence[str] | None = None,
		channel_names: Mapping[int, str] | None = None,
		eps: float = 1.0e-6,
	) -> None:
		super().__init__()
		self.input_channels = int(input_channels)
		self.num_levels = int(num_levels)
		self.vars_per_level = int(vars_per_level)
		self.u_var_index = int(u_var_index)
		self.v_var_index = int(v_var_index)
		self.w_var_index = int(w_var_index)
		self.eps = float(eps)
		for name, index in {"u_var_index": self.u_var_index, "v_var_index": self.v_var_index, "w_var_index": self.w_var_index}.items():
			if not 0 <= index < self.vars_per_level:
				raise ValueError(f"{name} must be in [0, {self.vars_per_level - 1}], got {index}.")
		keywords = _as_str_list(engineered_channel_keywords)
		engineered = select_channels_by_keywords(channel_names, keywords, input_channels=self.input_channels) if use_engineered_wind_channels else []
		self.engineered_indices = _dedupe_indices([index for index in engineered if index >= self.num_levels * self.vars_per_level], input_channels=self.input_channels)
		if use_engineered_wind_channels and keywords and channel_names is None:
			warnings.warn("WindEncoder channel manifest unavailable; using derived U/V/W wind features only.", stacklevel=2)
		derived_channels = 7 * self.num_levels
		self.encoder = SequenceSpatialEncoder(derived_channels + len(self.engineered_indices), int(hidden_dim), int(out_dim), int(num_blocks))
		self.temporal = TemporalConvBlock(int(out_dim))

	def _level_indices(self, var_index: int) -> list[int]:
		return [level * self.vars_per_level + int(var_index) for level in range(self.num_levels)]

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"WindEncoder expects B x T x C x H x W, got {tuple(x.shape)}.")
		min_required = (self.num_levels - 1) * self.vars_per_level + max(self.u_var_index, self.v_var_index, self.w_var_index) + 1
		if int(x.shape[2]) < min_required:
			raise ValueError(f"WindEncoder requires at least {min_required} channels for configured U/V/W extraction, got {int(x.shape[2])}.")
		u = x[:, :, self._level_indices(self.u_var_index)]
		v = x[:, :, self._level_indices(self.v_var_index)]
		w = x[:, :, self._level_indices(self.w_var_index)]
		speed = torch.sqrt(u.square() + v.square() + self.eps)
		dir_cos = u / speed.clamp_min(self.eps)
		dir_sin = v / speed.clamp_min(self.eps)
		updraft = torch.relu(w)
		parts = [u, v, w, speed, dir_cos, dir_sin, updraft]
		if self.engineered_indices:
			parts.append(x[:, :, self.engineered_indices])
		features = torch.cat(parts, dim=2)
		return self.temporal(self.encoder(features))


class FireFuelEncoder(nn.Module):
	"""Encode fuel and fire-history features with extra local capacity."""

	def __init__(
		self,
		*,
		input_channels: int = 129,
		out_dim: int = 64,
		hidden_dim: int = 64,
		num_blocks: int = 4,
		raw_fuel_channels: Sequence[int] | None = None,
		engineered_channel_keywords: Sequence[str] | None = None,
		channel_names: Mapping[int, str] | None = None,
		terrain_channels: int = 0,
	) -> None:
		super().__init__()
		self.input_channels = int(input_channels)
		raw = _as_int_list(raw_fuel_channels, [84, 85])
		engineered = select_channels_by_keywords(channel_names, _as_str_list(engineered_channel_keywords), input_channels=self.input_channels)
		self.terrain_channels = int(terrain_channels)
		self.indices = _dedupe_indices(raw + [index for index in engineered if index not in raw], input_channels=self.input_channels)
		self.indices += list(range(self.input_channels, self.input_channels + self.terrain_channels))
		if len(self.indices) < len(raw):
			raise ValueError(f"FireFuelEncoder requires raw fuel channels {raw}, but input_channels={self.input_channels}.")
		if engineered_channel_keywords and channel_names is None:
			warnings.warn("FireFuelEncoder channel manifest unavailable; using raw fuel channels only.", stacklevel=2)
		self.encoder = SequenceSpatialEncoder(len(self.indices), int(hidden_dim), int(out_dim), int(num_blocks))
		self.temporal = TemporalConvBlock(int(out_dim))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"FireFuelEncoder expects B x T x C x H x W, got {tuple(x.shape)}.")
		if int(x.shape[2]) <= max(self.indices):
			raise ValueError(f"FireFuelEncoder requires channels up to {max(self.indices)}, got {int(x.shape[2])}.")
		return self.temporal(self.encoder(x[:, :, self.indices]))


class FluxEnergyEncoder(nn.Module):
	"""Encode flux and energy-driver channels."""

	def __init__(
		self,
		*,
		input_channels: int = 129,
		out_dim: int = 64,
		hidden_dim: int = 64,
		num_blocks: int = 2,
		raw_flux_channels: Sequence[int] | None = None,
		engineered_channel_keywords: Sequence[str] | None = None,
		channel_names: Mapping[int, str] | None = None,
	) -> None:
		super().__init__()
		self.input_channels = int(input_channels)
		raw = _as_int_list(raw_flux_channels, [80, 81, 82, 83])
		engineered = select_channels_by_keywords(channel_names, _as_str_list(engineered_channel_keywords), input_channels=self.input_channels)
		self.indices = _dedupe_indices(raw + [index for index in engineered if index not in raw], input_channels=self.input_channels)
		if len(self.indices) < len(raw):
			raise ValueError(f"FluxEnergyEncoder requires raw flux channels {raw}, but input_channels={self.input_channels}.")
		if engineered_channel_keywords and channel_names is None:
			warnings.warn("FluxEnergyEncoder channel manifest unavailable; using raw flux channels only.", stacklevel=2)
		self.encoder = SequenceSpatialEncoder(len(self.indices), int(hidden_dim), int(out_dim), int(num_blocks))
		self.temporal = TemporalConvBlock(int(out_dim))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"FluxEnergyEncoder expects B x T x C x H x W, got {tuple(x.shape)}.")
		if int(x.shape[2]) <= max(self.indices):
			raise ValueError(f"FluxEnergyEncoder requires channels up to {max(self.indices)}, got {int(x.shape[2])}.")
		return self.temporal(self.encoder(x[:, :, self.indices]))


def tokens_to_grid(tokens: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
	"""Convert B x T x N x D row-major tokens back to B x T x D x H x W."""
	if tokens.ndim != 4:
		raise ValueError(f"tokens_to_grid expects B x T x N x D tokens, got {tuple(tokens.shape)}.")
	height, width = (int(value) for value in spatial_shape)
	batch, time_steps, num_tokens, dim = (int(value) for value in tokens.shape)
	if num_tokens != height * width:
		raise ValueError(f"Token count N={num_tokens} does not match spatial shape {(height, width)}.")
	return tokens.transpose(-1, -2).reshape(batch, time_steps, dim, height, width).contiguous()


class MultimodalAlignment(nn.Module):
	"""Align encoded CAWFE-Latte modalities before token fusion."""

	def __init__(
		self,
		*,
		dim: int = 64,
		max_time: int = 16,
		use_spatial_pos: bool = True,
		use_temporal_pos: bool = True,
		separate_layernorms: bool = True,
		eps: float = 1.0e-5,
		learned_spatial_pos: bool = True,
		learned_temporal_pos: bool = True,
		flatten_order: str = "row_major",
	) -> None:
		super().__init__()
		self.dim = int(dim)
		self.max_time = int(max_time)
		self.use_spatial_pos = bool(use_spatial_pos)
		self.use_temporal_pos = bool(use_temporal_pos)
		self.separate_layernorms = bool(separate_layernorms)
		self.learned_spatial_pos = bool(learned_spatial_pos)
		self.learned_temporal_pos = bool(learned_temporal_pos)
		self.flatten_order = str(flatten_order).lower()
		if self.flatten_order != "row_major":
			raise ValueError(f"CAWFE-Latte alignment supports only flatten_order='row_major', got {flatten_order!r}.")
		if self.dim < 1:
			raise ValueError(f"alignment.dim must be positive, got {self.dim}.")
		if self.max_time < 1:
			raise ValueError(f"alignment.max_time must be positive, got {self.max_time}.")
		if not self.learned_spatial_pos and self.use_spatial_pos:
			raise ValueError("CAWFE-Latte alignment v1 currently supports only learned spatial positional embeddings.")
		if not self.learned_temporal_pos and self.use_temporal_pos:
			raise ValueError("CAWFE-Latte alignment v1 currently supports only learned temporal positional embeddings.")
		if self.separate_layernorms:
			self.norm_atm = nn.LayerNorm(self.dim, eps=float(eps))
			self.norm_wind = nn.LayerNorm(self.dim, eps=float(eps))
			self.norm_fire = nn.LayerNorm(self.dim, eps=float(eps))
			self.norm_flux = nn.LayerNorm(self.dim, eps=float(eps))
		else:
			shared = nn.LayerNorm(self.dim, eps=float(eps))
			self.norm_atm = shared
			self.norm_wind = shared
			self.norm_fire = shared
			self.norm_flux = shared
		if self.use_spatial_pos:
			self.register_parameter("spatial_pos", None)
			self._spatial_shape: tuple[int, int] | None = None
		else:
			self.register_parameter("spatial_pos", None)
			self._spatial_shape = None
		if self.use_temporal_pos:
			self.temporal_pos = nn.Parameter(torch.zeros(1, self.max_time, 1, self.dim))
			nn.init.trunc_normal_(self.temporal_pos, std=0.02)
		else:
			self.register_parameter("temporal_pos", None)

	@staticmethod
	def to_tokens(x: torch.Tensor) -> torch.Tensor:
		# B,T,D,H,W -> B,T,N,D. flatten(-2) preserves row-major token order:
		# token index = y * W + x.
		return x.flatten(-2).transpose(-1, -2).contiguous()

	def _validate_inputs(self, tensors: Mapping[str, torch.Tensor]) -> tuple[int, int, int, int, int]:
		shapes = {name: tuple(tensor.shape) for name, tensor in tensors.items()}
		for name, tensor in tensors.items():
			if tensor.ndim != 5:
				raise ValueError(f"Alignment input {name} must be B x T x D x H x W, got {tuple(tensor.shape)}.")
		first = next(iter(tensors.values()))
		batch, time_steps, dim, height, width = (int(value) for value in first.shape)
		for name, tensor in tensors.items():
			if tuple(tensor.shape[:2]) != (batch, time_steps):
				raise ValueError(f"Alignment inputs must share B,T dimensions; got {shapes}.")
			if int(tensor.shape[2]) != self.dim:
				raise ValueError(f"Alignment input {name} has D={int(tensor.shape[2])}; expected {self.dim}.")
			if tuple(tensor.shape[-2:]) != (height, width):
				raise ValueError(f"Alignment inputs must share the same spatial grid; got {shapes}.")
		if time_steps > self.max_time:
			raise ValueError(f"Alignment received T={time_steps}, but max_time={self.max_time}.")
		return batch, time_steps, dim, height, width

	def _spatial_position(self, height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
		if not self.use_spatial_pos:
			return torch.zeros(1, 1, int(height) * int(width), self.dim, device=device, dtype=dtype)
		shape = (int(height), int(width))
		num_tokens = shape[0] * shape[1]
		if self.spatial_pos is None:
			param = nn.Parameter(torch.zeros(1, 1, num_tokens, self.dim, device=device, dtype=dtype))
			nn.init.trunc_normal_(param, std=0.02)
			self.spatial_pos = param
			self._spatial_shape = shape
		elif self._spatial_shape != shape:
			raise ValueError(f"Alignment learned spatial_pos was initialized for {self._spatial_shape}, got {shape}.")
		return self.spatial_pos.to(device=device, dtype=dtype)

	def forward(self, A: torch.Tensor, W: torch.Tensor, F: torch.Tensor, Q: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int]]:
		_, time_steps, _, height, width = self._validate_inputs({"atmosphere": A, "wind": W, "fire_fuel": F, "flux_energy": Q})
		A_tokens = self.norm_atm(self.to_tokens(A))
		W_tokens = self.norm_wind(self.to_tokens(W))
		F_tokens = self.norm_fire(self.to_tokens(F))
		Q_tokens = self.norm_flux(self.to_tokens(Q))
		spatial_pos = self._spatial_position(height, width, device=A_tokens.device, dtype=A_tokens.dtype)
		if self.use_temporal_pos:
			assert self.temporal_pos is not None
			temporal_pos = self.temporal_pos[:, :time_steps].to(device=A_tokens.device, dtype=A_tokens.dtype)
		else:
			temporal_pos = torch.zeros(1, time_steps, 1, self.dim, device=A_tokens.device, dtype=A_tokens.dtype)
		return {
			"atmosphere": A_tokens + spatial_pos + temporal_pos,
			"wind": W_tokens + spatial_pos + temporal_pos,
			"fire_fuel": F_tokens + spatial_pos + temporal_pos,
			"flux_energy": Q_tokens + spatial_pos + temporal_pos,
			"spatial_shape": (height, width),
		}


class FireQueryCrossAttentionFusion(nn.Module):
	"""Fuse modalities per spatial token/time using fire/fuel as the query."""

	modalities = ("atmosphere", "wind", "flux_energy")

	def __init__(
		self,
		*,
		dim: int = 64,
		num_heads: int = 4,
		dropout: float = 0.1,
		use_layer_norm: bool = True,
		residual: bool = True,
		attention_chunk_size: int | None = 32768,
		attention_force_math: bool = True,
	) -> None:
		super().__init__()
		self.dim = int(dim)
		self.residual = bool(residual)
		self.use_layer_norm = bool(use_layer_norm)
		self.attention_chunk_size = None if attention_chunk_size in (None, 0, "", "null") else int(attention_chunk_size)
		self.attention_force_math = bool(attention_force_math)
		if self.dim % int(num_heads) != 0:
			raise ValueError(f"fusion.num_heads={num_heads} must divide fusion.dim={self.dim}.")
		self.q_norm = nn.LayerNorm(self.dim) if self.use_layer_norm else nn.Identity()
		self.kv_norm = nn.LayerNorm(self.dim) if self.use_layer_norm else nn.Identity()
		self.attention = nn.MultiheadAttention(self.dim, int(num_heads), dropout=float(dropout), batch_first=True)
		self.dropout = nn.Dropout(float(dropout))
		self.out_norm = nn.LayerNorm(self.dim) if self.use_layer_norm else nn.Identity()
		self.mlp = nn.Sequential(nn.Linear(self.dim, self.dim * 2), nn.SiLU(), nn.Dropout(float(dropout)), nn.Linear(self.dim * 2, self.dim))

	def _attention_context(self):
		if not self.attention_force_math or torch is None:
			return contextlib.nullcontext()
		try:
			from torch.nn.attention import SDPBackend, sdpa_kernel

			return sdpa_kernel(SDPBackend.MATH)
		except Exception:
			pass
		if not hasattr(torch.backends, "cuda"):
			return contextlib.nullcontext()
		try:
			return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
		except Exception:
			return contextlib.nullcontext()

	def _apply_attention(self, q: torch.Tensor, kv: torch.Tensor, *, return_attention: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
		chunk_size = self.attention_chunk_size
		if chunk_size is None or int(chunk_size) <= 0 or int(q.shape[0]) <= int(chunk_size):
			attended, weights = _manual_multihead_attention(self.attention, q, kv, kv, training=self.training, need_weights=return_attention)
			return attended, weights if return_attention else None
		attended_chunks = []
		weight_chunks = [] if return_attention else None
		for start in range(0, int(q.shape[0]), int(chunk_size)):
			end = start + int(chunk_size)
			attended, weights = _manual_multihead_attention(
				self.attention,
				q[start:end],
				kv[start:end],
				kv[start:end],
				training=self.training,
				need_weights=return_attention,
			)
			attended_chunks.append(attended)
			if return_attention and weights is not None:
				assert weight_chunks is not None
				weight_chunks.append(weights)
		attended_all = torch.cat(attended_chunks, dim=0)
		weights_all = torch.cat(weight_chunks, dim=0) if weight_chunks else None
		return attended_all, weights_all

	def forward(
		self,
		atmosphere: torch.Tensor,
		wind: torch.Tensor,
		fire_fuel: torch.Tensor,
		flux_energy: torch.Tensor,
		*,
		return_attention: bool = False,
	) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
		for name, tensor in {"atmosphere": atmosphere, "wind": wind, "fire_fuel": fire_fuel, "flux_energy": flux_energy}.items():
			if tensor.ndim != 4:
				raise ValueError(f"Fusion input {name} must be B x T x N x D tokens, got {tuple(tensor.shape)}.")
			if int(tensor.shape[-1]) != self.dim:
				raise ValueError(f"Fusion input {name} has D={int(tensor.shape[-1])}; expected {self.dim}.")
		if not (atmosphere.shape == wind.shape == fire_fuel.shape == flux_energy.shape):
			raise ValueError("Fusion token inputs must all have the same B x T x N x D shape.")
		batch, time_steps, num_tokens, dim = (int(value) for value in fire_fuel.shape)
		q = self.q_norm(fire_fuel.reshape(batch * time_steps * num_tokens, 1, dim))
		kv = torch.stack([atmosphere, wind, flux_energy], dim=3).reshape(batch * time_steps * num_tokens, 3, dim)
		kv = self.kv_norm(kv)
		attended, weights = self._apply_attention(q, kv, return_attention=return_attention)
		attended = attended.reshape(batch, time_steps, num_tokens, dim)
		z = fire_fuel + self.dropout(attended) if self.residual else self.dropout(attended)
		z = z + self.dropout(self.mlp(self.out_norm(z)))
		if return_attention:
			attention_tokens = weights.squeeze(1).reshape(batch, time_steps, num_tokens, 3)
			return z, attention_tokens
		return z


class ConcatProjectionFusion(nn.Module):
	"""Simple aligned-grid concatenation and 1x1 projection without attention."""

	def __init__(self, *, dim: int = 64) -> None:
		super().__init__()
		self.dim = int(dim)
		self.projection = nn.Sequential(
			nn.Conv2d(4 * self.dim, self.dim, kernel_size=1),
			nn.GroupNorm(_group_count(self.dim), self.dim),
			nn.SiLU(inplace=True),
		)

	def forward(
		self,
		atmosphere: torch.Tensor,
		wind: torch.Tensor,
		fire_fuel: torch.Tensor,
		flux_energy: torch.Tensor,
	) -> torch.Tensor:
		inputs = (atmosphere, wind, fire_fuel, flux_energy)
		if any(tensor.ndim != 5 for tensor in inputs):
			raise ValueError("ConcatProjectionFusion inputs must be B x T x D x H x W.")
		if not all(tensor.shape == atmosphere.shape for tensor in inputs):
			raise ValueError("ConcatProjectionFusion inputs must have identical shapes.")
		if int(atmosphere.shape[2]) != self.dim:
			raise ValueError(f"ConcatProjectionFusion expected D={self.dim}, got {int(atmosphere.shape[2])}.")
		batch, time_steps, _, height, width = (int(value) for value in atmosphere.shape)
		concatenated = torch.cat(inputs, dim=2)
		projected = self.projection(concatenated.reshape(batch * time_steps, 4 * self.dim, height, width))
		return projected.reshape(batch, time_steps, self.dim, height, width)


class TemporalSpatialResidualBlock(nn.Module):
	"""Efficient residual block mixing time then local space."""

	def __init__(self, dim: int, *, temporal_kernel_size: int = 3, spatial_kernel_size: int = 3, dropout: float = 0.1, residual: bool = True) -> None:
		super().__init__()
		self.residual = bool(residual)
		t_padding = int(temporal_kernel_size) // 2
		s_padding = int(spatial_kernel_size) // 2
		self.temporal = nn.Conv3d(int(dim), int(dim), kernel_size=(int(temporal_kernel_size), 1, 1), padding=(t_padding, 0, 0))
		self.temporal_norm = nn.GroupNorm(_group_count(int(dim)), int(dim))
		self.spatial = nn.Conv3d(int(dim), int(dim), kernel_size=(1, int(spatial_kernel_size), int(spatial_kernel_size)), padding=(0, s_padding, s_padding))
		self.spatial_norm = nn.GroupNorm(_group_count(int(dim)), int(dim))
		self.dropout = nn.Dropout3d(float(dropout)) if float(dropout) > 0 else nn.Identity()
		self.activation = nn.SiLU(inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"TemporalSpatialResidualBlock expects B x T x C x H x W, got {tuple(x.shape)}.")
		x3d = x.permute(0, 2, 1, 3, 4).contiguous()
		y = self.activation(self.temporal_norm(self.temporal(x3d)))
		y = self.spatial_norm(self.spatial(y))
		y = self.dropout(y)
		if self.residual:
			y = x3d + y
		y = self.activation(y)
		return y.permute(0, 2, 1, 3, 4).contiguous()


class ResidualSpatiotemporalBlock(TemporalSpatialResidualBlock):
	"""Same-resolution residual spatiotemporal block.

	Input and output layout is B x T x D x H x W. Internally the block uses
	Conv3d over B x D x T x H x W, first mixing time and then local space.
	"""

	def __init__(self, dim: int, *, temporal_kernel_size: int = 3, spatial_kernel_size: int = 3, dropout: float = 0.1, norm: str = "groupnorm", activation: str = "silu", residual: bool = True) -> None:
		if str(norm).lower() != "groupnorm":
			raise ValueError(f"Residual spatiotemporal blocks require norm='groupnorm', got {norm!r}.")
		if str(activation).lower() != "silu":
			raise ValueError(f"Residual spatiotemporal blocks require activation='silu', got {activation!r}.")
		super().__init__(
			dim=int(dim),
			temporal_kernel_size=int(temporal_kernel_size),
			spatial_kernel_size=int(spatial_kernel_size),
			dropout=float(dropout),
			residual=bool(residual),
		)


class ResidualSpatiotemporalBackbone(nn.Module):
	"""Configurable-depth same-resolution residual spatiotemporal backbone."""

	def __init__(self, *, dim: int = 64, num_blocks: int = 6, temporal_kernel_size: int = 3, spatial_kernel_size: int = 3, dropout: float = 0.1, residual: bool = True, type: str = "residual_spatiotemporal", norm: str = "groupnorm", activation: str = "silu") -> None:
		super().__init__()
		if str(type).lower() != "residual_spatiotemporal":
			raise ValueError(f"Expected post_fusion_backbone.type='residual_spatiotemporal', got {type!r}.")
		self.blocks = nn.Sequential(
			*(
				ResidualSpatiotemporalBlock(
					int(dim),
					temporal_kernel_size=int(temporal_kernel_size),
					spatial_kernel_size=int(spatial_kernel_size),
					dropout=float(dropout),
					norm=str(norm),
					activation=str(activation),
					residual=bool(residual),
				)
				for _ in range(max(0, int(num_blocks)))
			)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.blocks(x)


class MultiScaleContextBackbone(nn.Module):
	"""Residual dilated spatial context applied independently at each timestep."""

	def __init__(self, *, dim: int = 64, hidden_dim: int | None = None, type: str = "multiscale_context") -> None:
		super().__init__()
		if str(type).lower() != "multiscale_context":
			raise ValueError(f"Expected post_fusion_backbone.type='multiscale_context', got {type!r}.")
		channels = int(dim)
		branch_channels = int(hidden_dim) if hidden_dim not in (None, 0) else max(1, channels // 2)
		self.dilations = (1, 2, 4)
		self.branches = nn.ModuleList(
			[
				nn.Sequential(
					nn.Conv2d(channels, branch_channels, kernel_size=3, padding=dilation, dilation=dilation),
					nn.GroupNorm(_group_count(branch_channels), branch_channels),
					nn.SiLU(inplace=True),
				)
				for dilation in self.dilations
			]
		)
		self.projection = nn.Sequential(
			nn.Conv2d(branch_channels * len(self.dilations), channels, kernel_size=1),
			nn.GroupNorm(_group_count(channels), channels),
		)
		self.activation = nn.SiLU(inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"MultiScaleContextBackbone expects B x T x D x H x W, got {tuple(x.shape)}.")
		batch, time_steps, channels, height, width = (int(value) for value in x.shape)
		flat = x.reshape(batch * time_steps, channels, height, width)
		context = self.projection(torch.cat([branch(flat) for branch in self.branches], dim=1))
		return self.activation(flat + context).reshape(batch, time_steps, channels, height, width)



def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
	"""Partition a padded B x H x W x D tensor into local spatial windows."""
	if x.ndim != 4:
		raise ValueError(f"window_partition expects B x H x W x D, got {tuple(x.shape)}.")
	batch, height, width, dim = (int(value) for value in x.shape)
	window = int(window_size)
	if height % window or width % window:
		raise ValueError(f"Padded spatial shape {(height, width)} must be divisible by window_size={window}.")
	return x.reshape(batch, height // window, window, width // window, window, dim).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window * window, dim)


def window_reverse(windows: torch.Tensor, window_size: int, batch: int, height: int, width: int) -> torch.Tensor:
	"""Reverse :func:`window_partition` to B x H x W x D."""
	window = int(window_size)
	return windows.reshape(int(batch), int(height) // window, int(width) // window, window, window, int(windows.shape[-1])).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(int(batch), int(height), int(width), int(windows.shape[-1]))


class LocalWindowAttentionBlock(nn.Module):
	"""Per-timestep non-overlapping window attention with automatic padding."""

	def __init__(self, dim: int, *, num_heads: int = 4, window_size: int = 8, mlp_ratio: float = 2.0, dropout: float = 0.1) -> None:
		super().__init__()
		self.dim = int(dim); self.window_size = int(window_size)
		if self.dim % int(num_heads): raise ValueError(f"num_heads={num_heads} must divide dim={self.dim}.")
		if self.window_size < 1: raise ValueError("window_size must be positive.")
		hidden = max(1, int(round(self.dim * float(mlp_ratio))))
		self.norm1 = nn.LayerNorm(self.dim)
		self.attention = nn.MultiheadAttention(self.dim, int(num_heads), dropout=float(dropout), batch_first=True)
		self.dropout = nn.Dropout(float(dropout)); self.norm2 = nn.LayerNorm(self.dim)
		self.mlp = nn.Sequential(nn.Linear(self.dim, hidden), nn.GELU(), nn.Dropout(float(dropout)), nn.Linear(hidden, self.dim), nn.Dropout(float(dropout)))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5: raise ValueError(f"LocalWindowAttentionBlock expects B x T x D x H x W, got {tuple(x.shape)}.")
		batch, time_steps, dim, height, width = (int(value) for value in x.shape)
		if dim != self.dim: raise ValueError(f"LocalWindowAttentionBlock expected D={self.dim}, got {dim}.")
		pad_h = (-height) % self.window_size; pad_w = (-width) % self.window_size
		flat = F.pad(x.reshape(batch * time_steps, dim, height, width), (0, pad_w, 0, pad_h))
		padded_h, padded_w = height + pad_h, width + pad_w
		windows = window_partition(flat.permute(0, 2, 3, 1).contiguous(), self.window_size)
		normalized = self.norm1(windows)
		attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
		windows = windows + self.dropout(attended)
		windows = windows + self.mlp(self.norm2(windows))
		grid = window_reverse(windows, self.window_size, batch * time_steps, padded_h, padded_w)[:, :height, :width]
		return grid.permute(0, 3, 1, 2).contiguous().reshape(batch, time_steps, dim, height, width)


class LocalWindowAttentionBackbone(nn.Module):
	"""Shape-preserving stack of local spatial attention blocks."""

	def __init__(self, *, dim: int = 64, num_blocks: int = 2, num_heads: int = 4, window_size: int = 8, mlp_ratio: float = 2.0, dropout: float = 0.1) -> None:
		super().__init__(); self.local_window_attention_enabled = True; self.window_size = int(window_size)
		self.blocks = nn.ModuleList([LocalWindowAttentionBlock(int(dim), num_heads=int(num_heads), window_size=self.window_size, mlp_ratio=float(mlp_ratio), dropout=float(dropout)) for _ in range(max(0, int(num_blocks)))])

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		for block in self.blocks: x = block(x)
		return x


def cuboid_partition(x: torch.Tensor, cuboid_size: Sequence[int]) -> torch.Tensor:
	"""Partition padded B x T x H x W x D features into space-time cuboids."""
	if x.ndim != 5: raise ValueError(f"cuboid_partition expects B x T x H x W x D, got {tuple(x.shape)}.")
	ct, ch, cw = (int(value) for value in cuboid_size)
	batch, time_steps, height, width, dim = (int(value) for value in x.shape)
	if time_steps % ct or height % ch or width % cw: raise ValueError(f"Padded shape {(time_steps, height, width)} must be divisible by cuboid {(ct, ch, cw)}.")
	return x.reshape(batch, time_steps // ct, ct, height // ch, ch, width // cw, cw, dim).permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().reshape(-1, ct * ch * cw, dim)


def cuboid_reverse(windows: torch.Tensor, cuboid_size: Sequence[int], batch: int, time_steps: int, height: int, width: int) -> torch.Tensor:
	"""Reverse :func:`cuboid_partition` to B x T x H x W x D."""
	ct, ch, cw = (int(value) for value in cuboid_size)
	return windows.reshape(int(batch), int(time_steps) // ct, int(height) // ch, int(width) // cw, ct, ch, cw, int(windows.shape[-1])).permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().reshape(int(batch), int(time_steps), int(height), int(width), int(windows.shape[-1]))


class CuboidAttentionBlock(nn.Module):
	"""Local joint space-time attention over padded cuboids."""

	def __init__(self, dim: int, *, num_heads: int = 4, cuboid_size: Sequence[int] = (2, 8, 8), shifted: bool = False, mlp_ratio: float = 2.0, dropout: float = 0.1) -> None:
		super().__init__(); self.dim = int(dim); self.cuboid_size = tuple(int(value) for value in cuboid_size); self.shifted = bool(shifted)
		if len(self.cuboid_size) != 3 or any(value < 1 for value in self.cuboid_size): raise ValueError(f"cuboid_size must contain three positive integers, got {cuboid_size}.")
		if self.dim % int(num_heads): raise ValueError(f"num_heads={num_heads} must divide dim={self.dim}.")
		hidden = max(1, int(round(self.dim * float(mlp_ratio))))
		self.norm1 = nn.LayerNorm(self.dim); self.attention = nn.MultiheadAttention(self.dim, int(num_heads), dropout=float(dropout), batch_first=True); self.dropout = nn.Dropout(float(dropout)); self.norm2 = nn.LayerNorm(self.dim)
		self.mlp = nn.Sequential(nn.Linear(self.dim, hidden), nn.GELU(), nn.Dropout(float(dropout)), nn.Linear(hidden, self.dim), nn.Dropout(float(dropout)))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5: raise ValueError(f"CuboidAttentionBlock expects B x T x D x H x W, got {tuple(x.shape)}.")
		batch, time_steps, dim, height, width = (int(value) for value in x.shape)
		if dim != self.dim: raise ValueError(f"CuboidAttentionBlock expected D={self.dim}, got {dim}.")
		ct = min(self.cuboid_size[0], time_steps); ch, cw = self.cuboid_size[1:]
		pad_t, pad_h, pad_w = (-time_steps) % ct, (-height) % ch, (-width) % cw
		grid = F.pad(x.permute(0, 2, 1, 3, 4).contiguous(), (0, pad_w, 0, pad_h, 0, pad_t)).permute(0, 2, 3, 4, 1).contiguous()
		padded_shape = (time_steps + pad_t, height + pad_h, width + pad_w)
		shift = (ct // 2, ch // 2, cw // 2) if self.shifted else (0, 0, 0)
		if self.shifted: grid = torch.roll(grid, shifts=tuple(-value for value in shift), dims=(1, 2, 3))
		windows = cuboid_partition(grid, (ct, ch, cw)); normalized = self.norm1(windows)
		attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
		windows = windows + self.dropout(attended); windows = windows + self.mlp(self.norm2(windows))
		grid = cuboid_reverse(windows, (ct, ch, cw), batch, *padded_shape)
		if self.shifted: grid = torch.roll(grid, shifts=shift, dims=(1, 2, 3))
		grid = grid[:, :time_steps, :height, :width]
		return grid.permute(0, 1, 4, 2, 3).contiguous()


class CuboidAttentionLiteBackbone(nn.Module):
	"""Lightweight Earthformer-style alternating regular/shifted cuboid attention."""

	def __init__(self, *, dim: int = 64, num_blocks: int = 2, num_heads: int = 4, cuboid_size: Sequence[int] = (2, 8, 8), use_shifted_cuboids: bool = True, mlp_ratio: float = 2.0, dropout: float = 0.1) -> None:
		super().__init__(); self.cuboid_size = tuple(int(value) for value in cuboid_size); self.use_shifted_cuboids = bool(use_shifted_cuboids)
		self.blocks = nn.ModuleList([CuboidAttentionBlock(int(dim), num_heads=int(num_heads), cuboid_size=self.cuboid_size, shifted=self.use_shifted_cuboids and index % 2 == 1, mlp_ratio=float(mlp_ratio), dropout=float(dropout)) for index in range(max(0, int(num_blocks)))])

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		for block in self.blocks: x = block(x)
		return x


class SpectralConv2d(nn.Module):
	"""Learned low-frequency 2D Fourier transform for one timestep."""

	def __init__(self, dim: int, *, modes_h: int = 16, modes_w: int = 16) -> None:
		super().__init__(); self.dim = int(dim); self.modes_h = int(modes_h); self.modes_w = int(modes_w)
		if self.modes_h < 1 or self.modes_w < 1: raise ValueError("modes_h and modes_w must be positive.")
		self.weight = nn.Parameter(torch.randn(self.dim, self.dim, self.modes_h, self.modes_w, 2) / max(1, self.dim))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 4 or int(x.shape[1]) != self.dim: raise ValueError(f"SpectralConv2d expects B x {self.dim} x H x W, got {tuple(x.shape)}.")
		original_dtype = x.dtype; frequency = torch.fft.rfft2(x.float(), norm="ortho")
		mh = min(self.modes_h, int(frequency.shape[-2])); mw = min(self.modes_w, int(frequency.shape[-1])); output = torch.zeros_like(frequency)
		weight = torch.view_as_complex(self.weight[:, :, :mh, :mw].contiguous())
		output[:, :, :mh, :mw] = torch.einsum("bihw,iohw->bohw", frequency[:, :, :mh, :mw], weight)
		return torch.fft.irfft2(output, s=x.shape[-2:], norm="ortho").to(dtype=original_dtype)


class SpectralFourierBlock(nn.Module):
	"""Fourier global mixing plus a local convolutional residual branch."""

	def __init__(self, dim: int, *, modes_h: int = 16, modes_w: int = 16, residual: bool = True, dropout: float = 0.1) -> None:
		super().__init__(); self.residual = bool(residual); self.spectral = SpectralConv2d(int(dim), modes_h=int(modes_h), modes_w=int(modes_w)); self.local = nn.Conv2d(int(dim), int(dim), kernel_size=3, padding=1); self.dropout = nn.Dropout2d(float(dropout)); self.norm = nn.GroupNorm(_group_count(int(dim)), int(dim)); self.activation = nn.SiLU(inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5: raise ValueError(f"SpectralFourierBlock expects B x T x D x H x W, got {tuple(x.shape)}.")
		batch, time_steps, dim, height, width = (int(value) for value in x.shape); flat = x.reshape(batch * time_steps, dim, height, width)
		mixed = self.dropout(self.spectral(flat) + self.local(flat)); mixed = flat + mixed if self.residual else mixed
		return self.activation(self.norm(mixed)).reshape(batch, time_steps, dim, height, width)


class SpectralFourierBackbone(nn.Module):
	"""Shape-preserving per-timestep spectral post-fusion backbone."""

	def __init__(self, *, dim: int = 64, num_blocks: int = 2, modes_h: int = 16, modes_w: int = 16, residual: bool = True, dropout: float = 0.1) -> None:
		super().__init__(); self.blocks = nn.ModuleList([SpectralFourierBlock(int(dim), modes_h=int(modes_h), modes_w=int(modes_w), residual=bool(residual), dropout=float(dropout)) for _ in range(max(0, int(num_blocks)))])

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		for block in self.blocks: x = block(x)
		return x


class SpatiotemporalMambaBlock(nn.Module):
	"""Official Mamba temporal scan followed by a spatial raster scan."""

	def __init__(self, dim: int, *, expansion: int = 2, dropout: float = 0.1, backend: str = "mamba_ssm", d_state: int = 16, d_conv: int = 4, scan_order: str = "temporal_then_spatial") -> None:
		super().__init__()
		if str(scan_order).lower() != "temporal_then_spatial": raise ValueError(f"SpatiotemporalMambaBlock supports only scan_order='temporal_then_spatial', got {scan_order!r}.")
		if str(backend).lower() != "mamba_ssm": raise ValueError("The CAWFE-Latte Mamba ablation requires backend='mamba_ssm'; fallback surrogates are not allowed.")
		self.dim = int(dim); self.scan_order = "temporal_then_spatial"; self.backend_name = "mamba_ssm"
		self.temporal_norm = nn.LayerNorm(self.dim); self.spatial_norm = nn.LayerNorm(self.dim)
		self.temporal_mamba = build_mamba_layer(self.dim, int(d_state), int(d_conv), int(expansion), "mamba_ssm")
		self.spatial_mamba = build_mamba_layer(self.dim, int(d_state), int(d_conv), int(expansion), "mamba_ssm")
		self.dropout = nn.Dropout(float(dropout))

	@staticmethod
	def to_temporal_sequences(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int, int]]:
		if x.ndim != 5: raise ValueError(f"Expected B x T x D x H x W, got {tuple(x.shape)}.")
		shape = tuple(int(value) for value in x.shape)
		batch, time_steps, dim, height, width = shape
		return x.permute(0, 3, 4, 1, 2).contiguous().reshape(batch * height * width, time_steps, dim), shape

	@staticmethod
	def from_temporal_sequences(sequence: torch.Tensor, shape: tuple[int, int, int, int, int]) -> torch.Tensor:
		batch, time_steps, dim, height, width = shape
		return sequence.reshape(batch, height, width, time_steps, dim).permute(0, 3, 4, 1, 2).contiguous()

	@staticmethod
	def to_spatial_sequences(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int, int]]:
		if x.ndim != 5: raise ValueError(f"Expected B x T x D x H x W, got {tuple(x.shape)}.")
		shape = tuple(int(value) for value in x.shape)
		batch, time_steps, dim, height, width = shape
		return x.permute(0, 1, 3, 4, 2).contiguous().reshape(batch * time_steps, height * width, dim), shape

	@staticmethod
	def from_spatial_sequences(sequence: torch.Tensor, shape: tuple[int, int, int, int, int]) -> torch.Tensor:
		batch, time_steps, dim, height, width = shape
		return sequence.reshape(batch, time_steps, height, width, dim).permute(0, 1, 4, 2, 3).contiguous()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5: raise ValueError(f"SpatiotemporalMambaBlock expects B x T x D x H x W, got {tuple(x.shape)}.")
		if int(x.shape[2]) != self.dim: raise ValueError(f"SpatiotemporalMambaBlock expected D={self.dim}, got {int(x.shape[2])}.")
		temporal, shape = self.to_temporal_sequences(x)
		temporal = temporal + self.dropout(self.temporal_mamba(self.temporal_norm(temporal)))
		x = self.from_temporal_sequences(temporal, shape)
		spatial, shape = self.to_spatial_sequences(x)
		spatial = spatial + self.dropout(self.spatial_mamba(self.spatial_norm(spatial)))
		return self.from_spatial_sequences(spatial, shape)


class SpatiotemporalMambaBackbone(nn.Module):
	"""Factorized temporal-then-spatial stack using the official Mamba backend."""

	def __init__(self, *, dim: int = 64, num_blocks: int = 2, expansion: int = 2, dropout: float = 0.1, backend: str = "mamba_ssm", d_state: int = 16, d_conv: int = 4, scan_order: str = "temporal_then_spatial") -> None:
		super().__init__(); self.backend_name = str(backend).lower(); self.scan_order = str(scan_order).lower()
		self.blocks = nn.ModuleList([SpatiotemporalMambaBlock(int(dim), expansion=int(expansion), dropout=float(dropout), backend=backend, d_state=int(d_state), d_conv=int(d_conv), scan_order=scan_order) for _ in range(max(0, int(num_blocks)))])

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		for block in self.blocks: x = block(x)
		return x


class TemporalCNNBackbone(nn.Module):
	"""Small temporal CNN backbone over fused sequence features."""

	def __init__(
		self,
		*,
		dim: int = 64,
		num_blocks: int = 3,
		temporal_kernel_size: int = 3,
		spatial_kernel_size: int = 3,
		dropout: float = 0.1,
		residual: bool = True,
		type: str = "temporal_cnn",
		norm: str = "groupnorm",
		activation: str = "silu",
	) -> None:
		super().__init__()
		if str(type).lower() != "temporal_cnn":
			raise ValueError(f"CAWFE-Latte v1 supports only backbone.type='temporal_cnn', got {type!r}.")
		if str(norm).lower() != "groupnorm":
			raise ValueError(f"CAWFE-Latte v1 supports only backbone.norm='groupnorm', got {norm!r}.")
		if str(activation).lower() != "silu":
			raise ValueError(f"CAWFE-Latte v1 supports only backbone.activation='silu', got {activation!r}.")
		self.blocks = nn.Sequential(
			*(
				TemporalSpatialResidualBlock(
					int(dim),
					temporal_kernel_size=int(temporal_kernel_size),
					spatial_kernel_size=int(spatial_kernel_size),
					dropout=float(dropout),
					residual=bool(residual),
				)
				for _ in range(max(0, int(num_blocks)))
			)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.blocks(x)


class TemporalAggregator(nn.Module):
	"""Aggregate B x T x C x H x W features to B x C x H x W."""

	def __init__(self, *, mode: str = "last", dim: int = 64, input_sequence_length: int = 5, hidden_dim: int | None = None) -> None:
		super().__init__()
		self.mode = str(mode).lower()
		self.dim = int(dim)
		if self.mode not in {"last", "mean", "attention_pool"}:
			raise ValueError(f"temporal_aggregation.mode must be last, mean, or attention_pool; got {mode!r}.")
		self.attention_hidden = nn.Sequential(nn.Conv2d(self.dim, int(hidden_dim), 1), nn.SiLU(), nn.Conv2d(int(hidden_dim), 1, 1)) if self.mode == "attention_pool" and hidden_dim not in (None, 0) else None
		self.attention_score = nn.Conv2d(self.dim, 1, kernel_size=1) if self.mode == "attention_pool" and self.attention_hidden is None else None
		self.last_attention_weights: torch.Tensor | None = None

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"TemporalAggregator expects B x T x C x H x W, got {tuple(x.shape)}.")
		if self.mode == "last":
			return x[:, -1]
		if self.mode == "mean":
			return x.mean(dim=1)
		scorer = self.attention_hidden if self.attention_hidden is not None else self.attention_score
		assert scorer is not None
		batch, time_steps, channels, height, width = (int(value) for value in x.shape)
		scores = scorer(x.reshape(batch * time_steps, channels, height, width)).reshape(batch, time_steps, 1, height, width)
		weights = torch.softmax(scores, dim=1)
		self.last_attention_weights = weights
		return (x * weights).sum(dim=1)



def build_post_fusion_backbone(config: Mapping[str, Any] | None, *, dim: int) -> nn.Module:
	"""Build the configurable post-fusion sequence backbone."""
	options = dict(config or {})
	kind = str(options.pop("type", "baseline_cnn")).lower()
	options.setdefault("dim", int(dim))
	if kind in {"baseline_cnn", "small_cnn", "temporal_cnn"}:
		options["type"] = "temporal_cnn"
		return TemporalCNNBackbone(**options)
	if kind == "residual_spatiotemporal":
		options["type"] = "residual_spatiotemporal"
		return ResidualSpatiotemporalBackbone(**options)
	if kind == "multiscale_context":
		multiscale_options = {key: value for key, value in options.items() if key in {"dim", "hidden_dim"}}
		multiscale_options["type"] = "multiscale_context"
		return MultiScaleContextBackbone(**multiscale_options)
	if kind == "local_window_attention":
		return LocalWindowAttentionBackbone(**{key: value for key, value in options.items() if key in {"dim", "num_blocks", "num_heads", "window_size", "mlp_ratio", "dropout"}})
	if kind == "cuboid_attention_lite":
		return CuboidAttentionLiteBackbone(**{key: value for key, value in options.items() if key in {"dim", "num_blocks", "num_heads", "cuboid_size", "use_shifted_cuboids", "mlp_ratio", "dropout"}})
	if kind == "spectral_fourier":
		return SpectralFourierBackbone(**{key: value for key, value in options.items() if key in {"dim", "num_blocks", "modes_h", "modes_w", "residual", "dropout"}})
	if kind == "spatiotemporal_mamba":
		return SpatiotemporalMambaBackbone(**{key: value for key, value in options.items() if key in {"dim", "num_blocks", "expansion", "dropout", "backend", "d_state", "d_conv", "scan_order"}})
	raise ValueError(f"Unsupported cawfe_latte.post_fusion_backbone.type: {kind!r}.")


class TemporalAttentionPooling(nn.Module):
	"""Learn pixelwise temporal weights and reduce B,T,D,H,W to B,D,H,W."""

	def __init__(self, *, dim: int = 64, hidden_dim: int | None = None) -> None:
		super().__init__()
		hidden_channels = int(hidden_dim) if hidden_dim not in (None, 0) else max(1, int(dim) // 4)
		self.score = nn.Sequential(
			nn.Conv2d(int(dim), hidden_channels, kernel_size=1),
			nn.SiLU(inplace=True),
			nn.Conv2d(hidden_channels, 1, kernel_size=1),
		)
		final_score = self.score[-1]
		assert isinstance(final_score, nn.Conv2d)
		nn.init.zeros_(final_score.weight)
		nn.init.zeros_(final_score.bias)
		self.last_attention_weights: torch.Tensor | None = None

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"TemporalAttentionPooling expects B x T x D x H x W, got {tuple(x.shape)}.")
		batch, time_steps, channels, height, width = (int(value) for value in x.shape)
		scores = self.score(x.reshape(batch * time_steps, channels, height, width))
		scores = scores.reshape(batch, time_steps, 1, height, width)
		alpha = torch.softmax(scores, dim=1)
		self.last_attention_weights = alpha
		return (alpha * x).sum(dim=1)


def build_temporal_pooling(config: Mapping[str, Any] | None, *, dim: int, input_sequence_length: int) -> nn.Module:
	"""Build baseline last-frame or pixelwise attention temporal pooling."""
	options = dict(config or {})
	kind = str(options.get("type", options.get("mode", "baseline"))).lower()
	if kind in {"baseline", "last"}:
		return TemporalAggregator(mode="last", dim=int(dim), input_sequence_length=int(input_sequence_length))
	if kind == "mean":
		return TemporalAggregator(mode="mean", dim=int(dim), input_sequence_length=int(input_sequence_length))
	if kind in {"attention", "attention_pool"}:
		return TemporalAttentionPooling(dim=int(dim), hidden_dim=options.get("hidden_dim"))
	raise ValueError(f"Unsupported cawfe_latte.temporal_pooling.type: {kind!r}.")


def build_regression_activation(config: Mapping[str, Any] | None) -> nn.Module:
	kind = str(dict(config or {}).get("activation", "none")).lower()
	if kind in {"none", "identity"}:
		return nn.Identity()
	if kind == "softplus":
		return nn.Softplus()
	raise ValueError(f"Unsupported cawfe_latte.regression.activation: {kind!r}.")


class SupportGate(nn.Module):
	"""Separate learned support map that gates continuous outputs only."""
	def __init__(self, *, enabled: bool, dim: int, source: str = "separate_support_head", gate_min: float = 0.05) -> None:
		super().__init__()
		self.enabled = bool(enabled); self.gate_min = float(gate_min)
		if not 0.0 <= self.gate_min <= 1.0: raise ValueError("support_gate.gate_min must be in [0, 1].")
		if self.enabled and str(source).lower() != "separate_support_head": raise ValueError("Enabled support_gate requires source='separate_support_head'.")
		self.head = nn.Conv2d(int(dim), 1, 1) if self.enabled else None
		self.last_gate: torch.Tensor | None = None
	def forward(self, decoded: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
		if self.head is None: return None, None
		logits = self.head(decoded)
		gate = self.gate_min + (1.0 - self.gate_min) * torch.sigmoid(logits)
		self.last_gate = gate
		return logits, gate


def build_support_gate(config: Mapping[str, Any] | None, *, dim: int) -> SupportGate:
	options = dict(config or {})
	return SupportGate(enabled=bool(options.get("enabled", False)), dim=int(dim), source=str(options.get("source", "separate_support_head")), gate_min=float(options.get("gate_min", 0.05)))


class ShallowDecoder(nn.Module):
	"""Small same-resolution decoder for CAWFE-Latte v1."""

	def __init__(
		self,
		*,
		in_dim: int = 64,
		hidden_dim: int = 64,
		num_blocks: int = 2,
		dropout: float = 0.1,
		type: str = "shallow_cnn",
		norm: str = "groupnorm",
		activation: str = "silu",
	) -> None:
		super().__init__()
		if str(type).lower() != "shallow_cnn":
			raise ValueError(f"CAWFE-Latte v1 supports only decoder.type='shallow_cnn', got {type!r}.")
		if str(norm).lower() != "groupnorm":
			raise ValueError(f"CAWFE-Latte v1 supports only decoder.norm='groupnorm', got {norm!r}.")
		if str(activation).lower() != "silu":
			raise ValueError(f"CAWFE-Latte v1 supports only decoder.activation='silu', got {activation!r}.")
		layers: list[nn.Module] = [ConvBlock2d(int(in_dim), int(hidden_dim))]
		for _ in range(max(0, int(num_blocks))):
			layers.append(ResidualBlock2d(int(hidden_dim)))
		if float(dropout) > 0:
			layers.append(nn.Dropout2d(float(dropout)))
		self.net = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 4:
			raise ValueError(f"ShallowDecoder expects B x C x H x W, got {tuple(x.shape)}.")
		return self.net(x)


def _activation_from_name(name: str) -> nn.Module:
	key = str(name).lower()
	if key in {"none", "identity", "logits"}:
		return nn.Identity()
	if key == "relu":
		return nn.ReLU(inplace=True)
	if key == "softplus":
		return nn.Softplus()
	raise ValueError(f"Unsupported CAWFE-Latte head activation {name!r}; expected none, relu, softplus, or logits.")


class PatchFirePresenceHead(nn.Module):
	"""Classify whether a target patch contains any future active-fire pixel."""

	def __init__(self, *, dim: int = 64, hidden_dim: int = 32, dropout: float = 0.1) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(int(dim), int(hidden_dim)),
			nn.SiLU(inplace=True),
			nn.Dropout(float(dropout)),
			nn.Linear(int(hidden_dim), 1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 4:
			raise ValueError(f"PatchFirePresenceHead expects B x D x H x W, got {tuple(x.shape)}.")
		return self.net(x)


class PredictionHead(nn.Module):
	"""Single-channel prediction head with optional activation."""

	def __init__(self, in_dim: int, *, activation: str = "none") -> None:
		super().__init__()
		self.proj = nn.Conv2d(int(in_dim), 1, kernel_size=1)
		self.activation = _activation_from_name(activation)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.activation(self.proj(x))


class GradientReversalFunction(torch.autograd.Function):
	"""Identity in the forward pass and sign-scaled reversal in backward."""

	@staticmethod
	def forward(ctx, x: torch.Tensor, lambda_grl: float) -> torch.Tensor:
		ctx.lambda_grl = float(lambda_grl)
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
		return -ctx.lambda_grl * grad_output, None


def gradient_reverse(x: torch.Tensor, lambda_grl: float = 1.0) -> torch.Tensor:
	return GradientReversalFunction.apply(x, float(lambda_grl))


class DomainAdversarialHead(nn.Module):
	"""Training-only fire-domain classifier attached through a GRL."""

	def __init__(self, dim: int, num_domains: int, hidden_dim: int = 64, lambda_grl: float = 1.0, dropout: float = 0.1) -> None:
		super().__init__()
		if int(num_domains) < 1:
			raise ValueError("domain_adversarial.num_domains must be positive.")
		self.lambda_grl = float(lambda_grl)
		self.classifier = nn.Sequential(
			nn.Linear(int(dim), int(hidden_dim)),
			nn.SiLU(inplace=True),
			nn.Dropout(float(dropout)),
			nn.Linear(int(hidden_dim), int(num_domains)),
		)

	def forward(self, pooled: torch.Tensor) -> torch.Tensor:
		return self.classifier(gradient_reverse(pooled, self.lambda_grl))


class MaskGuidedRegressionAttention(nn.Module):
	"""Soft mask-feature attention that is exactly parent-equivalent at alpha=0."""

	def __init__(self, mask_dim: int, alpha_init: float = 0.0, attention_activation: str = "sigmoid") -> None:
		super().__init__()
		if str(attention_activation).lower() != "sigmoid":
			raise ValueError("mask_guided_regression.attention_activation must be 'sigmoid'.")
		self.projection = nn.Conv2d(int(mask_dim), 1, kernel_size=1)
		self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

	def forward(self, regression_features: torch.Tensor, mask_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		attention = torch.sigmoid(self.projection(mask_features.detach()))
		return regression_features * (1.0 + self.alpha * attention), attention


class RegressionMixtureOfExperts(nn.Module):
	"""Three-way (configurable) soft mixture of shallow regression decoders."""

	def __init__(self, dim: int, decoder_config: Mapping[str, Any], num_experts: int = 3, routing: str = "soft") -> None:
		super().__init__()
		if str(routing).lower() != "soft":
			raise ValueError("regression_moe.routing currently supports only 'soft'.")
		if int(num_experts) < 2:
			raise ValueError("regression_moe.num_experts must be at least 2.")
		self.num_experts = int(num_experts)
		options = dict(decoder_config)
		self.experts = nn.ModuleList(
			ShallowDecoder(type="shallow_cnn", **options) for _ in range(self.num_experts)
		)
		router_hidden_dim = max(1, int(dim) // 2)
		self.router = nn.Sequential(
			nn.Linear(int(dim), router_hidden_dim),
			nn.SiLU(inplace=True),
			nn.Linear(router_hidden_dim, self.num_experts),
		)

	def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		context = F.adaptive_avg_pool2d(latent, 1).flatten(1)
		weights = torch.softmax(self.router(context), dim=1)
		expert_features = torch.stack([expert(latent) for expert in self.experts], dim=1)
		mixture = torch.sum(expert_features * weights[:, :, None, None, None], dim=1)
		return mixture, weights


class ContrastiveProjectionHead(nn.Module):
	def __init__(self, dim: int, projection_dim: int = 64, hidden_dim: int = 128) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(int(dim), int(hidden_dim)),
			nn.SiLU(inplace=True),
			nn.Linear(int(hidden_dim), int(projection_dim)),
		)

	def forward(self, pooled: torch.Tensor) -> torch.Tensor:
		return F.normalize(self.net(pooled), p=2, dim=1)


class PhysicalStateAuxHead(nn.Module):
	def __init__(self, dim: int, hidden_dim: int = 64) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(int(dim), int(hidden_dim)),
			nn.SiLU(inplace=True),
			nn.Linear(int(hidden_dim), 3),
		)

	def forward(self, pooled: torch.Tensor) -> torch.Tensor:
		prediction = self.net(pooled)
		return torch.cat([torch.sigmoid(prediction[:, :1]), prediction[:, 1:]], dim=1)


class CAWFELatte(nn.Module):
	"""Single configurable CAWFE-Latte architecture with named config ablations."""
	def __init__(self, *, input_channels: int = 129, input_sequence_length: int = 5, output_channels: int = 4, output_dim: int = 64, version: str = "v1_end_to_end", atmosphere: Mapping[str, Any] | None = None, wind: Mapping[str, Any] | None = None, fire_fuel: Mapping[str, Any] | None = None, flux_energy: Mapping[str, Any] | None = None, fusion: Mapping[str, Any] | None = None, alignment: Mapping[str, Any] | None = None, backbone: Mapping[str, Any] | None = None, temporal_aggregation: Mapping[str, Any] | None = None, post_fusion_backbone: Mapping[str, Any] | None = None, temporal_pooling: Mapping[str, Any] | None = None, regression: Mapping[str, Any] | None = None, support_gate: Mapping[str, Any] | None = None, ablation: Mapping[str, Any] | None = None, decoder: Mapping[str, Any] | None = None, heads: Mapping[str, Any] | None = None, auxiliary: Mapping[str, Any] | None = None, patch_fire_head: Mapping[str, Any] | None = None, domain_adversarial: Mapping[str, Any] | None = None, fire_mmd: Mapping[str, Any] | None = None, mask_guided_regression: Mapping[str, Any] | None = None, regression_moe: Mapping[str, Any] | None = None, supervised_contrastive: Mapping[str, Any] | None = None, physical_state_aux: Mapping[str, Any] | None = None, use_terrain_conditioning: bool = False, terrain_encoder: Mapping[str, Any] | None = None, terrain_film: Mapping[str, Any] | None = None, terrain_integration: Mapping[str, Any] | None = None, channel_names: Mapping[int, str] | None = None, debug_prediction_head: bool = False) -> None:
		super().__init__()
		self.input_channels=int(input_channels); self.input_sequence_length=int(input_sequence_length); self.output_channels=int(output_channels); self.output_dim=int(output_dim); self.version=str(version); self.debug_prediction_head=bool(debug_prediction_head)
		self.ablation_name=str(dict(ablation or {}).get("name", "baseline"))
		if self.output_channels != 4: raise ValueError(f"CAWFE-Latte expects output_channels=4, got {self.output_channels}.")
		if self.input_channels < 86: raise ValueError(f"CAWFE-Latte requires at least 86 input channels, got {self.input_channels}.")
		atmosphere_config=dict(atmosphere or {}); wind_config=dict(wind or {}); fire_config=dict(fire_fuel or {}); flux_config=dict(flux_energy or {}); fusion_config=dict(fusion or {}); alignment_config=dict(alignment or {}); decoder_config=dict(decoder or {}); head_config=dict(heads or {}); auxiliary_config=dict(auxiliary or {}); patch_fire_config=dict(patch_fire_head or {}); domain_config=dict(domain_adversarial or {}); mmd_config=dict(fire_mmd or {}); mask_guidance_config=dict(mask_guided_regression or {}); moe_config=dict(regression_moe or {}); contrastive_config=dict(supervised_contrastive or {}); physical_config=dict(physical_state_aux or {}); terrain_film_config=dict(terrain_film or {})
		legacy_terrain = bool(use_terrain_conditioning) and bool(terrain_film_config.get("enabled", True))
		self.terrain_mode = str(dict(terrain_integration or {}).get("mode", "postfusion_film" if legacy_terrain else "none")).lower()
		if self.terrain_mode not in {"none", "early_concat", "prefusion_film", "terrain_query", "late_concat", "postfusion_film"}:
			raise ValueError(f"Unsupported terrain integration mode: {self.terrain_mode!r}")
		self.use_terrain_conditioning = self.terrain_mode != "none"
		self.terrain_film_enabled = self.terrain_mode == "postfusion_film"
		terrain_encoder_config = dict(terrain_encoder or {})
		self.terrain_encoder = TerrainEncoder(**terrain_encoder_config) if self.terrain_mode in {"prefusion_film", "terrain_query", "late_concat", "postfusion_film"} else None
		self.terrain_film = TerrainFiLMConditioner(dim=int(terrain_film_config.get("dim", self.output_dim)), **{k:v for k,v in terrain_film_config.items() if k not in {"dim", "enabled"}}) if self.terrain_mode in {"prefusion_film", "postfusion_film"} else None
		self.terrain_query_projection = nn.Conv2d(self.output_dim, self.output_dim, 1) if self.terrain_mode == "terrain_query" else None
		self.terrain_late_projection = nn.Conv2d(2 * self.output_dim, self.output_dim, 1) if self.terrain_mode == "late_concat" else None
		if self.terrain_query_projection is not None:
			nn.init.zeros_(self.terrain_query_projection.weight); nn.init.zeros_(self.terrain_query_projection.bias)
		if self.terrain_late_projection is not None:
			nn.init.zeros_(self.terrain_late_projection.weight); nn.init.zeros_(self.terrain_late_projection.bias)
			with torch.no_grad():
				for channel in range(self.output_dim): self.terrain_late_projection.weight[channel, channel, 0, 0] = 1.0
		self.atmosphere_encoder=AtmosphereEncoder(out_dim=int(atmosphere_config.get("out_dim", self.output_dim)), **{k:v for k,v in atmosphere_config.items() if k != "out_dim"})
		self.wind_encoder=WindEncoder(input_channels=self.input_channels, out_dim=int(wind_config.get("out_dim", self.output_dim)), channel_names=channel_names, **{k:v for k,v in wind_config.items() if k != "out_dim"})
		self.fire_fuel_encoder=FireFuelEncoder(input_channels=self.input_channels, out_dim=int(fire_config.get("out_dim", self.output_dim)), channel_names=channel_names, terrain_channels=4 if self.terrain_mode == "early_concat" else 0, **{k:v for k,v in fire_config.items() if k != "out_dim"})
		self.flux_energy_encoder=FluxEnergyEncoder(input_channels=self.input_channels, out_dim=int(flux_config.get("out_dim", self.output_dim)), channel_names=channel_names, **{k:v for k,v in flux_config.items() if k != "out_dim"})
		if not bool(alignment_config.get("enabled", True)): raise ValueError("CAWFE-Latte multimodal alignment is mandatory before fusion.")
		spatial=dict(alignment_config.get("spatial", {})); distribution=dict(alignment_config.get("distribution", {})); temporal=dict(alignment_config.get("temporal", {}))
		self.alignment=MultimodalAlignment(dim=self.output_dim, max_time=int(temporal.get("max_time", max(16,self.input_sequence_length))), use_spatial_pos=bool(spatial.get("add_positional_embedding",True)), use_temporal_pos=bool(temporal.get("add_positional_embedding",True)), separate_layernorms=bool(distribution.get("separate_norm_per_modality",True)), eps=float(distribution.get("eps",1e-5)), learned_spatial_pos=bool(spatial.get("learned_positional_embedding",True)), learned_temporal_pos=bool(temporal.get("learned_positional_embedding",True)), flatten_order=str(spatial.get("flatten_order","row_major")))
		if int(fusion_config.get("dim",self.output_dim)) != self.output_dim: raise ValueError("fusion.dim must match output_dim for CAWFE-Latte.")
		self.fusion_type=str(fusion_config.get("type", "fire_query_attention")).lower()
		if self.fusion_type in {"fire_query_attention", "fire_query"}:
			self.fusion=FireQueryCrossAttentionFusion(dim=self.output_dim, num_heads=int(fusion_config.get("num_heads",4)), dropout=float(fusion_config.get("dropout",0.1)), use_layer_norm=bool(fusion_config.get("use_layer_norm",True)), residual=bool(fusion_config.get("residual",True)))
		elif self.fusion_type == "concat_projection":
			self.fusion=ConcatProjectionFusion(dim=self.output_dim)
		else:
			raise ValueError(f"Unsupported cawfe_latte.fusion.type: {self.fusion_type!r}.")
		# Start from the proven v1 sections, then overlay only the ablation switch.
		# baseline_cnn/baseline therefore retain the exact legacy module arguments.
		backbone_config=dict(backbone or {})
		if post_fusion_backbone is not None: backbone_config.update(dict(post_fusion_backbone))
		if str(backbone_config.get("type", "temporal_cnn")).lower() == "temporal_cnn": backbone_config["type"] = "baseline_cnn"
		self.post_fusion_backbone=build_post_fusion_backbone(backbone_config, dim=self.output_dim); self.temporal_backbone=self.post_fusion_backbone
		pooling_config=dict(temporal_aggregation or {})
		if temporal_pooling is not None: pooling_config.update(dict(temporal_pooling))
		if str(pooling_config.get("type", pooling_config.get("mode", "last"))).lower() == "last": pooling_config["type"] = "baseline"
		self.temporal_pooling=build_temporal_pooling(pooling_config, dim=self.output_dim, input_sequence_length=self.input_sequence_length); self.temporal_aggregator=self.temporal_pooling
		decoder_type=str(decoder_config.pop("type", "shared")).lower()
		decoder_config.setdefault("in_dim",self.output_dim); decoder_config.setdefault("hidden_dim",self.output_dim); decoder_out_dim=int(decoder_config["hidden_dim"])
		self.regression_moe_enabled=bool(moe_config.get("enabled", False))
		self.mask_guided_regression_enabled=bool(mask_guidance_config.get("enabled", False))
		if (self.regression_moe_enabled or self.mask_guided_regression_enabled) and decoder_type != "separate_regression":
			raise ValueError("regression_moe and mask_guided_regression require decoder.type='separate_regression'.")
		self.regression_moe=None
		if decoder_type in {"shared", "shallow_cnn"}:
			self.decoder=ShallowDecoder(type="shallow_cnn", **decoder_config); self.mask_decoder=None; self.regression_decoder=None
		elif decoder_type == "separate_regression":
			self.decoder=None; self.mask_decoder=ShallowDecoder(type="shallow_cnn", **decoder_config)
			if self.regression_moe_enabled:
				self.regression_decoder=None
				self.regression_moe=RegressionMixtureOfExperts(self.output_dim, decoder_config, num_experts=int(moe_config.get("num_experts", 3)), routing=str(moe_config.get("routing", "soft")))
			else:
				self.regression_decoder=ShallowDecoder(type="shallow_cnn", **decoder_config)
		else:
			raise ValueError(f"Unsupported cawfe_latte.decoder.type: {decoder_type!r}.")
		self.decoder_type="shared" if decoder_type == "shallow_cnn" else decoder_type
		regression_config=dict(regression or {}); self.regression_activation=build_regression_activation(regression_config)
		new_regression = regression is not None
		self.surface_head=PredictionHead(decoder_out_dim, activation="none" if new_regression else dict(head_config.get("surface",{})).get("activation","none")); self.canopy_head=PredictionHead(decoder_out_dim, activation="none" if new_regression else dict(head_config.get("canopy",{})).get("activation","none")); self.mask_head=PredictionHead(decoder_out_dim, activation=dict(head_config.get("mask",{})).get("activation","logits")); self.energy_head=PredictionHead(decoder_out_dim, activation="none" if new_regression else dict(head_config.get("energy",{})).get("activation","none"))
		if new_regression and regression_config.get("bias_init") is not None:
			for head in (self.surface_head,self.canopy_head,self.energy_head): nn.init.constant_(head.proj.bias, float(regression_config["bias_init"]))
		self.patch_fire_head_enabled=bool(patch_fire_config.get("enabled", False))
		self.patch_fire_head=PatchFirePresenceHead(dim=self.output_dim, hidden_dim=int(patch_fire_config.get("hidden_dim", max(1,self.output_dim//2))), dropout=float(patch_fire_config.get("dropout",0.1))) if self.patch_fire_head_enabled else None
		if support_gate is not None:
			self.support_gate=build_support_gate(support_gate, dim=decoder_out_dim); self.aux_fire_support_enabled=self.support_gate.enabled; self.aux_fire_support_detach_source=False; self.aux_fire_support_head=None
		else:
			legacy=dict(auxiliary_config.get("fire_support_head",{})); self.aux_fire_support_enabled=bool(legacy.get("enabled",True)); self.aux_fire_support_detach_source=bool(legacy.get("detach_source",False)); self.aux_fire_support_head=nn.Conv2d(self.output_dim,1,1) if self.aux_fire_support_enabled else None; self.support_gate=None
		# Initialize additions after every parent module so zero-alpha R is also
		# seed-for-seed parent-equivalent, not merely algebraically equivalent.
		self.mask_guided_regression=MaskGuidedRegressionAttention(decoder_out_dim, alpha_init=float(mask_guidance_config.get("alpha_init", 0.0)), attention_activation=str(mask_guidance_config.get("attention_activation", "sigmoid"))) if self.mask_guided_regression_enabled else None
		self.domain_adversarial_enabled=bool(domain_config.get("enabled", False))
		self.domain_adversarial_head=DomainAdversarialHead(self.output_dim, num_domains=int(domain_config.get("num_domains", 0)), hidden_dim=int(domain_config.get("hidden_dim", 64)), lambda_grl=float(domain_config.get("lambda_grl", 1.0)), dropout=float(domain_config.get("dropout", 0.1))) if self.domain_adversarial_enabled else None
		self.fire_mmd_enabled=bool(mmd_config.get("enabled", False))
		self.supervised_contrastive_enabled=bool(contrastive_config.get("enabled", False))
		self.contrastive_projection=ContrastiveProjectionHead(self.output_dim, projection_dim=int(contrastive_config.get("projection_dim", 64))) if self.supervised_contrastive_enabled else None
		self.physical_state_aux_enabled=bool(physical_config.get("enabled", False))
		self.physical_state_aux_head=PhysicalStateAuxHead(self.output_dim, hidden_dim=int(physical_config.get("hidden_dim", 64))) if self.physical_state_aux_enabled else None

	def _validate_input(self,x:torch.Tensor)->None:
		if x.ndim!=5: raise ValueError(f"CAWFE-Latte expects B x T x C x H x W input, got {tuple(x.shape)}.")
		if int(x.shape[1])!=self.input_sequence_length: raise ValueError(f"CAWFE-Latte expected T={self.input_sequence_length}, got T={int(x.shape[1])}.")
		if int(x.shape[2]) < 86: raise ValueError(f"CAWFE-Latte requires at least 86 input channels, got {int(x.shape[2])}.")
		if int(x.shape[2])<self.input_channels: raise ValueError(f"CAWFE-Latte was configured for input_channels={self.input_channels}, got {int(x.shape[2])}.")
	def forward(self,x:torch.Tensor,terrain:torch.Tensor|None=None,*,return_features:bool=False,return_attention:bool=False,return_aux:bool=False):
		self._validate_input(x)
		if self.use_terrain_conditioning:
			if terrain is None or terrain.ndim != 4 or terrain.shape[0] != x.shape[0] or terrain.shape[1] != 4 or terrain.shape[-2:] != x.shape[-2:]:
				raise ValueError(f"Terrain mode {self.terrain_mode} requires terrain B x 4 x H x W matching input.")
		terrain_embedding = self.terrain_encoder(terrain) if self.terrain_encoder is not None else None
		atmosphere=self.atmosphere_encoder(x); wind=self.wind_encoder(x)
		if self.terrain_mode == "early_concat":
			fire_input = torch.cat((x, terrain[:, None].expand(-1, x.shape[1], -1, -1, -1)), dim=2)
		else: fire_input = x
		fire_fuel=self.fire_fuel_encoder(fire_input); flux_energy=self.flux_energy_encoder(x)
		if self.terrain_mode == "prefusion_film": fire_fuel=self.terrain_film(fire_fuel, terrain_embedding)
		aligned=self.alignment(atmosphere,wind,fire_fuel,flux_energy)
		if self.terrain_mode == "terrain_query":
			query_terrain = self.terrain_query_projection(terrain_embedding)
			aligned["fire_fuel"] = aligned["fire_fuel"] + self.alignment.to_tokens(query_terrain[:, None])
		attention=None
		if self.fusion_type in {"fire_query_attention", "fire_query"}:
			fusion_result=self.fusion(aligned["atmosphere"],aligned["wind"],aligned["fire_fuel"],aligned["flux_energy"],return_attention=return_attention); fused_tokens,attention=fusion_result if return_attention else (fusion_result,None); z_dynamic=tokens_to_grid(fused_tokens,aligned["spatial_shape"])
		else:
			aligned_grids=[tokens_to_grid(aligned[name],aligned["spatial_shape"]) for name in ("atmosphere","wind","fire_fuel","flux_energy")]; z_dynamic=self.fusion(*aligned_grids); fused_tokens=self.alignment.to_tokens(z_dynamic)
		z_terrain=z_dynamic
		if self.terrain_film_enabled:
			z_terrain=self.terrain_film(z_dynamic,terrain_embedding)
		elif self.terrain_mode == "late_concat":
			terrain_time = terrain_embedding[:, None].expand(-1, z_dynamic.shape[1], -1, -1, -1)
			z_terrain = _apply_per_timestep(torch.cat((z_dynamic, terrain_time), dim=2), self.terrain_late_projection)
		z_backbone=self.post_fusion_backbone(z_terrain); z_shared=self.temporal_pooling(z_backbone)
		router_weights=None; mask_guidance_attention=None
		if self.decoder_type == "shared":
			decoded=self.decoder(z_shared); mask_features=decoded; regression_features_before_guidance=decoded
		else:
			mask_features=self.mask_decoder(z_shared)
			if self.regression_moe is not None: regression_features_before_guidance,router_weights=self.regression_moe(z_shared)
			else: regression_features_before_guidance=self.regression_decoder(z_shared)
			decoded=regression_features_before_guidance
		regression_features=regression_features_before_guidance
		if self.mask_guided_regression is not None:
			regression_features,mask_guidance_attention=self.mask_guided_regression(regression_features_before_guidance,mask_features)
		surface=self.regression_activation(self.surface_head(regression_features)); canopy=self.regression_activation(self.canopy_head(regression_features)); energy=self.regression_activation(self.energy_head(regression_features)); mask_logits=self.mask_head(mask_features)
		patch_fire_logit=self.patch_fire_head(z_shared) if self.patch_fire_head is not None else None
		support_logits=None; gate=None
		if self.support_gate is not None: support_logits,gate=self.support_gate(regression_features)
		elif self.aux_fire_support_head is not None:
			aux_source=z_backbone[:,-1].detach() if self.aux_fire_support_detach_source else z_backbone[:,-1]; support_logits=self.aux_fire_support_head(aux_source)
		if gate is not None: surface=surface*gate; canopy=canopy*gate; energy=energy*gate
		prediction=torch.cat([surface,canopy,mask_logits,energy],dim=1)
		if return_features:
			features = {"prediction": prediction, "atmosphere": atmosphere, "wind": wind, "fire_fuel": fire_fuel, "flux_energy": flux_energy, "aligned_atmosphere": aligned["atmosphere"], "aligned_wind": aligned["wind"], "aligned_fire_fuel": aligned["fire_fuel"], "aligned_flux_energy": aligned["flux_energy"], "fused_tokens": fused_tokens, "spatial_shape": aligned["spatial_shape"], "fused_grid": z_dynamic, "fused": z_terrain, "fused_dynamic": z_dynamic, "terrain_features": terrain_embedding, "fused_after_terrain": z_terrain if self.use_terrain_conditioning else None, "local": z_backbone, "aggregated": z_shared, "decoded": decoded, "mask_features": mask_features, "regression_features_before_guidance": regression_features_before_guidance, "regression_features": regression_features, "z_dynamic": z_dynamic, "z_terrain": z_terrain, "z_backbone": z_backbone, "z_shared": z_shared}
			if support_logits is not None:
				features["support_logits" if self.support_gate is not None else "aux_fire_support_logits"] = support_logits
			if patch_fire_logit is not None: features["patch_fire_logit"] = patch_fire_logit
			if self.temporal_pooling.last_attention_weights is not None:
				features["temporal_attention_alpha"] = self.temporal_pooling.last_attention_weights
				features["temporal_attention"] = self.temporal_pooling.last_attention_weights
			if getattr(self.post_fusion_backbone, "local_window_attention_enabled", False):
				features["local_window_attention_enabled"] = True
				features["window_size"] = self.post_fusion_backbone.window_size
			if mask_guidance_attention is not None: features["mask_guidance_attention"] = mask_guidance_attention
			if router_weights is not None: features["router_weights"] = router_weights
			if return_attention: features["fusion_attention"] = attention
			return features
		outputs={"prediction": prediction}
		if self.support_gate is not None and support_logits is not None: outputs["support_logits"]=support_logits
		elif support_logits is not None: outputs["aux_fire_support_logits"]=support_logits
		if patch_fire_logit is not None: outputs["patch_fire_logit"]=patch_fire_logit
		if self.mask_guided_regression is not None and (self.training or return_aux):
			outputs["mask_guidance_alpha"]=self.mask_guided_regression.alpha
			outputs["mask_guidance_attention"]=mask_guidance_attention
		if router_weights is not None and (self.training or return_aux): outputs["router_weights"]=router_weights
		if self.training:
			pooled=F.adaptive_avg_pool2d(z_shared,1).flatten(1)
			if self.domain_adversarial_head is not None: outputs["domain_logits"]=self.domain_adversarial_head(pooled)
			if self.fire_mmd_enabled: outputs["fire_mmd_features"]=pooled
			if self.contrastive_projection is not None: outputs["contrastive_embedding"]=self.contrastive_projection(pooled)
		if self.physical_state_aux_head is not None and (self.training or return_aux):
			pooled=F.adaptive_avg_pool2d(z_shared,1).flatten(1)
			outputs["physical_state_prediction"]=self.physical_state_aux_head(pooled)
		return outputs if len(outputs) > 1 or return_aux else prediction


__all__ = [
	"AtmosphereEncoder",
	"CAWFELatte",
	"TerrainEncoder",
	"TerrainFiLMConditioner",
	"FireFuelEncoder",
	"FireQueryCrossAttentionFusion",
	"ConcatProjectionFusion",
	"FluxEnergyEncoder",
	"MultimodalAlignment",
	"PredictionHead",
	"PatchFirePresenceHead",
	"GradientReversalFunction",
	"gradient_reverse",
	"DomainAdversarialHead",
	"MaskGuidedRegressionAttention",
	"RegressionMixtureOfExperts",
	"ContrastiveProjectionHead",
	"PhysicalStateAuxHead",
	"ShallowDecoder",
	"TemporalAggregator",
	"TemporalAttentionPooling",
	"SupportGate",
	"build_post_fusion_backbone",
	"build_temporal_pooling",
	"build_regression_activation",
	"build_support_gate",
	"TemporalCNNBackbone",
	"ResidualSpatiotemporalBackbone",
	"ResidualSpatiotemporalBlock",
	"MultiScaleContextBackbone",
	"LocalWindowAttentionBlock",
	"LocalWindowAttentionBackbone",
	"CuboidAttentionBlock",
	"CuboidAttentionLiteBackbone",
	"SpectralConv2d",
	"SpectralFourierBlock",
	"SpectralFourierBackbone",
	"SpatiotemporalMambaBlock",
	"SpatiotemporalMambaBackbone",
	"window_partition",
	"window_reverse",
	"cuboid_partition",
	"cuboid_reverse",
	"TemporalSpatialResidualBlock",
	"TemporalConvBlock",
	"WindEncoder",
	"select_channels_by_keywords",
	"tokens_to_grid",
]
