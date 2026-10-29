"""Factory for building wildfire forecasting models from config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.models.architecture_registry import (
	REMOVED_ARCHITECTURE_MESSAGE,
	REMOVED_ARCHITECTURES,
	get_architecture_spec,
	resolve_model_architecture,
)
from src.models.cawfe_latte import CAWFELatte
from src.models.convlstm_unet import ConvLSTMUNet, build_convlstm_unet_from_config
from src.models.earthformer_lite import EarthformerLite
from src.models.st_mamba_lite import STMamba
from src.models.weatherformer_lite import WeatherFormerLite



def _resolve_optional_path(config: Mapping[str, Any], value: Any) -> Path | None:
	if value in (None, "", "null"):
		return None
	path = Path(str(value)).expanduser()
	if path.is_absolute():
		return path.resolve()
	config_path = config.get("config_path")
	if config_path not in (None, "", "null"):
		return (Path(str(config_path)).expanduser().resolve().parent / path).resolve()
	return path.resolve()


def _load_channel_names(config: Mapping[str, Any]) -> dict[int, str]:
	candidates: list[Path] = []
	data_config = config.get("data")
	if isinstance(data_config, Mapping):
		path = _resolve_optional_path(config, data_config.get("channel_manifest_path"))
		if path is not None:
			candidates.append(path)
	for section_name in ("channels", "channel_manifest", "metadata"):
		section = config.get(section_name)
		if isinstance(section, Mapping):
			for key in ("manifest", "path", "channel_manifest"):
				path = _resolve_optional_path(config, section.get(key))
				if path is not None:
					candidates.append(path)
	for path in candidates:
		if not path.exists():
			continue
		payload = json.loads(path.read_text(encoding="utf-8"))
		items = payload.get("channels", payload.get("input_channels", payload)) if isinstance(payload, Mapping) else payload
		names: dict[int, str] = {}
		if isinstance(items, list):
			for index, item in enumerate(items):
				if isinstance(item, Mapping):
					channel_index = int(item.get("index", item.get("channel", index)))
					names[channel_index] = str(item.get("name", item.get("label", f"ch{channel_index:03d}")))
				else:
					names[index] = str(item)
		elif isinstance(items, Mapping):
			for key, value in items.items():
				try:
					channel_index = int(str(key).replace("ch", ""))
				except ValueError:
					continue
				names[channel_index] = str(value.get("name", value.get("label")) if isinstance(value, Mapping) else value)
		return names
	return {}


def build_model_from_config(config: Mapping[str, Any], input_channels: int):
	"""Build the configured architecture."""

	model_config = config.get("model", config)
	if not isinstance(model_config, Mapping):
		model_config = {}
	architecture = resolve_model_architecture(config)
	if architecture in REMOVED_ARCHITECTURES:
		raise ValueError(REMOVED_ARCHITECTURE_MESSAGE)
	_ = get_architecture_spec(architecture)
	if architecture == "convlstm_unet":
		return build_convlstm_unet_from_config(config, input_channels=input_channels)
	if architecture == "earthformer_lite":
		section = config.get("earthformer_lite", {})
		if not isinstance(section, Mapping):
			section = {}
		attention_pattern = str(section.get("attention_pattern", "axial")).lower()
		if attention_pattern != "axial":
			raise ValueError(
				"earthformer_lite currently supports only attention_pattern='axial'. "
				f"Got {attention_pattern!r}."
			)
		if bool(section.get("use_shifted_windows", False)):
			raise ValueError("earthformer_lite does not implement shifted windows in this simplified version.")
		return EarthformerLite(
			input_channels=int(input_channels),
			output_channels=int(model_config.get("output_channels", 4)),
			input_sequence_length=int(section.get("input_sequence_length", config.get("input_sequence_length", 1))),
			patch_size=int(section.get("patch_size", config.get("patch_size", 64))),
			embed_dim=int(section.get("embed_dim", 64)),
			depths=section.get("depths", [2, 2]),
			num_heads=section.get("num_heads", [4, 8]),
			mlp_ratio=float(section.get("mlp_ratio", 4.0)),
			dropout=float(section.get("dropout", 0.0)),
			attention_dropout=float(section.get("attention_dropout", 0.0)),
			drop_path=float(section.get("drop_path", 0.1)),
			use_global_vectors=bool(section.get("use_global_vectors", True)),
			num_global_vectors=int(section.get("num_global_vectors", 8)),
			use_time_pos_embed=bool(section.get("use_time_pos_embed", True)),
			use_space_pos_embed=bool(section.get("use_space_pos_embed", True)),
			gradient_checkpointing=bool(section.get("gradient_checkpointing", False)),
			temporal_readout=str(section.get("temporal_readout", "attention_pool")),
			downsample_stages=int(section.get("downsample_stages", 2)),
			patch_merge_factor=int(section.get("patch_merge_factor", 2)),
			required_patch_divisibility=int(section.get("required_patch_divisibility", 16)),
		)
	if architecture == "st_mamba_lite":
		section = config.get("st_mamba_lite", {})
		if not isinstance(section, Mapping):
			section = {}
		return STMamba(
			input_channels=int(input_channels),
			output_channels=int(model_config.get("output_channels", 4)),
			input_sequence_length=int(section.get("input_sequence_length", config.get("input_sequence_length", 1))),
			patch_size=int(section.get("patch_size", config.get("patch_size", 64))),
			embed_dim=int(section.get("embed_dim", 64)),
			encoder_channels=section.get("encoder_channels", [64, 128]),
			decoder_channels=section.get("decoder_channels", [128, 64]),
			depths=section.get("depths", [2, 2]),
			mamba_backend=str(section.get("mamba_backend", "auto")),
			d_state=int(section.get("d_state", 16)),
			d_conv=int(section.get("d_conv", 4)),
			expand=int(section.get("expand", 2)),
			scan_mode=str(section.get("scan_mode", "route_pair")),
			scan_routes=section.get("scan_routes", ["HVT", "TVH"]),
			bidirectional_scan=bool(section.get("bidirectional_scan", True)),
			use_depthwise_conv3d=bool(section.get("use_depthwise_conv3d", True)),
			temporal_readout=str(section.get("temporal_readout", "attention_pool")),
			dropout=float(section.get("dropout", 0.0)),
			drop_path=float(section.get("drop_path", 0.1)),
			mlp_ratio=float(section.get("mlp_ratio", 4.0)),
			gradient_checkpointing=bool(section.get("gradient_checkpointing", False)),
			depthwise_conv3d_kernel_size=section.get("depthwise_conv3d_kernel_size", [3, 3, 3]),
			use_st_mixer=bool(section.get("use_st_mixer", True)),
			st_mixer_sequence_order=str(section.get("st_mixer_sequence_order", "THW")),
			use_unet_decoder=bool(section.get("use_unet_decoder", True)),
			use_skip_connections=bool(section.get("use_skip_connections", True)),
			upsample_mode=str(section.get("upsample_mode", "bilinear")),
			use_adaln_conditioning=bool(section.get("use_adaln_conditioning", False)),
			use_fire_static_embedding=bool(section.get("use_fire_static_embedding", False)),
			required_patch_divisibility=int(section.get("required_patch_divisibility", 16)),
		)
	if architecture == "weatherformer_lite":
		section = config.get("weatherformer_lite", {})
		if not isinstance(section, Mapping):
			section = {}
		attention_type = str(section.get("attention_type", "factorized")).lower()
		if attention_type != "factorized":
			raise ValueError(
				"weatherformer_lite currently supports only attention_type='factorized'. "
				f"Got {attention_type!r}."
			)
		if not bool(section.get("temporal_attention", True)):
			raise ValueError("weatherformer_lite currently requires temporal_attention=true.")
		spatial_attention = str(section.get("spatial_attention", "window")).lower()
		if spatial_attention != "window":
			raise ValueError(
				"weatherformer_lite currently supports only spatial_attention='window'. "
				f"Got {spatial_attention!r}."
			)
		return WeatherFormerLite(
			input_channels=int(input_channels),
			output_channels=int(model_config.get("output_channels", 4)),
			input_sequence_length=int(section.get("input_sequence_length", config.get("input_sequence_length", 1))),
			patch_size=int(section.get("patch_size", config.get("patch_size", 64))),
			embed_dim=int(section.get("embed_dim", 64)),
			encoder_channels=section.get("encoder_channels", [64, 128]),
			decoder_channels=section.get("decoder_channels", [128, 64]),
			depths=section.get("depths", [2, 2]),
			num_heads=section.get("num_heads", [4, 8]),
			mlp_ratio=float(section.get("mlp_ratio", 4.0)),
			use_channel_scaler=bool(section.get("use_channel_scaler", True)),
			use_feature_gate=bool(section.get("use_feature_gate", True)),
			scaler_init=float(section.get("scaler_init", 1.0)),
			use_time_pos_embed=bool(section.get("use_time_pos_embed", True)),
			use_2d_space_pos_embed=bool(section.get("use_2d_space_pos_embed", True)),
			use_fourier_space_encoding=bool(section.get("use_fourier_space_encoding", True)),
			window_size=int(section.get("window_size", 8)),
			shifted_window=bool(section.get("shifted_window", True)),
			use_global_tokens=bool(section.get("use_global_tokens", True)),
			num_global_tokens=int(section.get("num_global_tokens", 4)),
			temporal_readout=str(section.get("temporal_readout", "attention_pool")),
			dropout=float(section.get("dropout", 0.0)),
			attention_dropout=float(section.get("attention_dropout", 0.0)),
			drop_path=float(section.get("drop_path", 0.1)),
			gradient_checkpointing=bool(section.get("gradient_checkpointing", False)),
			use_unet_decoder=bool(section.get("use_unet_decoder", True)),
			use_skip_connections=bool(section.get("use_skip_connections", True)),
			upsample_mode=str(section.get("upsample_mode", "bilinear")),
			downsample_stages=int(section.get("downsample_stages", 2)),
			patch_merge_factor=int(section.get("patch_merge_factor", 2)),
			required_patch_divisibility=int(section.get("required_patch_divisibility", 16)),
		)
	if architecture == "cawfe_latte":
		section = config.get("cawfe_latte", {})
		if not isinstance(section, Mapping):
			section = {}
		return CAWFELatte(
			input_channels=int(input_channels),
			input_sequence_length=int(section.get("input_sequence_length", config.get("input_sequence_length", 1))),
			output_channels=int(section.get("output_channels", model_config.get("output_channels", 4))),
			output_dim=int(section.get("output_dim", 64)),
			version=str(section.get("version", "v1_end_to_end")),
			atmosphere=section.get("atmosphere", {}),
			wind=section.get("wind", {}),
			fire_fuel=section.get("fire_fuel", {}),
			flux_energy=section.get("flux_energy", {}),
			fusion=section.get("fusion", {}),
			alignment=section.get("alignment", {}),
			backbone=section.get("backbone", {}),
			temporal_aggregation=section.get("temporal_aggregation", {}),
			post_fusion_backbone=section.get("post_fusion_backbone"),
			temporal_pooling=section.get("temporal_pooling"),
			regression=section.get("regression"),
			support_gate=section.get("support_gate"),
			ablation=section.get("ablation"),
			decoder=section.get("decoder", {}),
			heads=section.get("heads", {}),
			auxiliary=section.get("auxiliary", {}),
			patch_fire_head=section.get("patch_fire_head", {}),
			domain_adversarial=section.get("domain_adversarial", {}),
			fire_mmd=section.get("fire_mmd", {}),
			mask_guided_regression=section.get("mask_guided_regression", {}),
			regression_moe=section.get("regression_moe", {}),
			supervised_contrastive=section.get("supervised_contrastive", {}),
			physical_state_aux=section.get("physical_state_aux", {}),
			use_terrain_conditioning=bool(section.get("use_terrain_conditioning", False)),
			terrain_encoder=section.get("terrain_encoder", {}),
			terrain_film=section.get("terrain_film", {}),
			terrain_integration=model_config.get("terrain_integration"),
			channel_names=_load_channel_names(config),
			debug_prediction_head=bool(section.get("debug_prediction_head", False)),
		)
	if architecture in {"cawfe_latte_v1_1", "cawfe_latte_v1_2", "cawfe_latte_v1_3"}:
		raise ValueError(f"Unsupported architecture: {architecture}. Experimental CAWFE-Latte v1.1/v1.2/v1.3 variants were archived. Use cawfe_latte or restore from archive/failed_cawfe_latte_variants.")
	raise ValueError(f"Unsupported model architecture: {architecture!r}.")
