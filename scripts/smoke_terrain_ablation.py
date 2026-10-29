#!/usr/bin/env python3
"""CPU smoke checks: two training and two validation batches per terrain mode."""
from __future__ import annotations

import argparse
import torch

from src.config import load_config
from src.models.model_factory import build_model_from_config
from src.training.losses import get_loss_function
from src.training.model_outputs import extract_prediction
from scripts.run_terrain_ablation import MODES, config_path


def batch(seed: int):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 5, 129, 8, 8, generator=generator)
    terrain = torch.rand(2, 4, 8, 8, generator=generator)
    target = torch.rand(2, 4, 8, 8, generator=generator)
    target[:, 2] = (target[:, 2] > 0.7).float()
    return x, terrain, target


def prediction(model, x, terrain):
    return extract_prediction(model(x, terrain=terrain))


def check_path(model, mode, x, terrain):
    model.eval()
    with torch.no_grad():
        if mode == 'none':
            assert model.terrain_encoder is None and model.terrain_film is None
            a = prediction(model, x, terrain)
            b = prediction(model, x, terrain + 0.5)
            assert torch.equal(a, b), 'NONE changed when only terrain changed'
        elif mode == 'early_concat':
            assert model.fire_fuel_encoder.terrain_channels == 4
            assert model.fire_fuel_encoder.indices[-4:] == [129, 130, 131, 132]
            assert model.terrain_encoder is None and model.terrain_film is None
            a = model(x, terrain=terrain, return_features=True)['fire_fuel']
            b = model(x, terrain=terrain + 0.5, return_features=True)['fire_fuel']
            assert not torch.allclose(a, b), 'EARLY_CONCAT did not affect fire encoder'
        elif mode == 'prefusion_film':
            assert model.terrain_film is not None and not model.terrain_film_enabled
            model.terrain_film.shift.weight[0, 0, 0, 0] = 1.0
            a = model(x, terrain=terrain, return_features=True)
            b = model(x, terrain=terrain + 0.5, return_features=True)
            assert not torch.allclose(a['fire_fuel'], b['fire_fuel']), 'PREFUSION_FILM did not alter fire query source'
            assert torch.allclose(a['fused_dynamic'], a['fused']), 'PREFUSION_FILM applied terrain after fusion'
        elif mode == 'terrain_query':
            assert model.terrain_query_projection is not None and model.terrain_film is None
            model.terrain_query_projection.weight[0, 0, 0, 0] = 1.0
            a = model(x, terrain=terrain, return_features=True)
            b = model(x, terrain=terrain + 0.5, return_features=True)
            assert not torch.allclose(a['aligned_fire_fuel'], b['aligned_fire_fuel']), 'QUERY did not alter aligned query'
            assert torch.allclose(a['fused_dynamic'], a['fused']), 'QUERY applied terrain after fusion'
        elif mode == 'late_concat':
            assert model.terrain_late_projection is not None and model.terrain_film is None
            model.terrain_late_projection.weight[0, 64, 0, 0] = 1.0
            a = model(x, terrain=terrain, return_features=True)
            b = model(x, terrain=terrain + 0.5, return_features=True)
            assert torch.allclose(a['fused_dynamic'], b['fused_dynamic']), 'LATE_CONCAT changed pre-fusion path'
            assert not torch.allclose(a['fused'], b['fused']), 'LATE_CONCAT did not alter post-fusion latent'
        else:
            assert model.terrain_film_enabled and model.terrain_film is not None
            assert model.terrain_query_projection is None and model.terrain_late_projection is None
            original = model(x, terrain=terrain, return_features=True)
            expected = model.terrain_film(original['fused_dynamic'], original['terrain_features'])
            assert torch.allclose(original['fused'], expected), 'POSTFUSION_FILM differs from original path'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-data', action='store_true', help='Use two real train and val batches, cropped to 8x8 for a quick CPU check.')
    args = parser.parse_args()
    torch.set_num_threads(1)
    real_train = real_val = None
    if args.real_data:
        from itertools import islice
        from src.data.dataset import create_dataloaders
        from src.training.batch_utils import unpack_batch
        from src.training.input_normalization import apply_input_normalization, build_input_normalizer_for_loader
        config = load_config(config_path('postfusion_film'))
        config['training']['batch_size'] = 2
        config['data_loader']['batch_size'] = 2
        config['data_loader']['num_workers'] = 0
        for split in ('train', 'val'):
            config['data_loader'][split]['num_workers'] = 0
            config['data_loader'][split]['persistent_workers'] = False
        train_loader, val_loader, test_loader = create_dataloaders(config)
        assert test_loader is None
        def collect(loader):
            normalizer = build_input_normalizer_for_loader(loader, torch.device('cpu'), 129, config)
            result = []
            for item in islice(loader, 2):
                x, target, extra = unpack_batch(item)
                x = apply_input_normalization(x, normalizer)
                terrain = extra['terrain']
                assert terrain is not None and torch.isfinite(terrain).all()
                result.append((x[..., :8, :8], terrain[..., :8, :8], target[..., :8, :8]))
            assert len(result) == 2
            return result
        real_train, real_val = collect(train_loader), collect(val_loader)
    for mode in MODES:
        config = load_config(config_path(mode))
        model = build_model_from_config(config, input_channels=129)
        criterion = get_loss_function(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        for index in range(2):
            x, terrain, target = real_train[index] if real_train is not None else batch(100 + index)
            model.train(); optimizer.zero_grad(set_to_none=True)
            output = model(x, terrain=terrain)
            pred = extract_prediction(output)
            assert pred.shape == target.shape == (2, 4, 8, 8)
            result = criterion(output, target)
            loss = result['total_loss'] if isinstance(result, dict) else result
            assert torch.isfinite(loss), f'{mode} train loss nonfinite'
            loss.backward()
            assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for index in range(2):
                x, terrain, target = real_val[index] if real_val is not None else batch(200 + index)
                output = model(x, terrain=terrain)
                pred = extract_prediction(output)
                assert pred.shape == target.shape == (2, 4, 8, 8)
                result = criterion(output, target)
                loss = result['total_loss'] if isinstance(result, dict) else result
                assert torch.isfinite(loss), f'{mode} val loss nonfinite'
        x, terrain, _ = batch(300)
        check_path(model, mode, x, terrain)
        name = 'TERRAIN_QUERY' if mode == 'terrain_query' else f'TERRAIN_{mode.upper()}'
        print(f'{name}: two train batches, two val batches, and path checks passed', flush=True)
    # The explicit reference mode must load the exact legacy GA_Q2 state dict.
    legacy = load_config('configs/final_training/cawfe_latte_GA_Q2_full.yaml')
    explicit = load_config(config_path('postfusion_film'))
    torch.manual_seed(947)
    old_model = build_model_from_config(legacy, input_channels=129).eval()
    torch.manual_seed(947)
    new_model = build_model_from_config(explicit, input_channels=129).eval()
    x, terrain, _ = batch(947)
    with torch.no_grad():
        torch.manual_seed(123)
        old_prediction = prediction(old_model, x, terrain)
        torch.manual_seed(123)
        new_prediction = prediction(new_model, x, terrain)
    assert old_model.state_dict().keys() == new_model.state_dict().keys()
    assert torch.equal(old_prediction, new_prediction), 'Explicit POSTFUSION_FILM differs from GA_Q2'
    print('POSTFUSION_FILM matches legacy GA_Q2 prediction and state-dict structure', flush=True)


if __name__ == '__main__':
    main()
