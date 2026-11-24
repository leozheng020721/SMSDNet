"""Bounded synthetic AMP check; never reads dataset samples or saves weights."""

import copy
import json
import math
from pathlib import Path
import sys
import time
import traceback

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from losses import build_criterion
from model_factory import build_model, initialize_from_checkpoint
from train import train_one_epoch


class CheckedAdamW(torch.optim.AdamW):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actual_updates = 0
        self.successful_gradient_checks = 0

    def step(self, closure=None):
        gradients = [parameter.grad for group in self.param_groups
                     for parameter in group['params'] if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(value).all().item() for value in gradients):
            raise AssertionError('Optimizer received absent or non-finite gradients')
        self.successful_gradient_checks += 1
        result = super().step(closure)
        self.actual_updates += 1
        return result


def all_parameters_finite(model):
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def synthetic_batch(config, batch_size, size):
    rgb = torch.rand(batch_size, 3, size, size)
    normalization = config['dataset']['normalization']
    mean = torch.tensor(normalization['mean']).view(1, 3, 1, 1)
    std = torch.tensor(normalization['std']).view(1, 3, 1, 1)
    mask = torch.randint(0, 2, (batch_size, size, size), dtype=torch.long)
    mask[:, :size // 8, :size // 8] = 255
    nir_valid = torch.rand(batch_size, 1, size, size) > 0.1
    nir = (torch.rand(batch_size, 1, size, size) * 2 - 1).masked_fill(~nir_valid, 0)
    return {'image': (rgb - mean) / std, 'mask': mask,
            'nir': nir, 'nir_valid': nir_valid}


def run(report):
    if not torch.cuda.is_available():
        report.update(status='skipped', reason='CUDA unavailable')
        return
    torch.manual_seed(123)
    device = torch.device('cuda:0')
    report['gpu'] = torch.cuda.get_device_name(device)
    report['free_bytes_before_model'], report['total_bytes'] = torch.cuda.mem_get_info(device)
    if report['free_bytes_before_model'] < 3 * 1024 ** 3:
        report.update(status='skipped', reason='Less than 3 GiB CUDA free before model allocation')
        return
    with (ROOT / 'config_seo_cca_nir_server.yaml').open(encoding='utf-8') as stream:
        config = copy.deepcopy(yaml.safe_load(stream))
    config['model']['encoder_weights'] = None
    config['training']['mixed_precision'] = True
    config['training']['mixup']['probability'] = 1.0
    model = build_model(config)
    report['checkpoint_load'] = initialize_from_checkpoint(
        model, ROOT / 'checkpoints/seo_segformer_native512_aux_cca_nir_mixup_best.pth',
        include_auxiliary_heads=True,
    )
    model.to(device)
    criterion = build_criterion(config, device)
    optimizer = CheckedAdamW(model.parameters(), lr=config['training']['learning_rate'],
                             weight_decay=config['training']['weight_decay'])
    scaler = torch.amp.GradScaler('cuda')
    report.update(training_size=[2, 3, 128, 128], max_training_steps=3,
                  mixup_probability=1.0, initial_scale=scaler.get_scale(), steps=[])
    probe = model.segmentation_head[0].weight
    original_probe = probe.detach().clone()
    for index in range(3):
        scale_before = scaler.get_scale()
        updates_before = optimizer.actual_updates
        stats = train_one_epoch(model, [synthetic_batch(config, 2, 128)], criterion,
                                optimizer, scaler, device, config)
        torch.cuda.synchronize(device)
        finite_losses = all(math.isfinite(value) for value in stats.values())
        finite_parameters = all_parameters_finite(model)
        report['steps'].append({
            'attempt': index + 1, 'scale_before': scale_before,
            'scale_after': scaler.get_scale(), 'statistics': stats,
            'optimizer_updated': optimizer.actual_updates > updates_before,
            'finite_losses': finite_losses, 'finite_parameters': finite_parameters,
        })
        if not finite_losses or not finite_parameters:
            raise AssertionError('Non-finite training loss or model parameters')
        if optimizer.actual_updates:
            break
    report['actual_optimizer_updates'] = optimizer.actual_updates
    report['successful_finite_gradient_checks'] = optimizer.successful_gradient_checks
    report['probe_parameter_changed'] = not torch.equal(original_probe, probe.detach())
    if not optimizer.actual_updates or not report['probe_parameter_changed']:
        raise AssertionError('No actual parameter update in three AMP attempts')
    optimizer.zero_grad(set_to_none=True)
    del original_probe, optimizer, scaler
    torch.cuda.empty_cache()
    report['free_bytes_before_native512_eval'], _ = torch.cuda.mem_get_info(device)
    if report['free_bytes_before_native512_eval'] <= 4 * 1024 ** 3:
        report['native512_eval'] = {'status': 'skipped', 'reason': '4 GiB CUDA free required'}
    else:
        model.eval()
        batch = synthetic_batch(config, 1, 512)
        classifier_dtypes = []
        hook = model.segmentation_head[0].register_forward_hook(
            lambda module, args, output: classifier_dtypes.append(str(output.dtype)))
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                output = model(batch['image'].to(device), nir=batch['nir'].to(device),
                               nir_valid=batch['nir_valid'].to(device))
        finally:
            hook.remove()
        torch.cuda.synchronize(device)
        finite_output = torch.isfinite(output).all().item()
        report['native512_eval'] = {'status': 'passed' if finite_output else 'failed',
                                    'input_size': [1, 3, 512, 512],
                                    'output_size': list(output.shape),
                                    'output_dtype': str(output.dtype),
                                    'classifier_conv_dtypes': classifier_dtypes,
                                    'finite_output': finite_output}
        if not finite_output or classifier_dtypes != ['torch.float16']:
            raise AssertionError('Native512 output was non-finite or classifier conv was not FP16')
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
    report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
    report['status'] = 'passed'


if __name__ == '__main__':
    result = {'torch_version': torch.__version__, 'cuda_version': torch.version.cuda}
    start = time.monotonic()
    try:
        run(result)
    except torch.cuda.OutOfMemoryError as error:
        result.update(status='skipped', reason='CUDA OOM during bounded check', error=str(error),
                      traceback=traceback.format_exc())
    except Exception as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
    result['elapsed_seconds'] = time.monotonic() - start
    output_path = ROOT / 'remote_bundle/validation/fp16_cuda_smoke.json'
    output_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    if result['status'] == 'failed':
        raise SystemExit(1)
