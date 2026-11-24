"""Explicit learning-rate schedules and complete checkpoint restoration."""

import math

import torch


def _schedule_spec(config):
    training = config['training']
    options = training.get('scheduler', {})
    kind = options.get('type', 'CosineAnnealingWarmRestarts')
    eta_min = float(options.get('eta_min', 0.0))
    if kind == 'CosineAnnealingLR':
        return {
            'type': kind, 'T_max': options.get('T_max', training['epochs']),
            'eta_min': eta_min,
        }
    if kind == 'CosineAnnealingWarmRestarts':
        return {
            'type': kind, 'T_0': options['T_0'],
            'T_mult': options.get('T_mult', 1), 'eta_min': eta_min,
        }
    raise ValueError(f'Unknown scheduler type: {kind}')


def build_scheduler(optimizer, config, *, epochs_remaining=None):
    spec = _schedule_spec(config)
    learning_rate = float(config['training']['learning_rate'])
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if not math.isfinite(spec['eta_min']) or not 0 <= spec['eta_min'] <= learning_rate:
        raise ValueError('eta_min must be finite and between zero and learning_rate')
    if epochs_remaining is not None and epochs_remaining <= 0:
        raise ValueError('epochs_remaining must be positive')
    if spec['type'] == 'CosineAnnealingLR':
        horizon = spec['T_max'] if epochs_remaining is None else epochs_remaining
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError('CosineAnnealingLR T_max must be a positive integer')
        # Stepping past T_max would make PyTorch's cosine start increasing again.
        if epochs_remaining is None and horizon < config['training']['epochs']:
            raise ValueError('T_max must cover all training epochs to prevent learning-rate increases')
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=horizon, eta_min=spec['eta_min'],
        )
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=spec['T_0'], T_mult=spec['T_mult'], eta_min=spec['eta_min'],
    )


def restore_training_state(model, optimizer, checkpoint, config, *, reset_scheduler=False):
    """Restore every model tensor and Adam state; reset LR only when requested."""
    saved = checkpoint.get('config', {})
    scheduler_state = checkpoint.get('scheduler_state_dict')
    if not reset_scheduler:
        if saved.get('training'):
            saved_training = saved['training']
            if (
                _schedule_spec(saved) != _schedule_spec(config)
                or saved_training['learning_rate'] != config['training']['learning_rate']
            ):
                raise ValueError('Resume learning-rate strategy differs; use --reset-scheduler explicitly')
        elif scheduler_state:
            saved_kind = (
                'CosineAnnealingWarmRestarts' if 'T_0' in scheduler_state
                else 'CosineAnnealingLR'
            )
            if saved_kind != _schedule_spec(config)['type']:
                raise ValueError('Resume scheduler type differs; use --reset-scheduler explicitly')
        if scheduler_state is None:
            raise ValueError('Checkpoint has no scheduler state; use --reset-scheduler explicitly')

    start_epoch = int(checkpoint.get('epoch', -1)) + 1
    remaining = config['training']['epochs'] - start_epoch
    if reset_scheduler and remaining <= 0:
        raise ValueError('No remaining epochs; increase training.epochs before resetting the scheduler')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if reset_scheduler:
        learning_rate = float(config['training']['learning_rate'])
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
            group['initial_lr'] = learning_rate
        scheduler = build_scheduler(optimizer, config, epochs_remaining=remaining)
    else:
        # Warm-restart construction can modify lr; preserve the checkpoint's
        # next-step rates while restoring the scheduler's exact saved position.
        saved_rates = [group['lr'] for group in optimizer.param_groups]
        scheduler = build_scheduler(optimizer, config)
        scheduler.load_state_dict(scheduler_state)
        for group, rate in zip(optimizer.param_groups, saved_rates):
            group['lr'] = rate
        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            if remaining > scheduler.T_max - scheduler.last_epoch:
                raise ValueError('Remaining epochs exceed cosine horizon; use --reset-scheduler to extend training')
    best_ber = float(checkpoint.get('best_ber', checkpoint.get('ber', float('inf'))))
    if not math.isfinite(best_ber):
        best_ber = float('inf')
    return scheduler, start_epoch, best_ber
