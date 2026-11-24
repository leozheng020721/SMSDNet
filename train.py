import os
import yaml
import argparse
import copy
import csv
import random
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
import numpy as np
from metrics import fast_hist, get_metrics

from dataset import (KappaSetDataset, get_transforms, AISDDataset, get_aisd_loaders,
                     get_seo_loaders, seo_supervision_options)
from model_factory import build_model, count_parameters, initialize_from_checkpoint
from losses import build_criterion
from lr_schedule import build_scheduler, restore_training_state
from mixup import BatchMixUp, mixup_objective
from multimodal import unpack_segmentation_batch
from patch_inference import is_crop_mode, sliding_window_logits
#from model import build_model, count_parameters

def use_fp16(device, config=None):
    return (torch.device(device).type == 'cuda'
            and bool((config or {}).get('training', {}).get('mixed_precision', False)))


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, config=None):
    model.train()
    device = torch.device(device)
    amp_enabled = use_fp16(device, config)
    if scaler is None:
        scaler = GradScaler('cuda', enabled=amp_enabled)
    totals = {}
    mixup_options = (config or {}).get('training', {}).get('mixup', {})
    mixer = (
        BatchMixUp(mixup_options.get('alpha', 0.2), mixup_options.get('probability', 0.5))
        if mixup_options.get('enabled', False) else None
    )
    mixed_batches = 0
    lambda_sum = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images, masks, modalities = unpack_segmentation_batch(batch)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        modalities = {name: value.to(device, non_blocking=True)
                      for name, value in modalities.items()}
        valid_a = valid_b = None
        if mixer is not None:
            if modalities:
                mixed = mixer.with_nir(images, masks, **modalities)
                images = mixed['images']
                target_a, target_b, mixing = mixed['target_a'], mixed['target_b'], mixed['lam']
                modalities = {name: mixed[name] for name in ('nir', 'nir_valid')}
                valid_a, valid_b = {'nir': mixed['valid_a']}, {'nir': mixed['valid_b']}
            else:
                images, target_a, target_b, mixing = mixer(images, masks)
            mixed_batches += mixing < 1.0
            lambda_sum += mixing
        
        optimizer.zero_grad()
        
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            outputs = (
                model.forward_with_aux(images, **modalities)
                if criterion.auxiliary_weight > 0 else model(images, **modalities)
            )
        # Loss reductions use FP32 logits outside autocast, including every auxiliary head.
        if mixer is not None:
            loss, parts = mixup_objective(
                criterion, outputs, target_a, target_b, mixing, return_components=True,
                auxiliary_valid_a=valid_a, auxiliary_valid_b=valid_b,
            )
        else:
            loss_options = {'auxiliary_valid': {'nir': modalities['nir_valid']}} if modalities else {}
            loss, parts = criterion(outputs, masks, return_components=True, **loss_options)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        for name, value in parts.items():
            totals[name] = totals.get(name, 0.0) + value.detach().item()
        pbar.set_postfix(loss=loss.item(), main=parts['main'].item(), aux=parts['aux'].item())
        
    statistics = {name: value / len(loader) for name, value in totals.items()}
    if mixer is not None:
        statistics['mixup_fraction'] = mixed_batches / len(loader)
        statistics['mixup_lambda'] = lambda_sum / len(loader)
    return statistics

def validate(model, val_loader, criterion, device, num_classes, config=None):
    model.eval()
    device = torch.device(device)
    amp_enabled = use_fp16(device, config)
    val_loss = 0
    # 1. 在 CPU 上初始化总混淆矩阵（因为最后矩阵很小，2x2 或 4x4，CPU 累加最稳）
    total_hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    native_crops = config is not None and is_crop_mode(config)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images, masks, modalities = unpack_segmentation_batch(batch)
            if native_crops:
                options = config['dataset']
                # Keep full images on CPU; transfer fixed-size tiles only.
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    outputs = sliding_window_logits(
                        model, images, device,
                        tile_size=options.get('input_size', 512),
                        stride=options.get('tile_stride', 384),
                        tile_batch_size=options.get('tile_batch_size', 2),
                        **modalities,
                    )
                total_hist += fast_hist(outputs.argmax(dim=1), masks, num_classes)
                loss = criterion(outputs.to(device), masks.to(device))
            else:
                images = images.to(device)
                masks = masks.to(device)
                modalities = {name: value.to(device) for name, value in modalities.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    outputs = model(images, **modalities)
                loss = criterion(outputs, masks)
                total_hist += fast_hist(outputs.argmax(dim=1), masks, num_classes)
            val_loss += loss.item()
    
    # 3. 循环结束后，基于整个验证集统一计算指标
    metrics = get_metrics(total_hist)
    tp = total_hist[1, 1]
    fn = total_hist[1, :].sum() - tp
    fp = total_hist[:, 1].sum() - tp
    tn = total_hist.sum() - tp - fn - fp
    metrics['shadow_miss'] = 100.0 * fn / (tp + fn) if tp + fn else float('nan')
    metrics['background_false_positive'] = 100.0 * fp / (tn + fp) if tn + fp else float('nan')

    return val_loss / len(val_loader), metrics

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='KappaSet Cloud Shadow Detection Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Training configuration path')
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint to resume from')
    initialization.add_argument('--init-from', type=str, default=None,
                                help='Transfer compatible weights into a new experiment')
    parser.add_argument('--init-auxiliary-heads', action='store_true',
                        help='With --init-from, also transfer matching auxiliary prediction heads')
    parser.add_argument('--reset-scheduler', action='store_true',
                        help='With --resume, keep all model/optimizer states but reset LR and schedule from config')
    args = parser.parse_args()
    if args.reset_scheduler and not args.resume:
        parser.error('--reset-scheduler requires --resume')
    if args.init_auxiliary_heads and not args.init_from:
        parser.error('--init-auxiliary-heads requires --init-from')
    
    # 加载配置
    with open(args.config, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    supervision = (seo_supervision_options(config)
                   if config['dataset'].get('type', '').lower() == 'seo' else None)

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        saved_config = resume_checkpoint.get('config')
        saved_variant = resume_checkpoint.get('model_variant')
        if supervision is not None:
            # Missing supervision metadata denotes the historical Max protocol.
            saved_supervision = seo_supervision_options({
                'dataset': {'root_dir': config['dataset']['root_dir'],
                            **(saved_config or {}).get('dataset', {})},
                'model': (saved_config or {}).get('model', {}),
            })
            if any(saved_supervision[key] != supervision[key]
                   for key in ('mode', 'ignore_uncertainty', 'missing_policy')):
                raise ValueError('Resume dataset.supervision differs; use --init-from for a new supervision protocol')
        if saved_config:
            saved_variant = saved_config['model'].get('variant', 'cbam')
            model_keys = ('encoder', 'multicolor')
            for key in model_keys:
                if saved_config['model'].get(key) != config['model'].get(key):
                    raise ValueError(f"Resume configuration differs from checkpoint: model.{key}")
            for key in ('in_channels', 'num_classes', 'normalization'):
                if saved_config['dataset'].get(key) != config['dataset'].get(key):
                    raise ValueError(f"Resume configuration differs from checkpoint: dataset.{key}")
            saved_nir = {key: value for key, value in saved_config['dataset'].get('nir', {}).items()
                         if key not in ('root_dir', 'manifest')}
            requested_nir = {key: value for key, value in config['dataset'].get('nir', {}).items()
                             if key not in ('root_dir', 'manifest')}
            if saved_nir != requested_nir:
                raise ValueError('Resume dataset.nir differs; use --init-from for new NIR data settings')
            for key, default in (('spatial_mode', 'resize'), ('input_size', 512)):
                if saved_config['dataset'].get(key, default) != config['dataset'].get(key, default):
                    raise ValueError(f"Resume dataset.{key} differs; use --init-from for a new experiment")
            if saved_config.get('loss', {}) != config.get('loss', {}):
                raise ValueError('Resume loss differs; use --init-from for new supervision')
            previous_mixup = saved_config.get('training', {}).get('mixup', {})
            requested_mixup = config.get('training', {}).get('mixup', {})
            if not args.reset_scheduler and previous_mixup != requested_mixup:
                raise ValueError('Resume MixUp settings differ; use a new experiment with --reset-scheduler')
            if saved_config.get('random_seed') != config.get('random_seed'):
                raise ValueError('Resume random_seed differs and may change validation groups')
            if config['dataset'].get('type') == 'seo':
                if saved_config['dataset'].get('val_ratio', 0.2) != config['dataset'].get('val_ratio', 0.2):
                    raise ValueError('Resume val_ratio differs; use --init-from for a new validation split')
            if is_crop_mode(config):
                if saved_config['dataset'].get('tile_stride', 384) != config['dataset'].get('tile_stride', 384):
                    raise ValueError('Resume tile_stride differs; use --init-from for new evaluation settings')
        if saved_variant and saved_variant != config['model'].get('variant', 'cbam'):
            raise ValueError(f"Checkpoint variant {saved_variant!r} does not match requested model variant")
        if args.reset_scheduler and saved_config:
            if saved_config.get('experiment_name') == config.get('experiment_name'):
                raise ValueError('--reset-scheduler requires a new experiment_name to preserve the old run')

    history_path = f"checkpoints/{config['experiment_name']}_history.csv"
    is_new_history = not args.resume or args.reset_scheduler
    if is_new_history and os.path.exists(history_path):
        raise FileExistsError(f"History exists: {history_path}; use ordinary --resume or another experiment_name")
        
    # 设置随机种子
    torch.manual_seed(config['random_seed'])
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['random_seed'])
        
    # 设置设备
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
    
    # 显存优化设置
    torch.backends.cudnn.benchmark = config['cudnn_benchmark']
    
    # 数据集准备
    print("\n=== Loading Datasets ===")
    if supervision is not None:
        print(f"SEO supervision: mode={supervision['mode']}, "
              f"ignore_uncertainty={supervision['ignore_uncertainty']}, "
              f"missing_policy={supervision['missing_policy']}")
        print(f"SEO mask roots: min={supervision['min_root']} | max={supervision['max_root']}")
    '''
    train_ds = KappaSetDataset(
        config['dataset']['root_dir'], 
        split='train', 
        config=config,
        transform=get_transforms(config, 'train')
    )
    val_ds = KappaSetDataset(
        config['dataset']['root_dir'], 
        split='val', 
        config=config,
        transform=get_transforms(config, 'val')
    )
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=True,  # 保持Worker活跃
        prefetch_factor=2  # 预取2个batch
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=True,
        prefetch_factor=2
    )
    '''
    # 直接调用新的工厂函数
    if config['dataset'].get('type') == 'seo':
        train_loader, val_loader = get_seo_loaders(config)
    else:
        train_loader, val_loader = get_aisd_loaders(config)
    if not len(train_loader) or not len(val_loader):
        raise ValueError('Training and validation must each contain at least one batch')
    print(f'Training images: {len(train_loader.dataset)} | Validation images: {len(val_loader.dataset)}')
    for split, loader in (('train', train_loader), ('val', val_loader)):
        report = getattr(loader.dataset, 'supervision_filter_report', None)
        if report is not None:
            print(f'SEO supervision filter [{split}]: {report}')
    if is_crop_mode(config):
        print('Native-resolution random crops; full-resolution sliding-window validation')
        print('Validation loss uses the final output only; metrics use stitched complete images.')

    # 模型构建
    print("\n=== Building Model ===")
    model_config = copy.deepcopy(config)
    if args.resume or args.init_from:
        model_config['model']['encoder_weights'] = None
    model = build_model(model_config).to(device)
    count_parameters(model)
    if args.init_from:
        transfer = initialize_from_checkpoint(
            model, args.init_from, include_auxiliary_heads=args.init_auxiliary_heads,
        )
        print(
            f"Initialized from {args.init_from}: {transfer['loaded']} tensors loaded, "
            f"{transfer['skipped']} skipped, {transfer['new']} new tensors kept"
        )
    
    # 优化器与损失函数
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    
    criterion = build_criterion(config, device)
    print(f"Main loss: {config.get('loss', {}).get('type', 'ce')}")
    print(f"Auxiliary mean CE weight: {criterion.auxiliary_weight}")
    if config.get('training', {}).get('mixup', {}).get('enabled', False):
        print(f"Training MixUp: {config['training']['mixup']} | Validation images are not mixed")
    
    scaler = GradScaler('cuda', enabled=use_fp16(device, config))
    print('Precision: CUDA FP16 mixed precision (AMP + GradScaler)'
          if scaler.is_enabled() else 'Precision: FP32')
    
    # 断点续训：加载checkpoint
    start_epoch = 0
    best_ber = float('inf')

    if resume_checkpoint is not None:
        if os.path.isfile(args.resume):
            print(f"=== Loading checkpoint from {args.resume} ===")
            checkpoint = resume_checkpoint
            
            scheduler, start_epoch, best_ber = restore_training_state(
                model, optimizer, checkpoint, config,
                reset_scheduler=args.reset_scheduler,
            )
            if checkpoint.get('scaler_state_dict'):
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            print(f"✓ Model weights loaded")
            print(f"✓ Optimizer state loaded")
            print(f"✓ Resuming from epoch {start_epoch + 1}")
            if np.isfinite(best_ber):
                print(f"✓ Best BER so far: {best_ber:.2f}%")
            else:
                print("✓ Best BER so far: unavailable in this checkpoint")
        else:
            raise FileNotFoundError(args.resume)
    else:
        scheduler = build_scheduler(optimizer, config)

    print(f"Scheduler: {type(scheduler).__name__} | Next training LR: {optimizer.param_groups[0]['lr']:.8g}")
    if args.reset_scheduler:
        print('All model tensors and optimizer moments restored; old scheduler replaced explicitly.')
    if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
        print(f'Cosine horizon: {scheduler.T_max} epochs | Minimum LR: {scheduler.eta_min:.8g}')

    os.makedirs("checkpoints", exist_ok=True)
    if args.reset_scheduler and np.isfinite(best_ber):
        source_ber = float(resume_checkpoint.get('ber', float('nan')))
        if np.isfinite(source_ber) and np.isclose(source_ber, best_ber, rtol=0, atol=1e-10):
            # Keep the source best available in the new run even if no later
            # epoch improves it; save the explicitly reset schedule with it.
            starting_best = dict(resume_checkpoint)
            starting_best.update(
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                scaler_state_dict=scaler.state_dict(),
                config=copy.deepcopy(config), best_ber=best_ber,
            )
            torch.save(starting_best, f"checkpoints/{config['experiment_name']}_best.pth")
            del starting_best
            print(f'Source best preserved in new experiment: BER {best_ber:.2f}%')

    # CPU checkpoint copies are no longer needed after restore and preservation.
    resume_checkpoint = checkpoint = None
    
    print("\n=== Start Training ===")
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        learning_rate = optimizer.param_groups[0]['lr']
        print(f'Learning rate: {learning_rate:.8g}')
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, config)
        val_loss, metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            config['dataset']['num_classes'],
            config,
        )
        ber = metrics['ber']
        iou = metrics['iou']
        f1 = metrics['f1']
        kappa = metrics['kappa']
        scheduler.step()
        
        print(
            f"Train Total: {train_stats['total']:.4f} | Main: {train_stats['main']:.4f} | "
            f"Aux Mean CE: {train_stats['aux']:.4f} | Val Main Loss: {val_loss:.4f}"
        )
        print(
            f"Metrics - BER: {ber:.2f}% | IoU: {iou:.4f} | "
            f"F1: {f1:.4f} | Kappa: {kappa:.4f}"
        )
        print(
            f"Shadow miss: {metrics['shadow_miss']:.2f}% | "
            f"Background false positive: {metrics['background_false_positive']:.2f}%"
        )
        row = {
            'epoch': epoch + 1, 'learning_rate': learning_rate,
            **{'train_' + key: value for key, value in train_stats.items()},
            'val_main_loss': val_loss, **metrics,
        }
        write_header = not os.path.exists(history_path) or os.path.getsize(history_path) == 0
        with open(history_path, 'a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        # BER 越低越好，只在指标有效且刷新历史最优时保存最佳模型。
        if np.isfinite(ber) and ber < best_ber:
            best_ber = ber
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'config': copy.deepcopy(config),
                'model_variant': config['model'].get('variant', 'cbam'),
                'ber': ber,
                'iou': iou,
                'f1': f1,
                'kappa': kappa,
                'best_ber': best_ber,
            }, f"checkpoints/{config['experiment_name']}_best.pth")
            print(f">>> Best Model Saved! BER: {best_ber:.2f}%")
            
        # 保存最新模型 (用于断点续训和状态检查)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'config': copy.deepcopy(config),
            'model_variant': config['model'].get('variant', 'cbam'),
            'ber': ber,
            'iou': iou,
            'f1': f1,
            'kappa': kappa,
            'best_ber': best_ber,
        }, f"checkpoints/{config['experiment_name']}_last.pth")
        print(">>> Last Model Saved!")
    
    print(f"\n=== Training Complete ===")
    if np.isfinite(best_ber):
        print(f"Best BER: {best_ber:.2f}%")
    else:
        print("Best BER: unavailable (validation set contained no valid labels)")

if __name__ == "__main__":
    main()
