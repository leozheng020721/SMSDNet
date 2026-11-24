import os
import copy
import csv
import yaml
import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
from torch.utils.data import DataLoader, Subset

from dataset import (KappaSetDataset, get_aisd_dataset, get_seo_dataset, get_transforms,
                     get_nir_transforms, load_aligned_nir, resolve_seo_nir_paths,
                     seo_supervision_options)
from model_factory import build_model, resolve_checkpoint_config
from multimodal import unpack_segmentation_batch
from patch_inference import is_crop_mode, sliding_window_logits
from metrics import fast_hist, get_metrics
from utils import save_prediction_mask, visualize_prediction


def predict_logits(model, images, config, device, nir=None, nir_valid=None):
    if (nir is None) != (nir_valid is None):
        raise ValueError('NIR inference requires both nir and nir_valid')
    modalities = {'nir': nir, 'nir_valid': nir_valid} if nir is not None else {}
    if is_crop_mode(config):
        dataset_config = config['dataset']
        return sliding_window_logits(
            model,
            images,
            device,
            tile_size=dataset_config.get('input_size', 512),
            stride=dataset_config.get('tile_stride', 384),
            tile_batch_size=dataset_config.get('tile_batch_size', 2),
            **modalities,
        )
    modalities = {name: value.to(device) for name, value in modalities.items()}
    return model(images.to(device), **modalities)


def image_for_display(image_tensor, config):
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    normalization = config['dataset']['normalization']
    mean = np.asarray(normalization['mean'])
    std = np.asarray(normalization['std'])
    return np.clip((image * std + mean) * 255.0, 0, 255).astype(np.uint8)

def load_checkpoint(model, checkpoint_path, device, checkpoint=None):
    """加载训练好的模型权重"""
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)
    print(f"Loaded checkpoint from {checkpoint_path}")
    metric_names = ('ber', 'iou', 'f1', 'kappa')
    if all(name in checkpoint for name in metric_names):
        print(
            f"Checkpoint Stored Metrics - BER: {checkpoint['ber']:.2f}% | "
            f"IoU: {checkpoint['iou']:.4f} | F1: {checkpoint['f1']:.4f} | "
            f"Kappa: {checkpoint['kappa']:.4f}"
        )
    if 'best_ber' in checkpoint:
        print(f"Checkpoint Best BER: {checkpoint['best_ber']:.2f}%")
    return model

def inference_single_image(model, image_path, config, device, output_dir=None):
    """
    对单张图像进行推理
    
    Args:
        model: 训练好的模型
        image_path: 输入图像路径
        config: 配置字典
        device: 推理设备
        output_dir: 输出目录（可选）
    """
    model.eval()
    
    # 读取并预处理图像
    dataset_type = config['dataset'].get('type', 'kappaset').lower()
    standard_rgb = dataset_type in ('aisd', 'seo')
    image = cv2.imread(
        image_path, cv2.IMREAD_COLOR if standard_rgb else cv2.IMREAD_UNCHANGED
    )
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 归一化
    if standard_rgb:
        transform_image = image
    else:
        div_value = config['dataset']['normalization']['div_value']
        image_normalized = np.clip(image.astype(np.float32) / div_value, 0.0, 1.0)
        transform_image = image_normalized
    
    # 应用验证集的transform
    modalities = {}
    if config['dataset'].get('nir', {}).get('enabled', False):
        nir_path, valid_path = resolve_seo_nir_paths(config, image_path)
        nir, nir_valid = load_aligned_nir(nir_path, valid_path, image.shape[:2], config['dataset']['nir'])
        augmented = get_nir_transforms(config, 'val')(
            image=transform_image, mask=np.zeros(image.shape[:2], dtype=np.uint8),
            nir=nir, nir_valid=nir_valid,
        )
        modalities = {name: augmented[name].unsqueeze(0) for name in ('nir', 'nir_valid')}
    else:
        transform = get_transforms(config, 'val')
        augmented = transform(image=transform_image)
    image_tensor = augmented['image'].unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        # with torch.amp.autocast('cuda', dtype=torch.float16):
        outputs = predict_logits(model, image_tensor, config, device, **modalities)
        prediction = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
    
    # 可视化和保存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 保存预测掩膜
        save_prediction_mask(prediction, os.path.join(output_dir, f"{base_name}_pred.png"))
        
        # 保存可视化结果
        visualize_prediction(
            image_for_display(image_tensor[0], config),
            prediction, 
            save_path=os.path.join(output_dir, f"{base_name}_vis.png")
        )
        
        print(f"Results saved to {output_dir}")
    
    return prediction

def build_inference_dataset(config, split='val', count=None):
    """Use the training split with validation transforms, without reading pixels."""
    dataset_type = config['dataset'].get('type', 'kappaset').lower()
    if dataset_type == 'aisd':
        dataset = get_aisd_dataset(config, split, transform=get_transforms(config, 'val'))
    elif dataset_type == 'seo':
        transform = (get_nir_transforms(config, 'val')
                     if config['dataset'].get('nir', {}).get('enabled', False)
                     else get_transforms(config, 'val'))
        dataset = get_seo_dataset(config, split, transform=transform)
    else:
        dataset = KappaSetDataset(
            config['dataset']['root_dir'], split=split,
            config=config, transform=get_transforms(config, 'val'),
        )
    if count is not None:
        if count < 1:
            raise ValueError("count must be positive")
        sample_count = min(count, len(dataset))
        dataset = Subset(dataset, range(sample_count))
        print(f"Limiting inference to first {sample_count} images")
    return dataset


def export_dataset_manifest(dataset, output_dir, split='val', num_classes=2,
                            group_by_parent=False):
    """List source files and planned outputs without loading image arrays."""
    output_root = Path(output_dir).resolve()
    records = []
    output_names = set()
    for index in range(len(dataset)):
        source = dataset
        dataset_index = index
        while isinstance(source, Subset):
            dataset_index = int(source.indices[dataset_index])
            source = source.dataset
        image_paths = getattr(source, 'img_paths', getattr(source, 'nc_files', None))
        mask_paths = getattr(source, 'mask_paths', None)
        image_path = Path(image_paths[dataset_index]).resolve() if image_paths is not None else None
        mask_path = Path(mask_paths[dataset_index]).resolve() if mask_paths is not None else None
        group = image_path.parent.name if group_by_parent and image_path is not None else ''
        stem = image_path.stem if image_path is not None else f'sample_{dataset_index:04d}'
        target_dir = output_root / group
        prediction_path = target_dir / f'{stem}_pred.png'
        key = str(prediction_path).casefold()
        if key in output_names:
            raise ValueError(f"Duplicate prediction output: {prediction_path}")
        output_names.add(key)
        records.append({
            'dataset_index': dataset_index,
            'split': split,
            'group': group,
            'image_path': str(image_path) if image_path is not None else '',
            'mask_path': str(mask_path) if mask_path is not None else '',
            'prediction_path': str(prediction_path),
            'binary_path': str(target_dir / f'{stem}_shadow.png') if num_classes == 2 else '',
            'visualization_path': str(target_dir / f'{stem}_vis.png'),
        })
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f'{split}_manifest.csv'
    fields = ('dataset_index', 'split', 'group', 'image_path', 'mask_path',
              'prediction_path', 'binary_path', 'visualization_path')
    with manifest_path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Exported {len(records)} {split} images to {manifest_path}")
    print('Manifest output paths are planned; files are created during inference.')
    return records, manifest_path


def inference_dataset(model, config, device, split='val', output_dir='predictions', count=None):
    """
    对整个数据集进行批量推理
    
    Args:
        model: 训练好的模型
        config: 配置字典
        device: 推理设备
        split: 数据集划分 ('train', 'val', 'test')
        output_dir: 输出目录
        count: 限制处理的图片数量（用于快速测试）
    """
    model.eval()
    
    dataset = build_inference_dataset(config, split, count)
    records, _ = export_dataset_manifest(
        dataset, output_dir, split, config['dataset']['num_classes'],
        group_by_parent=config['dataset'].get('type', '').lower() == 'seo',
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=1,  # 强制batch_size=1以便逐张可视化
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running inference on {len(dataset)} images...")
    total_hist = np.zeros(
        (config['dataset']['num_classes'], config['dataset']['num_classes']),
        dtype=np.int64,
    )
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="Inference")):
            images, masks, modalities = unpack_segmentation_batch(batch)
            # with torch.amp.autocast('cuda', dtype=torch.float16):
            outputs = predict_logits(model, images, config, device, **modalities)
            prediction_tensors = torch.argmax(outputs, dim=1)
            total_hist += fast_hist(
                prediction_tensors.cpu(),
                masks,
                config['dataset']['num_classes'],
            )
            predictions = prediction_tensors.cpu().numpy()
            
            # 获取原始图像用于可视化 (反归一化)
            # images: (B, 3, H, W) -> (H, W, 3)
            img_vis = image_for_display(images[0], config)
            
            # 获取真值
            gt_mask = masks[0].cpu().numpy()
            
            # 获取预测
            pred_mask = predictions[0]
            
            record = records[idx]
            Path(record['prediction_path']).parent.mkdir(parents=True, exist_ok=True)
            
            # 1. 保存可视化对比图 (Input | GT | Pred)
            visualize_prediction(
                img_vis, 
                pred_mask, 
                ground_truth=gt_mask,
                save_path=record['visualization_path'],
            )
            
            # 2. 保存原始预测Mask (用于后续指标计算)
            if not cv2.imwrite(record['prediction_path'], pred_mask.astype(np.uint8)):
                raise OSError(f"Unable to save prediction: {record['prediction_path']}")
            if record['binary_path']:
                shadow = (pred_mask == 1).astype(np.uint8) * 255
                if not cv2.imwrite(record['binary_path'], shadow):
                    raise OSError(f"Unable to save shadow image: {record['binary_path']}")
            
    metrics = get_metrics(total_hist)
    print(
        f"Inference Metrics - BER: {metrics['ber']:.2f}% | "
        f"IoU: {metrics['iou']:.4f} | F1: {metrics['f1']:.4f} | "
        f"Kappa: {metrics['kappa']:.4f}"
    )
    print(f"All predictions saved to {output_dir}")
    return metrics

def main():
    parser = argparse.ArgumentParser(description='KappaSet Cloud Shadow Detection Inference')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重路径')
    parser.add_argument('--mode', type=str, choices=['single', 'dataset'], default='dataset',
                        help='推理模式: single (单张图像) 或 dataset (批量)')
    parser.add_argument('--image', type=str, help='单张图像路径 (mode=single时需要)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='数据集划分 (mode=dataset时使用)')
    parser.add_argument('--output', type=str, default='predictions', help='输出目录')
    parser.add_argument('--count', type=int, default=None, help='仅处理前N张图像')
    parser.add_argument('--export-only', action='store_true',
                        help='Export the selected split manifest without running the model')
    
    args = parser.parse_args()
    if args.export_only and args.mode != 'dataset':
        parser.error('--export-only requires --mode dataset')
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = resolve_checkpoint_config(config, checkpoint)
    if config['dataset'].get('type', '').lower() == 'seo':
        supervision = seo_supervision_options(config)
        print(f"SEO evaluation supervision: mode={supervision['mode']}, "
              f"ignore_uncertainty={supervision['ignore_uncertainty']}")

    if config['dataset'].get('type', '').lower() == 'seo':
        print(f"SEO split: seed={config['random_seed']}, val_ratio={config['dataset'].get('val_ratio', 0.2)}")
    if args.export_only:
        dataset = build_inference_dataset(config, args.split, args.count)
        export_dataset_manifest(
            dataset, args.output, args.split, config['dataset']['num_classes'],
            group_by_parent=config['dataset'].get('type', '').lower() == 'seo',
        )
        return
    
    # 设置设备
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # 构建模型
    print("Building model...")
    inference_config = copy.deepcopy(config)
    inference_config['model']['encoder_weights'] = None
    model = build_model(inference_config).to(device)
    model = load_checkpoint(model, args.checkpoint, device, checkpoint=checkpoint)
    
    # 执行推理
    if args.mode == 'single':
        if not args.image:
            raise ValueError("mode=single 时必须指定 --image 参数")
        inference_single_image(model, args.image, config, device, args.output)
        print("Single-image mode has no ground-truth mask; metrics were not calculated.")
    else:
        inference_dataset(model, config, device, args.split, args.output, args.count)

if __name__ == "__main__":
    main()
