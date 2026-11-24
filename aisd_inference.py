import torch
import torch.onnx
import onnxruntime
import copy
import numpy as np
import cv2
import os
import yaml
import glob
import argparse
import hashlib
import json
from tqdm import tqdm

from metrics import fast_hist, get_metrics
from model_factory import build_model, resolve_checkpoint_config
from patch_inference import is_crop_mode, sliding_window_logits


class ONNXTileModel(torch.nn.Module):
    """Expose the ONNX session to the shared CPU-accumulated sliding window."""

    def __init__(self, session):
        super().__init__()
        self.session = session
        self.input_name = session.get_inputs()[0].name

    def forward(self, images):
        logits = self.session.run(
            None, {self.input_name: images.detach().cpu().numpy()}
        )[0]
        return torch.from_numpy(logits)

# ==========================================
# 1. 辅助函数：加载配置与模型
# ==========================================
def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_segformer_model(config):
    inference_config = copy.deepcopy(config)
    inference_config['model']['encoder_weights'] = None
    return build_model(inference_config)

# ==========================================
# 2. 导出 ONNX (仅需执行一次)
# ==========================================
def export_onnx(pth_path, onnx_path, config, input_size=None, force=False):
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"找不到模型权重: {pth_path}")
    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
    config = resolve_checkpoint_config(config, checkpoint)
    input_size = input_size or config['dataset'].get('input_size', 512)
    config['dataset']['input_size'] = input_size
    model = get_segformer_model(config)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)
    model.eval()

    # Cache identity includes architecture and preprocessing, not just file dates.
    source_stat = os.stat(pth_path)
    signature_data = {
        'checkpoint': os.path.abspath(pth_path),
        'size': source_stat.st_size,
        'mtime_ns': source_stat.st_mtime_ns,
        'model': config['model'],
        'normalization': config['dataset']['normalization'],
        'input_size': input_size,
        'spatial_mode': config['dataset'].get('spatial_mode', 'resize'),
        'in_channels': config['dataset']['in_channels'],
        'num_classes': config['dataset']['num_classes'],
        'export_version': 1,
    }
    signature = hashlib.sha256(
        json.dumps(signature_data, sort_keys=True).encode('utf-8')
    ).hexdigest()
    metadata_path = onnx_path + '.metadata.json'
    cached_signature = None
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, encoding='utf-8') as metadata_file:
                cached_signature = json.load(metadata_file).get('signature')
        except (OSError, ValueError):
            pass
    if (
        not force
        and os.path.exists(onnx_path)
        and cached_signature == signature
        and os.path.getmtime(onnx_path) >= os.path.getmtime(pth_path)
    ):
        print(f"[INFO] ONNX 已是最新版本: {onnx_path}")
        return config

    print(f"[INFO] 正在导出 ONNX 模型...")
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    dummy_input = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=17,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                      dynamo=False)
    with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        json.dump({'signature': signature, 'source': signature_data}, metadata_file, indent=2)
    print(f"[INFO] ONNX 导出成功: {onnx_path}")
    return config

# ==========================================
# 3. 核心：批量推理函数
# ==========================================
def batch_infer(
    onnx_path,
    img_dir,
    output_root,
    config,
    num_to_infer=10,
    mask_dir=None,
):
    # 初始化 ONNX Session (只加载一次)
    available_providers = onnxruntime.get_available_providers()
    providers = [name for name in ('CUDAExecutionProvider', 'CPUExecutionProvider')
                 if name in available_providers]
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    print(f"[INFO] ONNX providers: {session.get_providers()}")
    
    # 搜索图片并排序（确保“按顺序”）
    extensions = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
    img_list = []
    for ext in extensions:
        img_list.extend(glob.glob(os.path.join(img_dir, ext)))
    img_list.sort() # 按文件名排序
    
    # 截取指定数量
    if num_to_infer is not None:
        if num_to_infer < 1:
            raise ValueError('num_to_infer must be positive or None')
        img_list = img_list[:num_to_infer]
    if not img_list:
        print(f"[ERROR] 在文件夹 {img_dir} 中未找到图片")
        return

    print(f"[INFO] 开始批量推理，目标数量: {len(img_list)}")
    os.makedirs(output_root, exist_ok=True)

    mask_lookup = {}
    if mask_dir:
        for ext in extensions:
            for mask_path in glob.glob(os.path.join(mask_dir, ext)):
                stem = os.path.splitext(os.path.basename(mask_path))[0].lower()
                mask_lookup[stem] = mask_path

    num_classes = config['dataset']['num_classes']
    total_hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    evaluated_samples = 0

    # 预处理参数
    target_size = (config['dataset'].get('input_size', 512), config['dataset'].get('input_size', 512))
    mean = np.array(config['dataset']['normalization']['mean'], dtype=np.float32)
    std = np.array(config['dataset']['normalization']['std'], dtype=np.float32)
    input_name = session.get_inputs()[0].name
    crop_mode = is_crop_mode(config)
    tile_model = ONNXTileModel(session).eval() if crop_mode else None

    for img_path in tqdm(img_list, desc="Inference Progress"):
        # 1. 读取与前处理
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f'Unable to read image: {img_path}')
        h_orig, w_orig = img.shape[:2]
        inference_image = img if crop_mode else cv2.resize(img, target_size)
        img_rgb = cv2.cvtColor(inference_image, cv2.COLOR_BGR2RGB)
        img_norm = (img_rgb.astype(np.float32) / 255.0 - mean) / std
        img_tensor = img_norm.transpose(2, 0, 1).astype(np.float32)[np.newaxis, ...]

        # 2. 推理
        if crop_mode:
            pred_logits = sliding_window_logits(
                tile_model,
                torch.from_numpy(img_tensor),
                torch.device('cpu'),
                tile_size=config['dataset'].get('input_size', 512),
                stride=config['dataset'].get('tile_stride', 384),
                tile_batch_size=config['dataset'].get('tile_batch_size', 2),
            ).numpy()
        else:
            pred_logits = session.run(None, {input_name: img_tensor})[0]
        pred_mask = np.argmax(pred_logits, axis=1)[0].astype(np.uint8)

        # 3. 缩放回原图大小
        if not crop_mode:
            pred_mask = cv2.resize(pred_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        if mask_dir:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            image_stem = base_name.lower()
            mask_path = mask_lookup.get(image_stem)
            if (
                mask_path is None
                and config['dataset'].get('type', 'aisd').lower() == 'seo'
                and image_stem.endswith('_rgb')
            ):
                mask_path = mask_lookup.get(image_stem[:-4] + '_shadow')
            if mask_path is None:
                print(f"[WARNING] 未找到同名真值掩膜，跳过指标: {base_name}")
            else:
                ground_truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if ground_truth is None:
                    print(f"[WARNING] 无法读取真值掩膜，跳过指标: {mask_path}")
                else:
                    if ground_truth.shape != pred_mask.shape:
                        if crop_mode:
                            raise ValueError(
                                f'Image/mask size mismatch: {img_path} and {mask_path}'
                            )
                        ground_truth = cv2.resize(
                            ground_truth,
                            (w_orig, h_orig),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    if config['dataset'].get('type', 'aisd').lower() == 'seo':
                        ground_truth = (ground_truth < 127).astype(np.int64)
                    else:
                        threshold = 0 if ground_truth.max() <= 1 else 127
                        ground_truth = (ground_truth > threshold).astype(np.int64)
                    total_hist += fast_hist(pred_mask, ground_truth, num_classes)
                    evaluated_samples += 1

        # 4. 可视化并保存
        base_name = os.path.splitext(os.path.basename(img_path))[0]
       
        # --- 新增：保存原始黑白掩膜 (0=背景, 255=阴影) ---
        mask_path = os.path.join(output_root, f"{base_name}_mask.png")
        cv2.imwrite(mask_path, pred_mask * 255) # 乘以 255 是为了让阴影类(1)在图片中显示为白色
        
        # 效果展示：半透明覆盖 (Overlay)
        overlay = img.copy()
        color_mask = np.zeros_like(img)
        color_mask[pred_mask == 1] = [0, 0, 255] # 红色标记
        cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0, overlay)
        
        # 效果展示：边缘勾勒 (Contours)
        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2) # 原图勾绿边
        
        # 保存：每张图存成一个拼接大图 (原图勾边 | 半透明掩膜)
        combined = np.hstack([img, overlay])
        cv2.imwrite(os.path.join(output_root, f"{base_name}_result.png"), combined)

    metrics = None
    if evaluated_samples > 0:
        metrics = get_metrics(total_hist)
        print(
            f"[METRICS] BER: {metrics['ber']:.2f}% | "
            f"IoU: {metrics['iou']:.4f} | F1: {metrics['f1']:.4f} | "
            f"Kappa: {metrics['kappa']:.4f}"
        )
        print(f"[METRICS] Evaluated samples: {evaluated_samples}")
    elif mask_dir:
        print("[WARNING] 未找到可用真值，未计算 BER、IoU、F1 和 Kappa。")
    else:
        print("[INFO] 未提供真值目录，跳过 BER、IoU、F1 和 Kappa 计算。")

    print(f"[SUCCESS] 所有推理结果已保存至: {output_root}")
    return metrics

# ==========================================
# 4. 主程序入口
# ==========================================
def main():
    # --- 你需要修改这里的路径 ---
    MODEL_PTH = r"aisd\model+attentioncb\aisd_segformer_v2_best.pth"
    ONNX_PATH = r"aisd\model+attentioncb\aisd_segformer_v2_best.onnx"
    INPUT_DIR = r"E:\MyPythonProject\MAMNet\datasets\Test51\shadow" # 图片文件夹
    MASK_DIR  = r"E:\MyPythonProject\MAMNet\datasets\Test51\mask"   # 真值掩膜文件夹
    SAVE_DIR  = r"cbam_results_best"              # 结果保存路径
    COUNT     = 51                           # 你想推理的个数
    # --------------------------

    parser = argparse.ArgumentParser(description='Export and evaluate AISD SegFormer variants')
    parser.add_argument('--config', default='config.yaml', help='Configuration path')
    parser.add_argument('--checkpoint', default=MODEL_PTH, help='Model checkpoint path')
    parser.add_argument('--onnx', default=None, help='ONNX cache path (default: beside checkpoint)')
    parser.add_argument('--images', default=INPUT_DIR, help='Input image directory')
    parser.add_argument('--masks', default=MASK_DIR, help='Ground-truth mask directory')
    parser.add_argument('--output', default=SAVE_DIR, help='Prediction output directory')
    parser.add_argument('--count', default=COUNT, type=int, help='Maximum images to evaluate')
    parser.add_argument('--force-export', action='store_true', help='Rebuild the ONNX cache')
    args = parser.parse_args()
    cfg = load_config(args.config)
    onnx_path = args.onnx or os.path.splitext(args.checkpoint)[0] + '.onnx'

    # 导出并执行
    cfg = export_onnx(args.checkpoint, onnx_path, cfg, force=args.force_export)
    batch_infer(
        onnx_path,
        args.images,
        args.output,
        cfg,
        num_to_infer=args.count,
        mask_dir=args.masks,
    )


if __name__ == '__main__':
    main()
