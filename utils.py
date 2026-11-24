import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from metrics import fast_hist, get_metrics

# 类别颜色映射 (BGR格式用于OpenCV)
CLASS_COLORS = {
    0: (0, 255, 0),      # Clear - 绿色
    1: (128, 0, 128),    # Shadow - 紫色
    2: (255, 165, 0),    # Semi-transparent - 橙色
    3: (255, 255, 255),  # Cloud - 白色
    255: (0, 0, 0)       # Ignore - 黑色
}

CLASS_NAMES = {
    0: 'Clear',
    1: 'Cloud Shadow',
    2: 'Semi-transparent',
    3: 'Cloud',
    255: 'Undefined/Missing'
}

def colorize_mask(mask, color_map=CLASS_COLORS):
    """
    将单通道掩膜转换为彩色可视化图像
    
    Args:
        mask: numpy array of shape (H, W) with class labels
        color_map: 字典，映射类别ID到BGR颜色
        
    Returns:
        colored_mask: numpy array of shape (H, W, 3) in BGR format
    """
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in color_map.items():
        colored_mask[mask == class_id] = color
    
    return colored_mask

def visualize_prediction(image, prediction, ground_truth=None, save_path=None):
    """
    可视化预测结果
    
    Args:
        image: 原始图像 (H, W, 3) 归一化后的RGB图像
        prediction: 预测掩膜 (H, W)
        ground_truth: 真实标签 (可选)
        save_path: 保存路径 (可选)
    """
    # 将图像从[0,1]转换到[0,255]用于显示
    # image_display = (image * 255).astype(np.uint8)
    # 确保 image 是 uint8
    # --- 核心修复：更科学的亮度拉伸 ---
    if image.dtype == np.uint8:
        image_display = image.copy()
    else:
        # 针对 Sentinel-2 的物理特性：绝大多数地表反射率在 0.4 以下
        # 我们手动设定一个截断值 0.4，高于 0.4 的全部算作纯白
        image_std = np.clip(image / 0.4, 0, 1) 
        image_display = (image_std * 255.0).astype(np.uint8)
    # 彩色化预测
    pred_colored = colorize_mask(prediction)
    pred_colored = cv2.cvtColor(pred_colored, cv2.COLOR_BGR2RGB)
    
    # 动态判断布局
    num_cols = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))

    # 统一处理 axes (防止只有 2 张图时 axes 不是数组的问题)
    if num_cols == 2:
        ax_list = axes
    else:
        ax_list = axes

        ax_list[0].imshow(image_display)
        ax_list[0].set_title('Original Image')
        ax_list[0].axis('off')
    
    if ground_truth is not None:
        gt_colored = colorize_mask(ground_truth)
        gt_colored = cv2.cvtColor(gt_colored, cv2.COLOR_BGR2RGB)
        ax_list[1].imshow(gt_colored)
        ax_list[1].set_title('Ground Truth')
        ax_list[1].axis('off')
        
        ax_list[2].imshow(pred_colored)
        ax_list[2].set_title('Prediction')
        ax_list[2].axis('off')
    else:
        # 只显示原图和预测        
        ax_list[0].imshow(image_display)
        ax_list[0].set_title('Original Image')
        ax_list[0].axis('off')
        
        ax_list[1].imshow(pred_colored)
        ax_list[1].set_title('Prediction')
        ax_list[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def save_prediction_mask(prediction, save_path):
    """
    保存预测掩膜为图像文件
    
    Args:
        prediction: numpy array (H, W) 类别标签
        save_path: 保存路径
    """
    # 保存为PNG格式，保留类别ID
    cv2.imwrite(save_path, prediction.astype(np.uint8))

def calculate_iou(pred, target, num_classes, ignore_index=255):
    """
    计算IoU指标
    
    Args:
        pred: 预测结果 (N, H, W) 或 (H, W)
        target: 真实标签 (N, H, W) 或 (H, W)
        num_classes: 类别数量
        ignore_index: 忽略的标签值
        
    Returns:
        iou_per_class: 每个类别的IoU
        mean_iou: 平均IoU
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # 展平为一维
    pred = pred.flatten()
    target = target.flatten()
    
    # 忽略特定标签
    valid_mask = (target != ignore_index)
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    iou_per_class = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union == 0:
            iou = float('nan')  # 该类别不存在
        else:
            iou = intersection / union
        
        iou_per_class.append(iou)
    
    # 计算mean IoU (忽略NaN值)
    iou_array = np.array(iou_per_class)
    mean_iou = np.nanmean(iou_array)
    
    return iou_per_class, mean_iou

def create_legend(save_path='legend.png'):
    """
    创建类别颜色图例
    
    Args:
        save_path: 图例保存路径
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    
    legend_elements = []
    for class_id in sorted([k for k in CLASS_NAMES.keys() if k != 255]):
        color = np.array(CLASS_COLORS[class_id][::-1]) / 255.0  # BGR to RGB, normalize
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, fc=color, label=CLASS_NAMES[class_id])
        )
    
    ax.legend(handles=legend_elements, loc='center', fontsize=12, frameon=False)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Legend saved to {save_path}")
    plt.close()

def overlay_prediction(image, prediction, alpha=0.5):
    """
    将预测掩膜叠加到原始图像上
    
    Args:
        image: 原始图像 (H, W, 3) RGB格式，归一化到[0,1]
        prediction: 预测掩膜 (H, W)
        alpha: 叠加透明度
        
    Returns:
        overlayed: 叠加后的图像
    """
    # 转换图像到[0, 255]
    image_display = (image * 255).astype(np.uint8)
    
    # 彩色化预测
    pred_colored = colorize_mask(prediction)
    pred_colored = cv2.cvtColor(pred_colored, cv2.COLOR_BGR2RGB)
    
    # 叠加
    overlayed = cv2.addWeighted(image_display, 1-alpha, pred_colored, alpha, 0)
    
    return overlayed

