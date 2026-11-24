import torch
import segmentation_models_pytorch as smp

def build_model(config):
    """
    构建SegFormer模型。
    使用 segmentation-models-pytorch 库，该库已支持 SegFormer 架构。
    """
    print(f"Building Model: {config['model']['arch']} with encoder {config['model']['encoder']}")
    
    # SMP库中的SegFormer实现
    model = smp.Segformer(
        encoder_name=config['model']['encoder'],      # e.g., "mit_b3"
        encoder_weights=config['model']['encoder_weights'], # "imagenet"
        in_channels=config['dataset']['in_channels'], # 3
        classes=config['dataset']['num_classes'],     # 4
        activation=None # 输出Logits，供CrossEntropy/FocalLoss使用
    )
    
    return model

def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    return total_params, trainable_params
