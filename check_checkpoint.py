import torch

ckpt_path = r'D:\segformer\checkpoints\KappaSet_SegFormer_MiT-B3_RGB_4Class_Optimized_best.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')

print("=" * 60)
print("Checkpoint Information")
print("=" * 60)
print(f"Keys in checkpoint: {list(ckpt.keys())}")
print()

if 'epoch' in ckpt:
    print(f"Epoch: {ckpt['epoch'] + 1}")
    
metric_names = ('ber', 'iou', 'f1', 'kappa')
if all(name in ckpt for name in metric_names):
    print(
        f"Metrics - BER: {ckpt['ber']:.2f}% | IoU: {ckpt['iou']:.4f} | "
        f"F1: {ckpt['f1']:.4f} | Kappa: {ckpt['kappa']:.4f}"
    )
if 'best_ber' in ckpt:
    print(f"Best BER: {ckpt['best_ber']:.2f}%")

print("\nModel state dict keys (first 5):")
if 'model_state_dict' in ckpt:
    keys = list(ckpt['model_state_dict'].keys())[:5]
    for key in keys:
        print(f"  {key}")
        
print("=" * 60)
