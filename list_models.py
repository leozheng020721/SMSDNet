import torch

checkpoints = [
    r'D:\segformer\checkpoints\KappaSet_SegFormer_MiT-B3_RGB_Baseline_best.pth',
    r'D:\segformer\checkpoints\KappaSet_SegFormer_MiT-B3_RGB_Baseline_last.pth',
    r'D:\segformer\checkpoints\KappaSet_SegFormer_MiT-B3_RGB_MergedCloud_best.pth',
    r'D:\segformer\checkpoints\KappaSet_SegFormer_MiT-B3_RGB_MergedCloud_last.pth',
]

for path in checkpoints:
    print("=" * 60)
    print(f"File: {path.split(chr(92))[-1]}")
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
        metric_names = ('ber', 'iou', 'f1', 'kappa')
        if all(name in ckpt for name in metric_names):
            print(
                f"Metrics - BER: {ckpt['ber']:.2f}% | IoU: {ckpt['iou']:.4f} | "
                f"F1: {ckpt['f1']:.4f} | Kappa: {ckpt['kappa']:.4f}"
            )
        else:
            print("Metrics: unavailable in this legacy checkpoint")
        if 'best_ber' in ckpt:
            print(f"Best BER: {ckpt['best_ber']:.2f}%")
    except Exception as e:
        print(f"Error: {e}")
    print()
