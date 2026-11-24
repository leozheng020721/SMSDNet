import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# ==========================================
# 1. 定义注意力机制模块 (以 CBAM 为例)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 注意力通道数的缩放
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        identity = x

        out = self.ca(x) * x
        out = self.sa(out) * out
        return out + identity  # 残差连接，保留原始特征信息


# ==========================================
# 1. 定义 Coordinate Attention (CA) 模块
# ==========================================
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        # 分别沿着 X 和 Y 方向进行池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # X 轴和 Y 轴特征提取
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        # 分离特征并生成注意力权重
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 将 X 和 Y 方向的注意力权重同时作用于原特征图
        out = identity * a_w * a_h
        
        # ★ 残差连接，保证训练稳定
        return out + identity
    

# ==========================================
# 2. 继承并重写 SMP 的 SegFormer
# ==========================================
class SegFormerWithAttention(smp.Segformer):
    def __init__(self, *args, **kwargs):
        # 初始化父类 (SMP 的 Segformer)
        super().__init__(*args, **kwargs)
        
        # SMP中SegFormer默认的解码器输出通道数是256，可以通过 decoder_segmentation_channels 修改
        # 我们安全地获取这个值来初始化注意力模块
        decoder_channels = kwargs.get('decoder_segmentation_channels', 256)
        
        # 实例化注意力模块
        self.attention = CBAM(channels=decoder_channels)
        # self.attention = CoordAtt(inp=decoder_channels)

    def forward(self, x):
        # 1. 特征提取 (Encoder)
        features = self.encoder(x)
        
        # 2. 解码器融合多尺度特征 (Decoder)
        decoder_output = self.decoder(*features)
        
        # 3. ★ 注入注意力机制 ★
        attended_output = self.attention(decoder_output)
        
        # 4. 最终分割头 (Segmentation Head)
        masks = self.segmentation_head(attended_output)
        
        return masks

# ==========================================
# 3. 修改你的 build_model 函数
# ==========================================
def build_model(config):
    """
    构建带有注意力机制的SegFormer模型。
    """
    print(f"Building Model: {config['model']['arch']} with encoder {config['model']['encoder']} + Attention")
    
    # 使用我们自定义的包含注意力的子类替代原生的 smp.Segformer
    model = SegFormerWithAttention(
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

