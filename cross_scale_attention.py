"""Cross-scale attention over the query pixel's row and column."""

import torch
from torch import nn
from torch.nn import functional as F


class CrossScaleCrissCrossAttention(nn.Module):
    """Query F3 with aligned F4 keys/values, using one joint row/column softmax.

    The zero-initialized residual scale preserves F3 at initialization. Q/K/V
    projections can consequently receive zero gradients on the first step.
    """

    def __init__(self, query_channels, context_channels, attention_channels=64,
                 value_channels=None):
        super().__init__()
        if value_channels is None:
            value_channels = query_channels
        channels = {
            "query_channels": query_channels,
            "context_channels": context_channels,
            "attention_channels": attention_channels,
            "value_channels": value_channels,
        }
        for name, count in channels.items():
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, count)

        self.query_projection = nn.Conv2d(query_channels, attention_channels, 1, bias=False)
        self.key_projection = nn.Conv2d(context_channels, attention_channels, 1, bias=False)
        self.value_projection = nn.Conv2d(context_channels, value_channels, 1, bias=False)
        self.out_projection = nn.Conv2d(value_channels, query_channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = attention_channels ** -0.5

    def forward(self, query_feature, context_feature):
        if not isinstance(query_feature, torch.Tensor) or not isinstance(context_feature, torch.Tensor):
            raise TypeError("Query and context features must be tensors")
        if query_feature.ndim != 4 or context_feature.ndim != 4:
            raise ValueError("Query and context features must have BCHW dimensions")
        if query_feature.shape[0] != context_feature.shape[0]:
            raise ValueError("Query and context features must have the same batch size")
        if query_feature.shape[1] != self.query_channels:
            raise ValueError(f"Query feature must have {self.query_channels} channels")
        if context_feature.shape[1] != self.context_channels:
            raise ValueError(f"Context feature must have {self.context_channels} channels")
        if any(size <= 0 for size in query_feature.shape) or any(size <= 0 for size in context_feature.shape):
            raise ValueError("Feature batch, channel and spatial dimensions must be positive")
        if query_feature.device != context_feature.device:
            raise ValueError("Query and context features must be on the same device")
        if query_feature.dtype != context_feature.dtype:
            raise ValueError("Query and context features must have the same dtype")
        if not query_feature.is_floating_point() or not context_feature.is_floating_point():
            raise TypeError("Query and context features must be floating-point tensors")

        height, width = query_feature.shape[-2:]
        aligned_context = F.interpolate(
            context_feature, size=(height, width), mode="bilinear", align_corners=False,
        )
        query = self.query_projection(query_feature)
        key = self.key_projection(aligned_context)
        value = self.value_projection(aligned_context)
        attention_dtype = (
            torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
        )

        # Autocast must stay disabled here: merely casting Q/K to float32 does
        # not prevent autocast-enabled matrix products from reverting to fp16.
        with torch.autocast(device_type=query_feature.device.type, enabled=False):
            query = query.to(dtype=attention_dtype)
            key = key.to(dtype=attention_dtype)
            attention_value = value.to(dtype=attention_dtype)
            row_scores = torch.einsum("bchw,bchs->bhws", query, key) * self.scale
            column_scores = torch.einsum("bchw,bctw->bhwt", query, key) * self.scale

            # Keep the center in the row and mask its duplicate in the column.
            center = torch.eye(height, dtype=torch.bool, device=query.device)[None, :, None, :]
            column_scores = column_scores.masked_fill(center, float("-inf"))
            weights = torch.softmax(torch.cat((row_scores, column_scores), dim=-1), dim=-1)
            row_value = torch.einsum("bhws,bchs->bchw", weights[..., :width], attention_value)
            column_value = torch.einsum("bhwt,bctw->bchw", weights[..., width:], attention_value)
            attended = row_value + column_value

        residual = self.out_projection(attended.to(dtype=value.dtype))
        return query_feature + self.gamma.to(dtype=query_feature.dtype) * residual.to(dtype=query_feature.dtype)
