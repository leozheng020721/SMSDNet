"""Torch color transforms for RGB images in [0, 1], using a D65 Lab white point.

HSV follows OpenCV's floating-point convention with hue divided by 360.
Lab is normalized as L / 100, (a + 128) / 255, (b + 128) / 255.
Reference: https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html
"""

import torch


def rgb_to_hsv(rgb):
    """Convert Bx3xHxW sRGB into normalized HSV without leaving the device."""
    red, green, blue = rgb.unbind(dim=1)
    value, maximum_channel = rgb.max(dim=1)
    minimum = rgb.min(dim=1).values
    delta = value - minimum
    safe_delta = delta.clamp_min(1e-6)

    hue_red = (green - blue) / safe_delta
    hue_green = (blue - red) / safe_delta + 2.0
    hue_blue = (red - green) / safe_delta + 4.0
    hue = torch.where(
        maximum_channel == 0,
        hue_red,
        torch.where(maximum_channel == 1, hue_green, hue_blue),
    ) / 6.0
    hue = torch.where(hue < 0, hue + 1.0, hue)
    hue = torch.where(delta > 0, hue, torch.zeros_like(hue))
    saturation = delta / value.clamp_min(1e-6)
    return torch.stack((hue, saturation, value), dim=1)


def rgb_to_lab(rgb):
    """Convert Bx3xHxW sRGB into normalized CIE Lab (D65)."""
    # Linearize sRGB before the XYZ transform; all branches remain finite at 0.
    linear = torch.where(
        rgb > 0.04045,
        ((rgb + 0.055) / 1.055).clamp_min(1e-6).pow(2.4),
        rgb / 12.92,
    )
    red, green, blue = linear.unbind(dim=1)
    x = (0.412453 * red + 0.357580 * green + 0.180423 * blue) / 0.950456
    y = 0.212671 * red + 0.715160 * green + 0.072169 * blue
    z = (0.019334 * red + 0.119193 * green + 0.950227 * blue) / 1.088754
    xyz = torch.stack((x, y, z), dim=1)
    delta = 6.0 / 29.0
    transformed = torch.where(
        xyz > delta ** 3,
        xyz.clamp_min(1e-6).pow(1.0 / 3.0),
        xyz / (3.0 * delta ** 2) + 4.0 / 29.0,
    )
    fx, fy, fz = transformed.unbind(dim=1)
    lightness = (116.0 * fy - 16.0) / 100.0
    a = (500.0 * (fx - fy) + 128.0) / 255.0
    b = (200.0 * (fy - fz) + 128.0) / 255.0
    return torch.stack((lightness, a, b), dim=1)
