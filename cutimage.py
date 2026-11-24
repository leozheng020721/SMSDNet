import cv2
import numpy as np
import os

# ================= 配置区 =================
# 你想要裁剪的图片路径
IMAGE_PATH = r"C:\Users\Administrator\Downloads\test\DSC_5772.jpg" 
# 裁剪后保存的图片名称
SAVE_PATH = r"C:\Users\Administrator\Downloads\test\test.jpg"
# 裁剪的目标尺寸
CROP_SIZE = 128
# ==========================================

img = None
clone = None
window_name = f"Crop Tool - {CROP_SIZE}x{CROP_SIZE} (Press 'q' to quit)"

def interactive_crop(event, x, y, flags, param):
    global img, clone

    # 1. 鼠标移动时，实时更新预览框
    if event == cv2.EVENT_MOUSEMOVE:
        clone = img.copy()
        half = CROP_SIZE // 2
        
        # 使用 np.clip 限制边界，防止预览框超出图片范围导致崩溃
        x_center = np.clip(x, half, img.shape[1] - half)
        y_center = np.clip(y, half, img.shape[0] - half)
        
        top_left = (x_center - half, y_center - half)
        bottom_right = (x_center + half, y_center + half)
        
        # 绘制绿色高亮预览框
        cv2.rectangle(clone, top_left, bottom_right, (0, 255, 0), 2)
        cv2.imshow(window_name, clone)

    # 2. 鼠标左键单击时，执行裁剪并保存
    elif event == cv2.EVENT_LBUTTONDOWN:
        half = CROP_SIZE // 2
        x_center = np.clip(x, half, img.shape[1] - half)
        y_center = np.clip(y, half, img.shape[0] - half)
        
        top_left_x, top_left_y = x_center - half, y_center - half
        
        # 提取裁剪区域
        cropped = img[top_left_y : top_left_y + CROP_SIZE, top_left_x : top_left_x + CROP_SIZE]
        
        # 保存图片
        cv2.imwrite(SAVE_PATH, cropped)
        print(f"✅ 裁剪成功！已保存至当前目录的 {SAVE_PATH}")
        
        # 完成后自动退出
        cv2.destroyAllWindows()
        exit()

def main():
    global img, clone
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 找不到图片，请检查路径: {IMAGE_PATH}")
        return

    img = cv2.imread(IMAGE_PATH)
    
    # 尺寸校验：确保原图大于 128x128
    if img.shape[0] < CROP_SIZE or img.shape[1] < CROP_SIZE:
        print(f"❌ 图片尺寸 ({img.shape[1]}x{img.shape[0]}) 太小，无法裁剪 {CROP_SIZE}x{CROP_SIZE} 的区域！")
        return

    clone = img.copy()
    cv2.namedWindow(window_name)
    
    # 绑定鼠标回调函数
    cv2.setMouseCallback(window_name, interactive_crop)

    print("🖱️  操作说明：")
    print("1. 移动鼠标即可预览裁剪区域（绿色框）。")
    print("2. 在合适的位置【点击鼠标左键】确认裁剪。")
    print("3. 按下键盘上的【q】键可退出程序。")
    
    while True:
        cv2.imshow(window_name, clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("退出裁剪程序。")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()