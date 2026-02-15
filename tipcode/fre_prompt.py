import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('test/Image/MAS_MarineFish_ElectricRay_Com_2867.jpg',0)
print(img.shape)

# 图像大小和窗口大小
image_size1 = img.shape[0]
image_size2 = img.shape[1]
window_size =20
image = img
# 创建一个随机的128x128图像
#image = np.random.randint(0, 255, (image_size, image_size))

# 修改处理方法，以统计每个窗口里像素值的总和，并找出总和最大的3个窗口

# 用于存储每个窗口的像素和及其位置
window_sums = []

# 遍历图像并计算每个窗口的像素和
for i in range(0, image_size1, window_size):
    for j in range(0, image_size2, window_size):
        # 提取当前窗口
        window = image[i:i+window_size, j:j+window_size]
        # 计算窗口的像素和
        window_sum = np.sum(window)
        # 存储窗口的像素和及其位置
        window_sums.append((window_sum, (i, j)))

# 对窗口按像素和排序并选出前3个
top_3_windows = sorted(window_sums, key=lambda x: x[0], reverse=True)[:10]

# 创建一个全黑的图像
result_image = np.zeros((image_size1, image_size2))

# 将像素和最大的3个窗口在结果图像中标记为白色
for _, (i, j) in top_3_windows:
    result_image[i:i+window_size, j:j+window_size] = 255

# 展示原始图像和处理后的图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(result_image, cmap='gray')
plt.title('Processed Image')
plt.show()
