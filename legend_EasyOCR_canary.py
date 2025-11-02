# import cv2
# import numpy as np
# from collections import Counter
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# plt.rc("font", family='AR PL UKai CN') #Ubuntu
# #plt.rc("font", family='Microsoft YaHei') #Windows

# def get_dominant_colors(image_path, k=8, display=True):
#     """
#     识别图片中的主要颜色并计算占比
    
#     参数:
#     image_path: 图片路径
#     k: 要提取的主要颜色数量
#     display: 是否显示结果
    
#     返回:
#     dominant_colors: 主要颜色的RGB值列表
#     percentages: 对应的占比列表
#     """
    
#     # 读取图片
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"无法读取图片: {image_path}")
    
#     # 转换颜色空间 BGR to RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # 重塑图片为像素列表
#     pixels = image_rgb.reshape(-1, 3)
    
#     # 使用K-means聚类找到主要颜色
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     kmeans.fit(pixels)
    
#     # 获取聚类中心和标签
#     colors = kmeans.cluster_centers_.astype(int)
#     labels = kmeans.labels_
    
#     # 计算每种颜色的像素数量
#     label_counts = Counter(labels)
    
#     # 计算每种颜色的占比
#     total_pixels = len(pixels)
#     percentages = [count / total_pixels * 100 for count in label_counts.values()]
    
#     # 按占比排序
#     sorted_indices = np.argsort(percentages)[::-1]  # 降序排列
#     dominant_colors = colors[sorted_indices]
#     percentages = [percentages[i] for i in sorted_indices]
    
#     if display:
#         display_results(image_rgb, dominant_colors, percentages)
    
#     return dominant_colors, percentages

# def rgb_to_hex(rgb):
#     """将RGB值转换为十六进制颜色代码"""
#     return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# def display_results(original_image, colors, percentages):
#     """显示原始图片和颜色分析结果"""
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
#     # 显示原始图片
#     ax1.imshow(original_image)
#     ax1.set_title('原始图片', fontsize=14)
#     ax1.axis('off')
    
#     # 显示颜色分析结果
#     color_patches = []
#     legend_labels = []
    
#     for i, (color, percent) in enumerate(zip(colors, percentages)):
#         hex_color = rgb_to_hex(color)
#         color_patches.append(Patch(color=hex_color))
#         legend_labels.append(f'颜色 {i+1}: {hex_color}\n({percent:.2f}%)')
    
#     # 创建颜色条
#     bar_height = 0.8
#     y_positions = np.arange(len(colors))
    
#     for i, (color, percent) in enumerate(zip(colors, percentages)):
#         ax2.barh(y_positions[i], percent, height=bar_height, 
#                 color=rgb_to_hex(color), edgecolor='black')
#         ax2.text(percent + 1, y_positions[i], f'{percent:.2f}%', 
#                 va='center', fontsize=10, fontweight='bold')
    
#     ax2.set_xlabel('占比 (%)', fontsize=12)
#     ax2.set_ylabel('颜色', fontsize=12)
#     ax2.set_title('颜色分布', fontsize=14)
#     ax2.set_yticks(y_positions)
#     ax2.set_yticklabels([f'颜色 {i+1}' for i in range(len(colors))])
#     ax2.grid(axis='x', alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()
    
#     # 打印详细结果
#     print("\n颜色分析结果:")
#     print("=" * 50)
#     for i, (color, percent) in enumerate(zip(colors, percentages)):
#         hex_code = rgb_to_hex(color)
#         print(f"颜色 {i+1}: RGB{tuple(color)} | {hex_code} | 占比: {percent:.2f}%")

# def find_similar_colors(colors, percentages, threshold=30):
#     """
#     合并相似的颜色
#     参数:
#     threshold: 颜色相似度阈值（欧氏距离）
#     """
#     merged_colors = []
#     merged_percentages = []
#     used_indices = set()
    
#     for i in range(len(colors)):
#         if i in used_indices:
#             continue
            
#         current_color = colors[i]
#         current_percent = percentages[i]
#         similar_indices = [i]
        
#         for j in range(i+1, len(colors)):
#             if j in used_indices:
#                 continue
                
#             # 计算颜色之间的欧氏距离
#             distance = np.linalg.norm(current_color - colors[j])
#             if distance < threshold:
#                 similar_indices.append(j)
#                 used_indices.add(j)
#                 current_percent += percentages[j]
        
#         # 计算平均颜色
#         avg_color = np.mean(colors[similar_indices], axis=0).astype(int)
#         merged_colors.append(avg_color)
#         merged_percentages.append(current_percent)
#         used_indices.add(i)
    
#     return merged_colors, merged_percentages

# # 使用示例
# if __name__ == "__main__":
#     # 示例1: 分析图片
#     try:
#         image_path = "5.jpg"  # 替换为你的图片路径
#         dominant_colors, percentages = get_dominant_colors(image_path, k=6)
        
#         # 可选：合并相似颜色
#         merged_colors, merged_percentages = find_similar_colors(dominant_colors, percentages)
#         print("\n合并相似颜色后的结果:")
#         for i, (color, percent) in enumerate(zip(merged_colors, merged_percentages)):
#             hex_code = rgb_to_hex(color)
#             print(f"颜色 {i+1}: RGB{tuple(color)} | {hex_code} | 占比: {percent:.2f}%")
            
#     except Exception as e:
#         print(f"错误: {e}")

import cv2
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rc("font", family='AR PL UKai CN') #Ubuntu
#plt.rc("font", family='Microsoft YaHei') #Windows

def get_dominant_colors(image_path, k=8, display=True):
    """
    识别图片中的主要颜色并计算占比
    
    参数:
    image_path: 图片路径
    k: 要提取的主要颜色数量
    display: 是否显示结果
    
    返回:
    dominant_colors: 主要颜色的RGB值列表
    percentages: 对应的占比列表
    """
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 转换颜色空间 BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 重塑图片为像素列表
    pixels = image_rgb.reshape(-1, 3)
    
    # 使用K-means聚类找到主要颜色
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # 获取聚类中心和标签
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    
    # 计算每种颜色的像素数量
    label_counts = Counter(labels)
    
    # 计算每种颜色的占比
    total_pixels = len(pixels)
    percentages = [count / total_pixels * 100 for count in label_counts.values()]
    
    # 按占比排序
    sorted_indices = np.argsort(percentages)[::-1]  # 降序排列
    dominant_colors = colors[sorted_indices]
    percentages = [percentages[i] for i in sorted_indices]
    
    if display:
        display_results(image_rgb, dominant_colors, percentages)
    
    return dominant_colors, percentages

def rgb_to_hex(rgb):
    """将RGB值转换为十六进制颜色代码"""
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def display_results(original_image, colors, percentages):
    """显示原始图片和颜色分析结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 显示原始图片
    ax1.imshow(original_image)
    ax1.set_title('原始图片', fontsize=14)
    ax1.axis('off')
    
    # 显示颜色分析结果
    color_patches = []
    legend_labels = []
    
    for i, (color, percent) in enumerate(zip(colors, percentages)):
        hex_color = rgb_to_hex(color)
        color_patches.append(Patch(color=hex_color))
        legend_labels.append(f'颜色 {i+1}: {hex_color}\n({percent:.2f}%)')
    
    # 创建颜色条
    bar_height = 0.8
    y_positions = np.arange(len(colors))
    
    for i, (color, percent) in enumerate(zip(colors, percentages)):
        ax2.barh(y_positions[i], percent, height=bar_height, 
                color=rgb_to_hex(color), edgecolor='black')
        ax2.text(percent + 1, y_positions[i], f'{percent:.2f}%', 
                va='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('占比 (%)', fontsize=12)
    ax2.set_ylabel('颜色', fontsize=12)
    ax2.set_title('颜色分布', fontsize=14)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([f'颜色 {i+1}' for i in range(len(colors))])
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细结果
    print("\n颜色分析结果:")
    print("=" * 50)
    for i, (color, percent) in enumerate(zip(colors, percentages)):
        hex_code = rgb_to_hex(color)
        print(f"颜色 {i+1}: RGB{tuple(color)} | {hex_code} | 占比: {percent:.2f}%")

def find_similar_colors(colors, percentages, threshold=30):
    """
    合并相似的颜色
    参数:
    threshold: 颜色相似度阈值（欧氏距离）
    """
    merged_colors = []
    merged_percentages = []
    used_indices = set()
    
    for i in range(len(colors)):
        if i in used_indices:
            continue
            
        current_color = colors[i]
        current_percent = percentages[i]
        similar_indices = [i]
        
        for j in range(i+1, len(colors)):
            if j in used_indices:
                continue
                
            # 计算颜色之间的欧氏距离
            distance = np.linalg.norm(current_color - colors[j])
            if distance < threshold:
                similar_indices.append(j)
                used_indices.add(j)
                current_percent += percentages[j]
        
        # 计算平均颜色
        avg_color = np.mean(colors[similar_indices], axis=0).astype(int)
        merged_colors.append(avg_color)
        merged_percentages.append(current_percent)
        used_indices.add(i)
    
    return merged_colors, merged_percentages

def find_most_dominant_non_bw_color(colors, percentages):
    """
    找出除黑色和白色之外占比最大且不超过50%的颜色
    
    参数:
    colors: 颜色列表 (RGB)
    percentages: 对应的占比列表
    
    返回:
    符合条件的颜色代码 (十六进制) 或 None (如果没有符合条件的颜色)
    """
    max_percentage = 0
    result_color = None
    
    for color, percentage in zip(colors, percentages):
        # 检查是否为黑色或白色 (RGB值接近0或255)
        is_black = all(c < 50 for c in color)  # 所有分量都小于50
        is_white = all(c > 200 for c in color)  # 所有分量都大于200
        
        # 排除黑色和白色，且占比不超过50%
        if not is_black and not is_white and percentage <= 50:
            if percentage > max_percentage:
                max_percentage = percentage
                result_color = color
    
    return rgb_to_hex(result_color) if result_color is not None else None

# 使用示例
if __name__ == "__main__":
    # 示例1: 分析图片
    try:
        image_path = "7.jpg"  # 替换为你的图片路径
        dominant_colors, percentages = get_dominant_colors(image_path, k=6)
        
        # 可选：合并相似颜色
        merged_colors, merged_percentages = find_similar_colors(dominant_colors, percentages)
        print("\n合并相似颜色后的结果:")
        for i, (color, percent) in enumerate(zip(merged_colors, merged_percentages)):
            hex_code = rgb_to_hex(color)
            print(f"颜色 {i+1}: RGB{tuple(color)} | {hex_code} | 占比: {percent:.2f}%")
        
        # 新增功能：找出除黑色和白色之外占比最大且不超过50%的颜色
        dominant_color_hex = find_most_dominant_non_bw_color(merged_colors, merged_percentages)
        if dominant_color_hex:
            print(f"\n除黑色和白色之外占比最大且不超过50%的颜色是: {dominant_color_hex}")
        else:
            print("\n未找到符合条件的颜色")
            
    except Exception as e:
        print(f"错误: {e}")