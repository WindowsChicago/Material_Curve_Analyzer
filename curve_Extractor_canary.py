import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.cluster import DBSCAN
import webcolors
import colorsys

class CurveExtractor:
    def __init__(self, image_path):
        """初始化曲线提取器"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为RGB格式
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image_rgb.shape[:2]
        
        # 转换为HSV格式用于颜色匹配
        self.image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
    def hex_to_rgb(self, hex_color):
        """将十六进制颜色转换为RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def hex_to_hsv(self, hex_color):
        """将十六进制颜色转换为HSV"""
        rgb = self.hex_to_rgb(hex_color)
        r, g, b = [x/255.0 for x in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        # 转换为OpenCV的HSV范围: H: 0-180, S: 0-255, V: 0-255
        return (int(h * 180), int(s * 255), int(v * 255))
    
    def find_color_mask_rgb(self, target_color, tolerance=40):
        """使用RGB颜色空间创建目标颜色的掩码"""
        target_rgb = self.hex_to_rgb(target_color)
        
        # 转换为numpy数组便于计算
        target_array = np.array(target_rgb, dtype=np.uint8)
        
        # 计算颜色距离
        color_diff = np.sqrt(np.sum((self.image_rgb.astype(np.float32) - target_array) ** 2, axis=2))
        
        # 创建掩码
        mask = color_diff < tolerance
        
        return mask.astype(np.uint8) * 255
    
    def find_color_mask_hsv(self, target_color, hue_tolerance=15, sat_tolerance=60, val_tolerance=60):
        """使用HSV颜色空间创建目标颜色的掩码（更灵活）"""
        target_hsv = self.hex_to_hsv(target_color)
        
        # 创建HSV范围
        lower_bound = np.array([
            max(0, target_hsv[0] - hue_tolerance),
            max(0, target_hsv[1] - sat_tolerance),
            max(0, target_hsv[2] - val_tolerance)
        ])
        
        upper_bound = np.array([
            min(180, target_hsv[0] + hue_tolerance),
            min(255, target_hsv[1] + sat_tolerance),
            min(255, target_hsv[2] + val_tolerance)
        ])
        
        # 创建掩码
        mask = cv2.inRange(self.image_hsv, lower_bound, upper_bound)
        
        return mask
    
    def find_color_mask_adaptive(self, target_color, tolerance=40, use_hsv=True):
        """自适应颜色掩码 - 结合RGB和HSV的优点"""
        if use_hsv:
            # 使用HSV，但根据tolerance调整各通道容差
            hue_tol = max(10, min(30, tolerance // 3))
            sat_tol = max(30, min(80, tolerance * 2))
            val_tol = max(30, min(80, tolerance * 2))
            mask = self.find_color_mask_hsv(target_color, hue_tol, sat_tol, val_tol)
        else:
            # 使用RGB，但放宽容差
            relaxed_tolerance = min(80, tolerance * 1.5)
            mask = self.find_color_mask_rgb(target_color, relaxed_tolerance)
        
        # 形态学操作来连接断开的区域
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def find_color_mask(self, target_color, tolerance=40):
        """创建目标颜色的掩码 - 主入口点，使用改进的策略"""
        # 尝试自适应HSV方法
        mask = self.find_color_mask_adaptive(target_color, tolerance, use_hsv=True)
        
        # 检查是否找到足够的像素
        if np.sum(mask > 0) < 50:  # 如果像素太少
            # 尝试放宽条件
            relaxed_mask = self.find_color_mask_adaptive(target_color, tolerance * 2, use_hsv=True)
            if np.sum(relaxed_mask > 0) > np.sum(mask > 0):
                mask = relaxed_mask
        
        return mask
    
    def remove_axes_and_text(self, mask):
        """去除坐标轴、边框和文字"""
        # 创建处理后的掩码
        processed_mask = mask.copy()
        
        # 方法1: 通过边缘检测去除边框
        edges = cv2.Canny(mask, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤掉小轮廓（可能是噪声）和大轮廓（可能是边框）
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # 过滤条件：面积适中，不在图像边缘
            if (area > 50 and area < (self.width * self.height * 0.8) and
                x > self.width * 0.05 and x + w < self.width * 0.95 and
                y > self.height * 0.05 and y + h < self.height * 0.95):
                filtered_contours.append(contour)
        
        # 创建新的掩码
        new_mask = np.zeros_like(mask)
        cv2.drawContours(new_mask, filtered_contours, -1, 255, -1)
        
        return new_mask
    
    def extract_curve_points(self, mask):
        """从掩码中提取曲线点"""
        # 找到所有非零像素的位置
        points = np.column_stack(np.where(mask > 0))
        
        if len(points) == 0:
            return np.array([])
        
        # 将坐标转换为 (x, y) 格式
        points = points[:, [1, 0]]  # 从 (y, x) 转换为 (x, y)
        
        # 使用DBSCAN聚类去除离散点
        if len(points) > 10:
            # 自适应调整聚类参数
            eps = max(3, min(10, 2000 / len(points)))
            min_samples = max(3, min(10, len(points) // 100))
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
            
            # 找到最大的簇
            unique_labels, counts = np.unique(clustering.labels_[clustering.labels_ >= 0], return_counts=True)
            if len(unique_labels) > 0:
                main_cluster = unique_labels[np.argmax(counts)]
                main_cluster_mask = clustering.labels_ == main_cluster
                points = points[main_cluster_mask]
        
        return points
    
    def sort_curve_points(self, points):
        """对曲线点进行排序"""
        if len(points) == 0:
            return points
        
        # 计算点的中心
        center = np.mean(points, axis=0)
        
        # 按角度排序（从左侧开始）
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        
        return points[sorted_indices]
    
    def resample_curve(self, points, num_points=128):
        """重新采样曲线到指定数量的点"""
        if len(points) < 3:
            return np.array([])
        
        # 对点进行排序
        sorted_points = self.sort_curve_points(points)
        
        # 计算累积距离
        distances = np.cumsum(np.sqrt(np.sum(np.diff(sorted_points, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0)
        
        # 归一化距离到 [0, 1]
        normalized_distances = distances / distances[-1]
        
        # 创建插值函数
        try:
            fx = interpolate.interp1d(normalized_distances, sorted_points[:, 0], kind='linear')
            fy = interpolate.interp1d(normalized_distances, sorted_points[:, 1], kind='linear')
            
            # 在新距离上采样
            new_distances = np.linspace(0, 1, num_points)
            new_x = fx(new_distances)
            new_y = fy(new_distances)
            
            return np.column_stack((new_x, new_y))
        except:
            # 如果插值失败，返回原始点的均匀采样
            indices = np.linspace(0, len(sorted_points)-1, num_points).astype(int)
            return sorted_points[indices]
    
    def extract_curve(self, target_color, num_points=128, tolerance=40, visualize=False):
        """主函数：提取指定颜色的曲线"""
        print(f"提取颜色: {target_color}")
        
        # 1. 创建颜色掩码
        color_mask = self.find_color_mask(target_color, tolerance)
        
        # 2. 去除坐标轴和文字
        cleaned_mask = self.remove_axes_and_text(color_mask)
        
        # 3. 提取曲线点
        curve_points = self.extract_curve_points(cleaned_mask)
        
        if len(curve_points) == 0:
            print("未找到曲线点")
            return np.array([])
        
        print(f"找到 {len(curve_points)} 个原始点")
        
        # 4. 重新采样到指定数量的点
        resampled_points = self.resample_curve(curve_points, num_points)
        
        if len(resampled_points) == 0:
            print("重新采样失败")
            return np.array([])
        
        print(f"重新采样到 {len(resampled_points)} 个点")
        
        # 5. 可视化结果（可选）
        if visualize:
            self.visualize_extraction(color_mask, cleaned_mask, resampled_points)
        
        return resampled_points
    
    def visualize_extraction(self, original_mask, cleaned_mask, final_points):
        """可视化提取过程"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(self.image_rgb)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 颜色掩码
        axes[1].imshow(original_mask, cmap='gray')
        axes[1].set_title('颜色掩码')
        axes[1].axis('off')
        
        # 清理后的掩码和提取的点
        axes[2].imshow(cleaned_mask, cmap='gray')
        if len(final_points) > 0:
            axes[2].scatter(final_points[:, 0], final_points[:, 1], 
                           c='red', s=10, marker='o')
        axes[2].set_title('提取的曲线点')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    # 使用示例
    image_path = "1.jpg"  # 替换为您的图像路径
    target_color = "#c4fff"  # 替换为您要提取的颜色
    
    try:
        # 创建提取器实例
        extractor = CurveExtractor(image_path)
        
        # 提取曲线
        curve_points = extractor.extract_curve(
            target_color=target_color,
            num_points=128,
            tolerance=10,
            visualize=True  # 设置为True可以看到处理过程
        )
        
        if len(curve_points) > 0:
            print("\n提取的128个点坐标:")
            print("格式: (x, y)")
            for i, point in enumerate(curve_points):
                print(f"点 {i+1:3d}: ({point[0]:6.1f}, {point[1]:6.1f})")
            
            # 保存到文件
            np.savetxt(f"curve_points_{target_color}.txt", curve_points, 
                      fmt='%.2f', delimiter=',', header='x,y')
            print(f"\n坐标已保存到: curve_points_{target_color}.txt")
        else:
            print("未能成功提取曲线")
            
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()