import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.rc("font", family='AR PL UKai CN') #Ubuntu
#plt.rc("font", family='Microsoft YaHei') #Windows
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
    
    def extract_curve_points_advanced(self, mask, num_points=128):
        """改进的曲线点提取策略 - 强化顺序，去除极端点，连接曲线段"""
        # 找到所有非零像素的位置
        points = np.column_stack(np.where(mask > 0))
        
        if len(points) == 0:
            return np.array([])
        
        # 将坐标转换为 (x, y) 格式
        points = points[:, [1, 0]]  # 从 (y, x) 转换为 (x, y)
        
        # 步骤1: 按x坐标分组，对每个x坐标的y值取平均
        x_coords = np.unique(points[:, 0])
        averaged_points = []
        
        for x in x_coords:
            # 找到当前x坐标的所有点
            same_x_points = points[points[:, 0] == x]
            y_values = same_x_points[:, 1]
            
            # 如果y值差异太大，可能存在多个曲线分支
            if len(y_values) > 1:
                y_diff = np.max(y_values) - np.min(y_values)
                if y_diff > self.height * 0.1:  # 如果y方向差异超过图像高度的10%
                    # 计算每个簇到图像边缘的距离
                    edge_distances = []
                    for y in y_values:
                        # 计算到上边缘和下边缘的最小距离
                        dist_to_top = y
                        dist_to_bottom = self.height - y
                        min_dist_to_edge = min(dist_to_top, dist_to_bottom)
                        edge_distances.append(min_dist_to_edge)
                    
                    # 找到距离边缘最近的簇
                    min_edge_dist_idx = np.argmin(edge_distances)
                    # 移除距离边缘最近的簇，保留其他簇
                    y_values = np.delete(y_values, min_edge_dist_idx)
            
            # 计算平均y值
            avg_y = np.mean(y_values)
            averaged_points.append([x, avg_y])
        
        averaged_points = np.array(averaged_points)
        
        if len(averaged_points) == 0:
            return np.array([])
        
        # 步骤2: 按x坐标排序，强化从左到右的顺序
        averaged_points = averaged_points[averaged_points[:, 0].argsort()]
        
        # 步骤3: 去除极端点 - 基于局部变化率
        filtered_points = self.filter_extreme_points(averaged_points)
        
        # 步骤4: 连接曲线段 - 检测并填补大的间隙
        connected_points = self.connect_curve_segments(filtered_points)
        
        # 步骤5: 均匀采样到指定数量的点
        sampled_points = self.resample_curve(connected_points, num_points)
        
        return sampled_points
    
    def filter_extreme_points(self, points, window_size=1, threshold_factor=2.0):
        """去除极端点 - 基于局部变化率"""
        if len(points) <= window_size:
            return points
        
        filtered_points = []
        n = len(points)
        
        for i in range(n):
            # 计算局部窗口
            start_idx = max(0, i - window_size // 2)
            end_idx = min(n, i + window_size // 2 + 1)
            
            # 计算局部y值的均值和标准差
            local_y = points[start_idx:end_idx, 1]
            local_mean = np.mean(local_y)
            local_std = np.std(local_y)
            
            # 如果当前点与局部均值的差异小于阈值倍的标准差，则保留
            if abs(points[i, 1] - local_mean) < threshold_factor * local_std:
                filtered_points.append(points[i])
        
        return np.array(filtered_points) if filtered_points else points
    
    def connect_curve_segments(self, points, max_gap_ratio=0.05):
        """连接曲线段 - 检测并填补大的间隙"""
        if len(points) < 2:
            return points
        
        connected_points = [points[0]]
        
        for i in range(1, len(points)):
            # 计算当前点与前一个点的x间隙
            x_gap = points[i, 0] - points[i-1, 0]
            
            # 如果间隙过大，进行插值
            if x_gap > self.width * max_gap_ratio:
                # 计算需要插入的点数
                num_insert = int(x_gap / (self.width * 0.01))  # 每1%宽度插入一个点
                num_insert = max(1, min(10, num_insert))  # 限制插入点数
                
                # 线性插值
                x_start, y_start = points[i-1]
                x_end, y_end = points[i]
                
                for j in range(1, num_insert + 1):
                    ratio = j / (num_insert + 1)
                    x_insert = x_start + ratio * (x_end - x_start)
                    y_insert = y_start + ratio * (y_end - y_start)
                    connected_points.append([x_insert, y_insert])
            
            connected_points.append(points[i])
        
        return np.array(connected_points)
    
    def resample_curve(self, points, num_points):
        """将曲线重新采样到指定数量的点"""
        if len(points) < 2:
            return points
        
        # 按x坐标排序
        points = points[points[:, 0].argsort()]
        
        # 使用样条插值
        try:
            # 确保x坐标是严格递增的
            unique_x, unique_indices = np.unique(points[:, 0], return_index=True)
            if len(unique_x) < 2:
                return points
            
            unique_points = points[unique_indices]
            
            # 创建插值函数
            tck = interpolate.splrep(unique_points[:, 0], unique_points[:, 1], s=0)
            
            # 生成新的x坐标
            x_min = np.min(unique_points[:, 0])
            x_max = np.max(unique_points[:, 0])
            x_new = np.linspace(x_min, x_max, num_points)
            
            # 插值得到新的y坐标
            y_new = interpolate.splev(x_new, tck, der=0)
            
            return np.column_stack((x_new, y_new))
        except:
            # 如果样条插值失败，使用线性插值
            x_min = np.min(points[:, 0])
            x_max = np.max(points[:, 0])
            x_new = np.linspace(x_min, x_max, num_points)
            
            # 线性插值
            y_new = np.interp(x_new, points[:, 0], points[:, 1])
            
            return np.column_stack((x_new, y_new))
    
    def extract_curve(self, target_color, num_points=128, tolerance=40, visualize=False):
        """主函数：提取指定颜色的曲线"""
        print(f"提取颜色: {target_color}")
        
        # 1. 创建颜色掩码
        color_mask = self.find_color_mask(target_color, tolerance)
        
        # 2. 使用改进的曲线点提取方法
        curve_points = self.extract_curve_points_advanced(color_mask, num_points)
        
        if len(curve_points) == 0:
            print("未找到曲线点")
            return np.array([])
        
        print(f"提取到 {len(curve_points)} 个点")
        
        # 3. 可视化结果（可选）
        if visualize:
            self.visualize_extraction_advanced(color_mask, curve_points)
        
        return curve_points
    
    def visualize_extraction_advanced(self, mask, final_points):
        """改进的可视化提取过程"""
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # 原始图像
        axes[0].imshow(self.image_rgb)
        if len(final_points) > 0:
            axes[0].scatter(final_points[:, 0], final_points[:, 1], 
                           c='red', s=10, marker='o')
            axes[0].plot(final_points[:, 0], final_points[:, 1], 
                        c='red', linewidth=1, alpha=0.7)
        axes[0].set_title('原始图像与提取的曲线')
        axes[0].axis('off')
        
        # 颜色掩码和提取的点
        axes[1].imshow(mask, cmap='gray')
        if len(final_points) > 0:
            axes[1].scatter(final_points[:, 0], final_points[:, 1], 
                           c='red', s=10, marker='o')
            axes[1].plot(final_points[:, 0], final_points[:, 1], 
                        c='red', linewidth=1, alpha=0.7)
        axes[1].set_title('提取的曲线点')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    # 使用示例
    image_path = "7.jpg"  # 替换为您的图像路径
    target_color = "#0b0bf1"  # 替换为您要提取的颜色
    
    try:
        # 创建提取器实例
        extractor = CurveExtractor(image_path)
        
        # 提取曲线
        curve_points = extractor.extract_curve(
            target_color=target_color,
            num_points=128,
            tolerance=23,
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