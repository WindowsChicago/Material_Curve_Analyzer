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
    
    def extract_curve_points_simple(self, mask, num_points=128):
        """简化的曲线点提取策略"""
        # 找到所有非零像素的位置
        points = np.column_stack(np.where(mask > 0))
        
        if len(points) == 0:
            return np.array([])
        
        # 将坐标转换为 (x, y) 格式
        points = points[:, [1, 0]]  # 从 (y, x) 转换为 (x, y)
        
        # 按x坐标分组，对每个x坐标的y值取平均
        x_coords = np.unique(points[:, 0])
        averaged_points = []
        
        for x in x_coords:
            # 找到当前x坐标的所有点
            same_x_points = points[points[:, 0] == x]
            y_values = same_x_points[:, 1]
            
            # 如果y值差异太大，可能存在多个曲线分支，取最上面的
            if len(y_values) > 1:
                y_diff = np.max(y_values) - np.min(y_values)
                if y_diff > self.height * 0.1:  # 如果y方向差异超过图像高度的10%
                    # 只保留最上面的点（y值最小）
                    y_values = [np.min(y_values)]
            
            # 计算平均y值
            avg_y = np.mean(y_values)
            averaged_points.append([x, avg_y])
        
        averaged_points = np.array(averaged_points)
        
        if len(averaged_points) == 0:
            return np.array([])
        
        # 按x坐标排序
        averaged_points = averaged_points[averaged_points[:, 0].argsort()]
        
        # 找到最左和最右的x坐标
        x_min = np.min(averaged_points[:, 0])
        x_max = np.max(averaged_points[:, 0])
        
        # 计算x方向的步长
        x_step = (x_max - x_min) / (num_points - 1)
        
        # 均匀采样
        sampled_points = []
        for i in range(num_points):
            target_x = x_min + i * x_step
            
            # 找到最接近目标x的实际点
            distances = np.abs(averaged_points[:, 0] - target_x)
            closest_idx = np.argmin(distances)
            
            sampled_points.append(averaged_points[closest_idx])
        
        return np.array(sampled_points)
    
    def extract_curve(self, target_color, num_points=128, tolerance=40, visualize=False):
        """主函数：提取指定颜色的曲线"""
        print(f"提取颜色: {target_color}")
        
        # 1. 创建颜色掩码
        color_mask = self.find_color_mask(target_color, tolerance)
        
        # 2. 使用简化的曲线点提取方法
        curve_points = self.extract_curve_points_simple(color_mask, num_points)
        
        if len(curve_points) == 0:
            print("未找到曲线点")
            return np.array([])
        
        print(f"提取到 {len(curve_points)} 个点")
        
        # 3. 可视化结果（可选）
        if visualize:
            self.visualize_extraction_simple(color_mask, curve_points)
        
        return curve_points
    
    def visualize_extraction_simple(self, mask, final_points):
        """简化的可视化提取过程"""
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # 原始图像
        axes[0].imshow(self.image_rgb)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 颜色掩码和提取的点
        axes[1].imshow(mask, cmap='gray')
        if len(final_points) > 0:
            axes[1].scatter(final_points[:, 0], final_points[:, 1], 
                           c='red', s=10, marker='o')
        axes[1].set_title('提取的曲线点')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    # 使用示例
    image_path = "1.jpg"  # 替换为您的图像路径
    target_color = "#d3d3d3"  # 替换为您要提取的颜色
    
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