# curve_api.py
import cv2
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Any, Optional

# 导入现有的功能模块
from curve_Extractor_RC9 import CurveExtractor
from legend_API import LegendDetectionPipeline

class CurveAnalysisAPI:
    """曲线分析API - 整合图例检测和曲线提取功能"""
    
    def __init__(self, yolo_model_path: str = "best-SimAM.onnx"):
        """
        初始化API
        
        Args:
            yolo_model_path: YOLO模型文件路径
        """
        # 初始化图例检测管道
        self.legend_pipeline = LegendDetectionPipeline()
        
        # 存储结果
        self.results = {}
        
    def _mask_legend_regions(self, image_path: str, legend_results: List[Dict]) -> str:
        """
        用白色覆盖所有图例区域，防止图例被误识别为曲线
        
        Args:
            image_path: 原始图像路径
            legend_results: 图例检测结果
            
        Returns:
            处理后的图像临时文件路径
        """
        import tempfile
        
        # 读取原始图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 复制图像，避免修改原始图像
        masked_img = img.copy()
        
        # 用白色填充所有图例区域
        for legend_result in legend_results:
            bbox = legend_result['bbox']
            x, y, w, h = bbox
            
            # 确保坐标在图像范围内
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img.shape[1], int(x + w))
            y2 = min(img.shape[0], int(y + h))
            
            # 用白色填充矩形区域
            cv2.rectangle(masked_img, (x1, y1), (x2, y2), (255, 255, 255), -1)
        
        # 创建临时文件保存处理后的图像
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        cv2.imwrite(temp_file.name, masked_img)
        
        print(f"已用白色覆盖 {len(legend_results)} 个图例区域")
        return temp_file.name
    
    def analyze_image(self, image_path: str, 
                     yolo_threshold: float = 0.3,
                     curve_points: int = 1024,
                     color_tolerance: int = 23,
                     visualize: bool = False) -> Dict[str, Any]:
        """
        分析图像，提取图例和对应的曲线数据
        
        Args:
            image_path: 输入图像路径
            yolo_threshold: YOLO检测阈值
            curve_points: 每条曲线采样点数
            color_tolerance: 颜色容差
            visualize: 是否可视化处理过程
            
        Returns:
            包含图例和曲线数据的字典
        """
        print("=" * 60)
        print("开始分析图像:", image_path)
        print("=" * 60)
        
        # 验证图像文件存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 步骤1: 检测图例区域和颜色信息
        print("步骤1: 检测图例信息...")
        legend_results = self.legend_pipeline.process_image_pipeline(
            image_path, yolo_threshold
        )
        
        if not legend_results:
            print("警告: 未检测到图例区域")
            return {"error": "未检测到图例区域"}
        
        # 步骤1.5: 用白色覆盖图例区域
        print("\n步骤1.5: 用白色覆盖图例区域...")
        masked_image_path = self._mask_legend_regions(image_path, legend_results)
        
        # 步骤2: 提取曲线数据（使用覆盖后的图像）
        print("\n步骤2: 提取曲线数据...")
        curve_extractor = CurveExtractor(masked_image_path)
        all_curves_data = []
        
        for i, legend_result in enumerate(legend_results):
            print(f"\n处理图例区域 {i+1}...")
            
            if legend_result.get('texts') and legend_result.get('colors'):
                # 处理每个图例项
                for j, (legend_text, color_hex) in enumerate(zip(
                    legend_result['texts'], legend_result['colors']
                )):
                    print(f"  提取图例 '{legend_text}' (颜色: {color_hex}) 的曲线...")
                    
                    try:
                        # 提取该颜色对应的曲线
                        curve_points_data = curve_extractor.extract_curve(
                            target_color=color_hex,
                            num_points=curve_points,
                            tolerance=color_tolerance,
                            visualize=visualize
                        )
                        
                        if len(curve_points_data) > 0:
                            # 转换为列表格式
                            points_list = curve_points_data.tolist()
                            
                            curve_data = {
                                "legend_id": f"legend_{i+1}_{j+1}",
                                "legend_name": legend_text,
                                "color": color_hex,
                                "points_count": len(points_list),
                                "points": points_list,
                                "bbox": legend_result['bbox'],
                                "confidence": float(legend_result['confidence'])
                            }
                            
                            all_curves_data.append(curve_data)
                            print(f"  成功提取 {len(points_list)} 个点")
                        else:
                            print(f"  警告: 未找到颜色 {color_hex} 对应的曲线")
                            
                    except Exception as e:
                        print(f"  提取曲线时出错: {e}")
            else:
                print(f"  图例区域 {i+1}: 只检测到黑色")
                # 处理黑色图例
                try:
                    curve_points_data = curve_extractor.extract_curve(
                        target_color="#000000",
                        num_points=curve_points,
                        tolerance=color_tolerance,
                        visualize=visualize
                    )
                    
                    if len(curve_points_data) > 0:
                        points_list = curve_points_data.tolist()
                        
                        curve_data = {
                            "legend_id": f"legend_{i+1}_black",
                            "legend_name": "黑色曲线",
                            "color": "#000000",
                            "points_count": len(points_list),
                            "points": points_list,
                            "bbox": legend_result['bbox'],
                            "confidence": float(legend_result['confidence'])
                        }
                        
                        all_curves_data.append(curve_data)
                        print(f"  成功提取黑色曲线的 {len(points_list)} 个点")
                        
                except Exception as e:
                    print(f"  提取黑色曲线时出错: {e}")
        
        # 清理临时文件
        try:
            if os.path.exists(masked_image_path):
                os.unlink(masked_image_path)
                print(f"\n已清理临时文件: {masked_image_path}")
        except Exception as e:
            print(f"清理临时文件时出错: {e}")
        
        # 构建最终结果
        final_result = {
            "image_path": image_path,
            "total_legends": len(all_curves_data),
            "curves": all_curves_data,
            "processing_info": {
                "yolo_threshold": yolo_threshold,
                "curve_points": curve_points,
                "color_tolerance": color_tolerance,
                "legend_regions_masked": True  # 标记已进行图例区域覆盖
            }
        }
        
        self.results = final_result
        return final_result
    
    def get_legend_points(self, legend_name: Optional[str] = None) -> List[Dict]:
        """
        获取指定图例名称的曲线点数据
        
        Args:
            legend_name: 图例名称，如果为None则返回所有
            
        Returns:
            曲线点数据列表
        """
        if not self.results:
            return []
        
        if legend_name is None:
            return self.results.get("curves", [])
        else:
            return [curve for curve in self.results.get("curves", []) 
                   if curve["legend_name"] == legend_name]
    
    def save_results(self, output_path: str, format: str = "json"):
        """
        保存分析结果到文件
        
        Args:
            output_path: 输出文件路径
            format: 输出格式，支持 'json' 或 'txt'
        """
        if not self.results:
            print("没有可保存的结果")
            return
        
        if format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存为JSON: {output_path}")
            
        elif format.lower() == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"图像分析结果: {self.results['image_path']}\n")
                f.write(f"总图例数量: {self.results['total_legends']}\n")
                f.write("=" * 50 + "\n\n")
                
                for curve in self.results["curves"]:
                    f.write(f"图例ID: {curve['legend_id']}\n")
                    f.write(f"图例名称: {curve['legend_name']}\n")
                    f.write(f"颜色: {curve['color']}\n")
                    f.write(f"点数: {curve['points_count']}\n")
                    f.write("坐标点 (x, y):\n")
                    
                    for i, point in enumerate(curve["points"]):
                        f.write(f"  点 {i+1:3d}: ({point[0]:8.2f}, {point[1]:8.2f})\n")
                    
                    f.write("-" * 30 + "\n")
            
            print(f"结果已保存为TXT: {output_path}")
        
        else:
            raise ValueError("不支持的格式，请使用 'json' 或 'txt'")
    
    def visualize_results(self, image_path: str, save_path: Optional[str] = None):
        """
        可视化分析结果
        
        Args:
            image_path: 原始图像路径
            save_path: 保存路径，如果为None则显示不保存
        """
        if not self.results:
            print("没有可可视化的结果")
            return
        
        try:
            import matplotlib.pyplot as plt
            plt.rc("font", family='AR PL UKai CN') #Ubuntu
            #plt.rc("font", family='Microsoft YaHei') #Windows
            from matplotlib.patches import Rectangle

            
            # 读取图像
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # 左侧：显示检测结果和曲线
            ax1.imshow(img_rgb)
            ax1.set_title('检测结果和提取的曲线')
            ax1.axis('off')
            
            # 绘制边界框和曲线点
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.results["curves"])))
            legend_handles = []
            
            for i, curve in enumerate(self.results["curves"]):
                color = colors[i]
                
                # 绘制边界框
                x, y, w, h = curve["bbox"]
                rect = Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor=color, facecolor='none')
                ax1.add_patch(rect)
                
                # 绘制曲线点
                points = np.array(curve["points"])
                if len(points) > 0:
                    ax1.scatter(points[:, 0], points[:, 1], 
                              color=color, s=20, alpha=0.7, 
                              label=curve["legend_name"])
                
                legend_handles.append(plt.Line2D([0], [0], color=color, 
                                               lw=4, label=curve["legend_name"]))
            
            ax1.legend(handles=legend_handles, loc='upper right')
            
            # 右侧：显示图例信息
            ax2.axis('off')
            ax2.set_title('图例信息')
            
            y_pos = 0.95
            for curve in self.results["curves"]:
                ax2.text(0.05, y_pos, f"{curve['legend_name']}", 
                        transform=ax2.transAxes, fontsize=10, fontweight='bold')
                ax2.text(0.05, y_pos-0.03, f"颜色: {curve['color']}", 
                        transform=ax2.transAxes, fontsize=9)
                ax2.text(0.05, y_pos-0.06, f"点数: {curve['points_count']}", 
                        transform=ax2.transAxes, fontsize=9)
                y_pos -= 0.12
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"可视化结果已保存: {save_path}")
            
            plt.show()
            
        except ImportError:
            print("警告: 无法导入matplotlib，跳过可视化")
        except Exception as e:
            print(f"可视化时出错: {e}")


# 创建全局API实例
_api_instance = None

def get_curve_api(yolo_model_path: str = "best.onnx") -> CurveAnalysisAPI:
    """
    获取曲线分析API实例（单例模式）
    
    Args:
        yolo_model_path: YOLO模型路径
        
    Returns:
        CurveAnalysisAPI实例
    """
    global _api_instance
    if _api_instance is None:
        _api_instance = CurveAnalysisAPI(yolo_model_path)
    return _api_instance

def analyze_image(image_path: str, 
                 yolo_threshold: float = 0.3,
                 curve_points: int = 128,
                 color_tolerance: int = 30,
                 visualize: bool = False) -> Dict[str, Any]:
    """
    快速分析图像（便捷函数）
    
    Args:
        image_path: 图像路径
        yolo_threshold: YOLO检测阈值
        curve_points: 曲线点数
        color_tolerance: 颜色容差
        visualize: 是否可视化
        
    Returns:
        分析结果字典
    """
    api = get_curve_api()
    return api.analyze_image(
        image_path=image_path,
        yolo_threshold=yolo_threshold,
        curve_points=curve_points,
        color_tolerance=color_tolerance,
        visualize=visualize
    )


# 使用示例和测试
if __name__ == "__main__":
    # 示例用法
    image_path = "fig2/004.jpg"  # 替换为您的图像路径
    
    try:
        # 使用API类
        print("\n使用API类")
        api = CurveAnalysisAPI()
        result = api.analyze_image(image_path, visualize=True)
        
        # 保存结果
        # api.save_results("analysis_result.json", "json")
        api.save_results("analysis_result.txt", "txt")
        
        # 获取特定图例的点
        if result["curves"]:
            first_legend = result["curves"][0]["legend_name"]
            points = api.get_legend_points(first_legend)
            print(f"\n图例 '{first_legend}' 的点数: {len(points[0]['points'])}")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")