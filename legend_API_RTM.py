import cv2
import numpy as np
import os
import tempfile
#from legend_YOLO import recognize
from legend_YOLO_SAMPLE_CUDA import recognize
from legend_EasyOCR_RTM import OCRLegendTextExtractor
from legend_YOLO_BOX_CUDA import clean_image_background_only

class LegendDetectionPipeline:
    def __init__(self):
        # 初始化YOLO检测器
        self.yolo_net = cv2.dnn.readNetFromONNX("best-SimAM.onnx")
        self.input_shape = (640, 640)
        self.model_classify = ["legend"]
        
        # 初始化图例提取器
        self.legend_extractor = OCRLegendTextExtractor()
    
    def preprocess_image_background(self, image_path, threshold=0.3):
        """
        预处理步骤：使用图例背景清除功能处理图片
        
        Args:
            image_path: 原始图像路径
            threshold: 检测阈值
        
        Returns:
            cleaned_image_path: 清除背景后的图像临时文件路径
        """
        print("步骤0: 使用图例背景清除功能预处理图像...")
        try:
            # 清除图像背景
            cleaned_img = clean_image_background_only(image_path, threshold=threshold, clean_background=True)
            
            # 保存处理后的图像为临时文件
            temp_dir = "temp_preprocessed"
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.jpg', 
                prefix='preprocessed_', 
                dir=temp_dir, 
                delete=False
            )
            temp_path = temp_file.name
            
            # 保存清除背景后的图像
            cv2.imwrite(temp_path, cleaned_img)
            print("图像背景清除完成")
            
            return temp_path
            
        except Exception as e:
            print(f"图像背景清除失败: {e}")
            # 如果失败，返回原始图像路径
            return image_path
    
    def detect_crop_legend_region(self, image_path, threshold=0.3):
        """使用YOLO检测图例区域并裁剪出来"""
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 准备YOLO输入
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, self.input_shape, swapRB=True, crop=False)
        self.yolo_net.setInput(blob)
        
        # 前向传播
        output = self.yolo_net.forward()
        output = output.transpose((0, 2, 1))
        
        height, width, _ = img.shape
        x_factor, y_factor = width / self.input_shape[0], height / self.input_shape[1]
        
        # 解析检测结果
        boxes = []
        scores = []
        
        for i in range(output[0].shape[0]):
            box = output[0][i]
            _, _, _, max_idx = cv2.minMaxLoc(box[4:])
            class_id = max_idx[1]
            score = box[4:][class_id]
            
            if score > threshold:
                scores.append(score)
                x, y, w, h = box[0].item(), box[1].item(), box[2].item(), box[3].item()
                x = int((x - 0.5 * w) * x_factor)
                y = int((y - 0.5 * h) * y_factor)
                w = int(w * x_factor)
                h = int(h * y_factor)
                box_coords = np.array([x, y, w, h])
                boxes.append(box_coords)
        
        # 应用NMS
        indexes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
        
        # 裁剪检测到的区域
        cropped_regions = []
        for i in indexes:
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            
            # 确保坐标在图像范围内
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)
            
            # 裁剪区域
            cropped_region = img[y:y+h, x:x+w]
            cropped_regions.append({
                'image': cropped_region,
                'bbox': (x, y, w, h),
                'score': scores[i]
            })
            
            print(f"检测到图例区域: 位置({x}, {y}), 尺寸({w}, {h}), 置信度: {scores[i]:.2f}")
        
        return cropped_regions
    
    def save_temp_image(self, image_array):
        """将图像数组保存为临时文件"""
        # 创建临时目录
        temp_dir = "temp_crops"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 生成临时文件名
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.jpg', 
            prefix='legend_crop_', 
            dir=temp_dir, 
            delete=False
        )
        temp_path = temp_file.name
        
        # 保存图像
        cv2.imwrite(temp_path, image_array)
        return temp_path
    
    def process_image_pipeline(self, image_path, yolo_threshold=0.3, background_clean_threshold=0.3):
        """完整的处理流程：背景清除 -> YOLO检测 -> 裁剪 -> 图例提取"""
        print("=" * 60)
        print("开始处理图像:", image_path)
        print("=" * 60)
        
        # 步骤0: 使用图例背景清除功能预处理图像
        print("步骤0: 使用图例背景清除功能预处理图像...")
        preprocessed_image_path = self.preprocess_image_background(image_path, threshold=background_clean_threshold)
        
        # 步骤1: 使用YOLO检测图例区域（在预处理后的图像上）
        print("步骤1: 使用YOLO检测图例区域...")
        cropped_regions = self.detect_crop_legend_region(preprocessed_image_path, yolo_threshold)
        
        if not cropped_regions:
            print("未检测到图例区域")
            # 清理临时文件
            if preprocessed_image_path != image_path:
                try:
                    os.unlink(preprocessed_image_path)
                except:
                    pass
            return []
        
        print(f"检测到 {len(cropped_regions)} 个图例区域")
        
        all_results = []
        
        # 步骤2: 对每个裁剪区域进行图例提取
        for i, region in enumerate(cropped_regions):
            print(f"\n步骤2: 处理第 {i+1} 个图例区域...")
            
            # 保存裁剪区域为临时文件
            temp_path = self.save_temp_image(region['image'])
            
            try:
                # 步骤3: 使用pytesseract提取图例信息
                print("步骤3: 使用OCR提取图例文本和颜色...")
                texts, colors = self.legend_extractor.extract_legend(temp_path)
                
                # 收集结果
                result = {
                    'region_id': i + 1,
                    'bbox': region['bbox'],
                    'confidence': region['score'],
                    'texts': texts,
                    'colors': colors,
                    'crop_path': temp_path
                }
                
                all_results.append(result)
                
                # 显示结果
                print(f"\n图例区域 {i+1} 提取结果:")
                print("-" * 40)
                if texts and colors:
                    for j, (text, color) in enumerate(zip(texts, colors)):
                        print(f"  {j+1}. 颜色: {color}, 文本: {text}")
                else:
                    print("  只检测到黑色")
                    print("  黑色: #000000")
                    
            except Exception as e:
                print(f"处理图例区域 {i+1} 时出错: {e}")
            finally:
                # 可选: 清理临时文件
                # os.unlink(temp_path)
                pass
        
        # 清理预处理图像临时文件
        if preprocessed_image_path != image_path:
            try:
                os.unlink(preprocessed_image_path)
            except:
                pass
        
        return all_results
    
    def visualize_all_results(self, image_path, results, save_path=None):
        """可视化所有结果"""
        import matplotlib.pyplot as plt
        plt.rc("font", family='AR PL UKai CN') #Ubuntu
        #plt.rc("font", family='Microsoft YaHei') #Windows
        
        # 读取原始图像
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左侧: 显示YOLO检测结果
        detection_img = img_rgb.copy()
        for result in results:
            x, y, w, h = result['bbox']
            cv2.rectangle(detection_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            label = f'Legend {result["region_id"]}: {result["confidence"]:.2f}'
            cv2.putText(detection_img, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        axes[0].imshow(detection_img)
        axes[0].set_title('YOLO图例检测结果')
        axes[0].axis('off')
        
        # 右侧: 显示提取的图例信息
        axes[1].axis('off')
        axes[1].set_title('提取的图例信息')
        
        y_pos = 0.9
        for result in results:
            axes[1].text(0.05, y_pos, f"图例区域 {result['region_id']}:", 
                        transform=axes[1].transAxes, fontsize=12, fontweight='bold')
            y_pos -= 0.08
            
            if result['texts'] and result['colors']:
                for text, color in zip(result['texts'], result['colors']):
                    axes[1].add_patch(plt.Rectangle((0.1, y_pos-0.03), 0.1, 0.05, 
                                              facecolor=color, transform=axes[1].transAxes))
                    axes[1].text(0.25, y_pos, f"{text} - {color}", 
                              transform=axes[1].transAxes, fontsize=10, verticalalignment='center')
                    y_pos -= 0.06
            else:
                axes[1].text(0.1, y_pos, "只检测到黑色: #000000", 
                          transform=axes[1].transAxes, fontsize=10)
                y_pos -= 0.06
            y_pos -= 0.02
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")
        plt.show()

def main():
    # 初始化处理管道
    pipeline = LegendDetectionPipeline()
    
    # 图像路径
    image_path = "10.1016-j.jmatprotec.2019.04.020_Fig.18_(d).jpg"  # 替换为您的图像路径
    
    try:
        # 执行完整的处理流程
        results = pipeline.process_image_pipeline(image_path, yolo_threshold=0.3, background_clean_threshold=0.3)
        
        # 输出最终总结
        print("\n" + "=" * 60)
        print("处理完成总结")
        print("=" * 60)
        
        if results:
            for result in results:
                print(f"\n图例区域 {result['region_id']}:")
                print(f"  位置: {result['bbox']}")
                print(f"  置信度: {result['confidence']:.2f}")
                if result['texts'] and result['colors']:
                    for text, color in zip(result['texts'], result['colors']):
                        print(f"  → 颜色: {color}, 文本: {text}")
                else:
                    print("  → 只检测到黑色: #000000")
        else:
            print("未找到任何图例信息")
        
        # 可视化结果
        pipeline.visualize_all_results(image_path, results, "final_result.png")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")

if __name__ == "__main__":
    main()