# import cv2
# import numpy as np
# import pytesseract
# from PIL import Image, ImageEnhance, ImageFilter
# import matplotlib.pyplot as plt
# plt.rc("font", family='Microsoft YaHei')
# from collections import defaultdict
# import re
# import os

# class BlackTextLegendExtractor:
#     def __init__(self):
#         self.text_data = []
#         self.color_data = []
        
#     def preprocess_image(self, image_path):
#         """图像预处理"""
#         # 读取图像
#         img = cv2.imread(image_path)
#         if img is None:
#             raise ValueError(f"无法读取图像: {image_path}")
        
#         # 转换为RGB
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # 保存原始图像用于颜色提取
#         original_img = img_rgb.copy()
        
#         # 调整图像大小以提高处理速度
#         height, width = img_rgb.shape[:2]
#         if max(height, width) > 2000:
#             scale = 2000 / max(height, width)
#             new_width = int(width * scale)
#             new_height = int(height * scale)
#             img_rgb = cv2.resize(img_rgb, (new_width, new_height))
#             original_img = cv2.resize(original_img, (new_width, new_height))
            
#         return img_rgb, original_img
    
#     def enhance_text_for_detection(self, img):
#         """增强文本区域检测"""
#         # 转换为灰度图
#         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
#         # 使用自适应阈值增强文本
#         binary = cv2.adaptiveThreshold(
#             gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#             cv2.THRESH_BINARY_INV, 11, 2
#         )
        
#         # 形态学操作连接文本区域
#         kernel = np.ones((2, 2), np.uint8)
#         binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
#         return binary
    
#     def detect_text_regions(self, img):
#         """检测所有文本区域，不进行颜色过滤"""
#         # 增强文本
#         text_enhanced = self.enhance_text_for_detection(img)
        
#         # 查找轮廓
#         contours, _ = cv2.findContours(text_enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         text_regions = []
        
#         # 计算排除区域的宽度（只排除最左边1/15区域的文字）
#         height, width = img.shape[:2]
#         exclude_width = width // 15
        
#         for contour in contours:
#             area = cv2.contourArea(contour)
#             if area < 20:  # 过滤小区域
#                 continue
                
#             x, y, w, h = cv2.boundingRect(contour)
            
#             # 排除最左边1/15区域的文字
#             if x < exclude_width:
#                 continue

#             text_regions.append({
#                 'bbox': (x, y, w, h),
#                 'center': (x + w//2, y + h//2),
#                 'area': area
#             })
            
#         return text_regions
    
#     def detect_color_regions(self, img):
#         """检测非黑色和白色的颜色区域"""
#         # 转换为HSV颜色空间便于颜色分割
#         hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
#         # 定义黑色和白色的HSV范围
#         # 黑色
#         lower_black = np.array([0, 0, 0])
#         upper_black = np.array([180, 255, 50])
#         mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
#         # 白色
#         lower_white = np.array([0, 0, 200])
#         upper_white = np.array([180, 30, 255])
#         mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
#         # 非黑色和白色的区域
#         mask_non_bw = cv2.bitwise_not(cv2.bitwise_or(mask_black, mask_white))
        
#         # 形态学操作来连接相邻区域
#         kernel = np.ones((5, 5), np.uint8)
#         mask_cleaned = cv2.morphologyEx(mask_non_bw, cv2.MORPH_CLOSE, kernel)
#         mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
        
#         return mask_cleaned, mask_black, mask_white
    
#     def extract_color_blocks(self, img, mask):
#         """提取颜色块及其位置"""
#         # 查找轮廓
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         color_blocks = []
        
#         for contour in contours:
#             # 过滤小区域
#             area = cv2.contourArea(contour)
#             if area < 100:  # 最小面积阈值
#                 continue
                
#             # 获取边界框
#             x, y, w, h = cv2.boundingRect(contour)
            
#             # 计算区域的代表颜色
#             mask_roi = np.zeros(img.shape[:2], np.uint8)
#             cv2.drawContours(mask_roi, [contour], -1, 255, -1)
            
#             mean_color = cv2.mean(img, mask=mask_roi)[:3]
#             mean_color = tuple(map(int, mean_color))
            
#             color_blocks.append({
#                 'bbox': (x, y, w, h),
#                 'color': mean_color,
#                 'area': area,
#                 'center': (x + w//2, y + h//2)
#             })
            
#         return color_blocks
    
#     def prepare_text_for_ocr(self, img, bbox):
#         """为OCR准备文本区域"""
#         x, y, w, h = bbox
        
#         # 扩展区域
#         padding = 5
#         x1 = max(0, x - padding)
#         y1 = max(0, y - padding)
#         x2 = min(img.shape[1], x + w + padding)
#         y2 = min(img.shape[0], y + h + padding)
        
#         # 提取ROI
#         roi = img[y1:y2, x1:x2]
        
#         # 转换为灰度
#         if len(roi.shape) == 3:
#             gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
#         else:
#             gray_roi = roi
        
#         # 使用自适应阈值增强文本
#         binary_roi = cv2.adaptiveThreshold(
#             gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#             cv2.THRESH_BINARY, 11, 2
#         )
        
#         # 转换为PIL图像
#         pil_img = Image.fromarray(binary_roi)
        
#         return pil_img
    
#     def advanced_ocr_for_text(self, img, text_regions):
#         """为文本优化的OCR"""
#         text_data = []
        
#         for i, region in enumerate(text_regions):
#             # 准备文本区域
#             pil_img = self.prepare_text_for_ocr(img, region['bbox'])
            
#             # 尝试多种OCR配置
#             ocr_configs = [
#                 '--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,- ',
#                 '--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,- ',
#                 '--psm 7',
#                 '--psm 8',
#                 '--psm 6'
#             ]
            
#             best_text = ""
#             best_confidence = 0
            
#             for config in ocr_configs:
#                 try:
#                     # 使用pytesseract进行OCR，获取详细数据
#                     ocr_data = pytesseract.image_to_data(
#                         pil_img, 
#                         config=config,
#                         output_type=pytesseract.Output.DICT
#                     )
                    
#                     # 计算平均置信度
#                     confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
#                     if confidences:
#                         avg_confidence = sum(confidences) / len(confidences)
                        
#                         # 提取文本
#                         text = ' '.join([
#                             ocr_data['text'][i] 
#                             for i in range(len(ocr_data['text'])) 
#                             if int(ocr_data['conf'][i]) > 30
#                         ]).strip()
                        
#                         if text and avg_confidence > best_confidence:
#                             best_text = text
#                             best_confidence = avg_confidence
                            
#                 except Exception as e:
#                     continue
            
#             # 如果主要方法失败，尝试简单OCR
#             if not best_text:
#                 try:
#                     simple_text = pytesseract.image_to_string(pil_img, lang='eng')
#                     best_text = simple_text.strip()
#                 except:
#                     pass
            

#             if best_text and len(best_text) > 0:  # 只保留非空且长度大于1的文本
#                 text_data.append({
#                     'bbox': region['bbox'],
#                     'text': best_text,
#                     'center': region['center'],
#                     'confidence': best_confidence
#                 })
                
#         return text_data
        
#     def merge_close_text_regions(self, text_regions, max_gap=10):
#         """合并靠近的文本区域"""
#         if not text_regions:
#             return text_regions
            
#         # 按x坐标排序
#         text_regions.sort(key=lambda x: x['bbox'][0])
        
#         merged_regions = []
#         current_region = text_regions[0].copy()
        
#         for i in range(1, len(text_regions)):
#             current_bbox = current_region['bbox']
#             next_bbox = text_regions[i]['bbox']
            
#             # 检查是否应该合并
#             current_right = current_bbox[0] + current_bbox[2]
#             next_left = next_bbox[0]
            
#             # 如果两个区域很近，合并它们
#             if next_left - current_right <= max_gap:
#                 # 合并边界框
#                 new_x = min(current_bbox[0], next_bbox[0])
#                 new_y = min(current_bbox[1], next_bbox[1])
#                 new_right = max(current_bbox[0] + current_bbox[2], next_bbox[0] + next_bbox[2])
#                 new_bottom = max(current_bbox[1] + current_bbox[3], next_bbox[1] + next_bbox[3])
#                 new_w = new_right - new_x
#                 new_h = new_bottom - new_y
                
#                 current_region['bbox'] = (new_x, new_y, new_w, new_h)
#                 current_region['center'] = (new_x + new_w//2, new_y + new_h//2)
                
#                 # 合并文本（在实际OCR后处理中处理）
#             else:
#                 merged_regions.append(current_region)
#                 current_region = text_regions[i].copy()
        
#         merged_regions.append(current_region)
#         return merged_regions
    
#     def merge_all_text_regions(self, text_data):
#         """合并所有文本区域为一个字符串"""
#         if not text_data:
#             return ""
        
#         # 按y坐标排序，假设文本是垂直排列的
#         text_data_sorted = sorted(text_data, key=lambda x: x['center'][1])
        
#         # 拼接所有文本
#         combined_text = ' '.join([item['text'] for item in text_data_sorted])
        
#         return combined_text
    
#     def find_leftmost_color_block(self, color_blocks):
#         """找到最左侧的颜色块"""
#         if not color_blocks:
#             return None
        
#         # 按x坐标排序，找到最左侧的颜色块
#         leftmost_block = min(color_blocks, key=lambda x: x['bbox'][0])
#         return leftmost_block
    
#     def match_single_legend(self, color_block, combined_text):
#         """匹配单个图例的颜色和文本"""
#         if color_block and combined_text:
#             return {
#                 'color': color_block['color'],
#                 'text': combined_text,
#                 'color_bbox': color_block['bbox'],
#                 'confidence': 0  # 由于是合并的文本，没有统一的置信度
#             }
#         return None
    
#     def rgb_to_hex(self, rgb):
#         """将RGB颜色转换为十六进制"""
#         return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
    
#     def extract_legend(self, image_path):
#         """主函数：提取图例信息"""
#         # 预处理图像
#         img, original_img = self.preprocess_image(image_path)
        
#         # 检测颜色区域
#         color_mask, black_mask, white_mask = self.detect_color_regions(original_img)
        
#         # 提取颜色块
#         color_blocks = self.extract_color_blocks(original_img, color_mask)
        
#         # 检查是否有非黑色/白色颜色
#         if not color_blocks:
#             # 检查是否有黑色区域
#             black_blocks = self.extract_color_blocks(original_img, black_mask)
#             if black_blocks:
#                 # 返回黑色
#                 return ["黑色"], ["#000000"]
#             else:
#                 return [], []
        
#         # 检测所有文本区域（不进行颜色过滤，已排除最左边1/15区域）
#         text_regions = self.detect_text_regions(img)
        
#         # 合并靠近的文本区域
#         merged_text_regions = self.merge_close_text_regions(text_regions)
        
#         # OCR识别
#         text_data = self.advanced_ocr_for_text(img, merged_text_regions)
        
#         # 合并所有文本区域为一个字符串
#         combined_text = self.merge_all_text_regions(text_data)
        
#         # 找到最左侧的颜色块
#         leftmost_color_block = self.find_leftmost_color_block(color_blocks)
        
#         # 匹配单个图例
#         match = self.match_single_legend(leftmost_color_block, combined_text)
        
#         # 提取结果
#         colors = []
#         texts = []
        
#         if match:
#             colors.append(self.rgb_to_hex(match['color']))
#             texts.append(match['text'])
            
#         return texts, colors
    
#     def visualize_results(self, image_path, save_path=None):
#         """可视化结果"""
#         img, original_img = self.preprocess_image(image_path)
        
#         # 检测文本区域
#         text_regions = self.detect_text_regions(img)
        
#         # 创建可视化图像
#         fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
#         # 显示原图
#         axes[0, 0].imshow(img)
#         axes[0, 0].set_title('原始图像')
#         axes[0, 0].axis('off')
        
#         # 显示文本检测结果
#         text_detection_img = img.copy()
#         for region in text_regions:
#             x, y, w, h = region['bbox']
#             cv2.rectangle(text_detection_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         axes[0, 1].imshow(text_detection_img)
#         axes[0, 1].set_title('文本检测结果')
#         axes[0, 1].axis('off')
        
#         # 显示颜色检测结果
#         color_mask, _, _ = self.detect_color_regions(original_img)
#         axes[1, 0].imshow(color_mask, cmap='gray')
#         axes[1, 0].set_title('颜色区域检测')
#         axes[1, 0].axis('off')
        
#         # 显示最终提取结果
#         texts, colors = self.extract_legend(image_path)
#         axes[1, 1].axis('off')
#         axes[1, 1].set_title('提取的图例')
        
#         if texts and colors:
#             y_pos = 0.9
#             for text, color in zip(texts, colors):
#                 axes[1, 1].add_patch(plt.Rectangle((0.1, y_pos-0.05), 0.1, 0.1, 
#                                               facecolor=color, transform=axes[1, 1].transAxes))
#                 axes[1, 1].text(0.25, y_pos, f"{text} - {color}", 
#                           transform=axes[1, 1].transAxes, fontsize=10, verticalalignment='center')
#                 y_pos -= 0.12
#         else:
#             axes[1, 1].text(0.5, 0.5, "只检测到黑色", 
#                       transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=14)
        
#         plt.tight_layout()
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.show()

# # 使用示例
# def main():
#     extractor = BlackTextLegendExtractor()
    
#     # 替换为您的图像路径
#     image_path = "temp_crops/legend_crop_2wmsoa2i.jpg"  # 请替换为实际图像路径
    
#     try:
#         # 提取图例信息
#         texts, colors = extractor.extract_legend(image_path)
        
#         print("文本图例提取结果:")
#         print("=" * 50)
        
#         if texts and colors:
#             for i, (text, color) in enumerate(zip(texts, colors), 1):
#                 print(f"{i}. 颜色: {color}, 文本: {text}")
#         else:
#             print("只检测到黑色")
#             print("黑色: #000000")
        
#         # 可视化结果
#         extractor.visualize_results(image_path, "text_result.png")
        
#     except Exception as e:
#         print(f"处理错误: {e}")

# if __name__ == "__main__":
#     main()
import cv2
import numpy as np
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
plt.rc("font", family='Microsoft YaHei')
from collections import defaultdict
import re
import os

class OCRLegendTextExtractor:
    def __init__(self):
        self.text_data = []
        self.color_data = []
        
    def preprocess_image(self, image_path):
        """图像预处理"""
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 保存原始图像用于颜色提取
        original_img = img_rgb.copy()
        
        # 调整图像大小以提高处理速度
        height, width = img_rgb.shape[:2]
        if max(height, width) > 2000:
            scale = 2000 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_rgb = cv2.resize(img_rgb, (new_width, new_height))
            original_img = cv2.resize(original_img, (new_width, new_height))
            
        return img_rgb, original_img
    
    def detect_text_regions(self, img):
        """检测所有文本区域，不进行颜色过滤"""
        # 直接使用灰度图查找轮廓
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 使用简单的阈值处理
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        
        # 计算排除区域的宽度（只排除最左边1/15区域的文字）
        height, width = img.shape[:2]
        exclude_width = width // 15
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20:  # 过滤小区域
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # 排除最左边1/15区域的文字
            if x < exclude_width:
                continue

            text_regions.append({
                'bbox': (x, y, w, h),
                'center': (x + w//2, y + h//2),
                'area': area
            })
            
        return text_regions
    
    def detect_color_regions(self, img):
        """检测非黑色和白色的颜色区域"""
        # 转换为HSV颜色空间便于颜色分割
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # 定义黑色和白色的HSV范围
        # 黑色
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        # 白色
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # 非黑色和白色的区域
        mask_non_bw = cv2.bitwise_not(cv2.bitwise_or(mask_black, mask_white))
        
        # 形态学操作来连接相邻区域
        kernel = np.ones((5, 5), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_non_bw, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
        
        return mask_cleaned, mask_black, mask_white
    
    def extract_color_blocks(self, img, mask):
        """提取颜色块及其位置"""
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        color_blocks = []
        
        for contour in contours:
            # 过滤小区域
            area = cv2.contourArea(contour)
            if area < 100:  # 最小面积阈值
                continue
                
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算区域的代表颜色
            mask_roi = np.zeros(img.shape[:2], np.uint8)
            cv2.drawContours(mask_roi, [contour], -1, 255, -1)
            
            mean_color = cv2.mean(img, mask=mask_roi)[:3]
            mean_color = tuple(map(int, mean_color))
            
            color_blocks.append({
                'bbox': (x, y, w, h),
                'color': mean_color,
                'area': area,
                'center': (x + w//2, y + h//2)
            })
            
        return color_blocks
    
    def prepare_text_for_ocr(self, img, bbox):
        """为OCR准备文本区域 - 直接使用原图"""
        x, y, w, h = bbox
        
        # 扩展区域
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        # 提取ROI - 直接使用原图，不进行任何增强处理
        roi = img[y1:y2, x1:x2]
        
        # 转换为PIL图像
        pil_img = Image.fromarray(roi)
        
        return pil_img
    
    def advanced_ocr_for_text(self, img, text_regions):
        """为文本优化的OCR - 直接使用原图"""
        text_data = []
        
        for i, region in enumerate(text_regions):
            # 准备文本区域 - 直接使用原图
            pil_img = self.prepare_text_for_ocr(img, region['bbox'])
            
            # 尝试多种OCR配置
            ocr_configs = [
                '--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,- ',
                '--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,- ',
                '--psm 7',
                '--psm 8',
                '--psm 6'
            ]
            
            best_text = ""
            best_confidence = 0
            
            for config in ocr_configs:
                try:
                    # 使用pytesseract进行OCR，获取详细数据
                    ocr_data = pytesseract.image_to_data(
                        pil_img, 
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # 计算平均置信度
                    confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        
                        # 提取文本
                        text = ' '.join([
                            ocr_data['text'][i] 
                            for i in range(len(ocr_data['text'])) 
                            if int(ocr_data['conf'][i]) > 30
                        ]).strip()
                        
                        if text and avg_confidence > best_confidence:
                            best_text = text
                            best_confidence = avg_confidence
                            
                except Exception as e:
                    continue
            
            # 如果主要方法失败，尝试简单OCR
            if not best_text:
                try:
                    simple_text = pytesseract.image_to_string(pil_img, lang='eng')
                    best_text = simple_text.strip()
                except:
                    pass
            

            if best_text and len(best_text) > 0:  # 只保留非空且长度大于1的文本
                text_data.append({
                    'bbox': region['bbox'],
                    'text': best_text,
                    'center': region['center'],
                    'confidence': best_confidence
                })
                
        return text_data
        
    def merge_close_text_regions(self, text_regions, max_gap=10):
        """合并靠近的文本区域"""
        if not text_regions:
            return text_regions
            
        # 按x坐标排序
        text_regions.sort(key=lambda x: x['bbox'][0])
        
        merged_regions = []
        current_region = text_regions[0].copy()
        
        for i in range(1, len(text_regions)):
            current_bbox = current_region['bbox']
            next_bbox = text_regions[i]['bbox']
            
            # 检查是否应该合并
            current_right = current_bbox[0] + current_bbox[2]
            next_left = next_bbox[0]
            
            # 如果两个区域很近，合并它们
            if next_left - current_right <= max_gap:
                # 合并边界框
                new_x = min(current_bbox[0], next_bbox[0])
                new_y = min(current_bbox[1], next_bbox[1])
                new_right = max(current_bbox[0] + current_bbox[2], next_bbox[0] + next_bbox[2])
                new_bottom = max(current_bbox[1] + current_bbox[3], next_bbox[1] + next_bbox[3])
                new_w = new_right - new_x
                new_h = new_bottom - new_y
                
                current_region['bbox'] = (new_x, new_y, new_w, new_h)
                current_region['center'] = (new_x + new_w//2, new_y + new_h//2)
                
                # 合并文本（在实际OCR后处理中处理）
            else:
                merged_regions.append(current_region)
                current_region = text_regions[i].copy()
        
        merged_regions.append(current_region)
        return merged_regions
    
    def merge_all_text_regions(self, text_data):
        """合并所有文本区域为一个字符串"""
        if not text_data:
            return ""
        
        # 按y坐标排序，假设文本是垂直排列的
        text_data_sorted = sorted(text_data, key=lambda x: x['center'][1])
        
        # 拼接所有文本
        combined_text = ' '.join([item['text'] for item in text_data_sorted])
        
        return combined_text
    
    def find_leftmost_color_block(self, color_blocks):
        """找到最左侧的颜色块"""
        if not color_blocks:
            return None
        
        # 按x坐标排序，找到最左侧的颜色块
        leftmost_block = min(color_blocks, key=lambda x: x['bbox'][0])
        return leftmost_block
    
    def match_single_legend(self, color_block, combined_text):
        """匹配单个图例的颜色和文本"""
        if color_block and combined_text:
            return {
                'color': color_block['color'],
                'text': combined_text,
                'color_bbox': color_block['bbox'],
                'confidence': 0  # 由于是合并的文本，没有统一的置信度
            }
        return None
    
    def rgb_to_hex(self, rgb):
        """将RGB颜色转换为十六进制"""
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
    
    def extract_legend(self, image_path):
        """主函数：提取图例信息"""
        # 预处理图像
        img, original_img = self.preprocess_image(image_path)
        
        # 检测颜色区域
        color_mask, black_mask, white_mask = self.detect_color_regions(original_img)
        
        # 提取颜色块
        color_blocks = self.extract_color_blocks(original_img, color_mask)
        
        # 检查是否有非黑色/白色颜色
        if not color_blocks:
            # 检查是否有黑色区域
            black_blocks = self.extract_color_blocks(original_img, black_mask)
            if black_blocks:
                # 返回黑色
                return ["黑色"], ["#000000"]
            else:
                return [], []
        
        # 检测所有文本区域（不进行颜色过滤，已排除最左边1/15区域）
        text_regions = self.detect_text_regions(img)
        
        # 合并靠近的文本区域
        merged_text_regions = self.merge_close_text_regions(text_regions)
        
        # OCR识别 - 直接使用原图
        text_data = self.advanced_ocr_for_text(img, merged_text_regions)
        
        # 合并所有文本区域为一个字符串
        combined_text = self.merge_all_text_regions(text_data)
        
        # 找到最左侧的颜色块
        leftmost_color_block = self.find_leftmost_color_block(color_blocks)
        
        # 匹配单个图例
        match = self.match_single_legend(leftmost_color_block, combined_text)
        
        # 提取结果
        colors = []
        texts = []
        
        if match:
            colors.append(self.rgb_to_hex(match['color']))
            texts.append(match['text'])
            
        return texts, colors
    
    def visualize_results(self, image_path, save_path=None):
        """可视化结果"""
        img, original_img = self.preprocess_image(image_path)
        
        # 检测文本区域
        text_regions = self.detect_text_regions(img)
        
        # 创建可视化图像
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 显示原图
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')
        
        # 显示文本检测结果
        text_detection_img = img.copy()
        for region in text_regions:
            x, y, w, h = region['bbox']
            cv2.rectangle(text_detection_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        axes[0, 1].imshow(text_detection_img)
        axes[0, 1].set_title('文本检测结果')
        axes[0, 1].axis('off')
        
        # 显示颜色检测结果
        color_mask, _, _ = self.detect_color_regions(original_img)
        axes[1, 0].imshow(color_mask, cmap='gray')
        axes[1, 0].set_title('颜色区域检测')
        axes[1, 0].axis('off')
        
        # 显示最终提取结果
        texts, colors = self.extract_legend(image_path)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('提取的图例')
        
        if texts and colors:
            y_pos = 0.9
            for text, color in zip(texts, colors):
                axes[1, 1].add_patch(plt.Rectangle((0.1, y_pos-0.05), 0.1, 0.1, 
                                              facecolor=color, transform=axes[1, 1].transAxes))
                axes[1, 1].text(0.25, y_pos, f"{text} - {color}", 
                          transform=axes[1, 1].transAxes, fontsize=10, verticalalignment='center')
                y_pos -= 0.12
        else:
            axes[1, 1].text(0.5, 0.5, "只检测到黑色", 
                      transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# 使用示例
def main():
    extractor = OCRLegendTextExtractor()
    
    # 替换为您的图像路径
    image_path = "temp_crops/legend_crop_2fyihhxs.jpg"  # 请替换为实际图像路径
    
    try:
        # 提取图例信息
        texts, colors = extractor.extract_legend(image_path)
        
        print("文本图例提取结果:")
        print("=" * 50)
        
        if texts and colors:
            for i, (text, color) in enumerate(zip(texts, colors), 1):
                print(f"{i}. 颜色: {color}, 文本: {text}")
        else:
            print("只检测到黑色")
            print("黑色: #000000")
        
        # 可视化结果
        extractor.visualize_results(image_path, "text_result.png")
        
    except Exception as e:
        print(f"处理错误: {e}")

if __name__ == "__main__":
    main()