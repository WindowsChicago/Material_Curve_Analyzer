# # import easyocr
# # import cv2
# # import numpy as np
# # from typing import List, Tuple

# # class TextRecognizer:
# #     def __init__(self, languages=['ch_sim', 'en'], gpu=False):
# #         """
# #         初始化EasyOCR识别器
# #         languages: 支持的语言列表
# #         gpu: 是否使用GPU加速
# #         """
# #         self.reader = easyocr.Reader(languages, gpu=gpu)
    
# #     def preprocess_image(self, image_path: str) -> np.ndarray:
# #         """
# #         图像预处理以提高识别准确率
# #         """
# #         # 读取图像
# #         if isinstance(image_path, str):
# #             image = cv2.imread(image_path)
# #         else:
# #             image = image_path
            
# #         if image is None:
# #             raise ValueError("无法读取图像文件")
            
# #         # 转换为灰度图
# #         if len(image.shape) == 3:
# #             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #         else:
# #             gray = image
            
# #         # 对比度增强
# #         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# #         enhanced = clahe.apply(gray)
        
# #         # 降噪
# #         denoised = cv2.medianBlur(enhanced, 3)
        
# #         return denoised
    
# #     def recognize_text(self, image_path: str, 
# #                       detail: int = 0, 
# #                       preprocess: bool = True,
# #                       min_confidence: float = 0.5) -> str:
# #         """
# #         识别图像中的文本并拼接成字符串
        
# #         Args:
# #             image_path: 图像路径或图像数组
# #             detail: 0-返回字符串, 1-返回详细信息
# #             preprocess: 是否进行图像预处理
# #             min_confidence: 最小置信度阈值
# #         """
# #         # 图像预处理
# #         if preprocess:
# #             processed_image = self.preprocess_image(image_path)
# #         else:
# #             if isinstance(image_path, str):
# #                 processed_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# #             else:
# #                 processed_image = image_path
        
# #         # 文本识别
# #         results = self.reader.readtext(processed_image, detail=detail)
        
# #         if detail == 0:
# #             # 直接返回拼接的文本
# #             return ' '.join(results)
# #         else:
# #             # 过滤低置信度的结果并按位置排序
# #             filtered_results = [
# #                 result for result in results 
# #                 if result[2] >= min_confidence
# #             ]
            
# #             # 按从上到下、从左到右的顺序排序
# #             sorted_results = sorted(filtered_results, 
# #                                   key=lambda x: (x[0][0][1], x[0][0][0]))
            
# #             # 拼接文本
# #             text_parts = [result[1] for result in sorted_results]
# #             return ' '.join(text_parts)
    
# #     def recognize_with_confidence(self, image_path: str, 
# #                                 min_confidence: float = 0.5) -> Tuple[str, List]:
# #         """
# #         识别文本并返回文本字符串及详细信息
# #         """
# #         if isinstance(image_path, str):
# #             image = cv2.imread(image_path)
# #         else:
# #             image = image_path
            
# #         processed_image = self.preprocess_image(image)
        
# #         # 获取详细信息
# #         results = self.reader.readtext(processed_image, detail=1)
        
# #         # 过滤和排序
# #         filtered_results = [
# #             result for result in results 
# #             if result[2] >= min_confidence
# #         ]
        
# #         sorted_results = sorted(filtered_results, 
# #                               key=lambda x: (x[0][0][1], x[0][0][0]))
        
# #         # 拼接文本
# #         text_parts = [result[1] for result in sorted_results]
# #         full_text = ' '.join(text_parts)
        
# #         return full_text, sorted_results

# # # 使用示例
# # def main():
# #     # 初始化识别器
# #     recognizer = TextRecognizer(languages=['ch_sim', 'en'], gpu=False)
    
# #     # 识别图像中的文本
# #     image_path = 'temp_crops/legend_crop_2fyihhxs.jpg'  # 替换为你的图像路径
    
# #     try:
# #         # 方法1: 直接获取拼接的文本
# #         text = recognizer.recognize_text(image_path)
# #         print("识别结果:")
# #         print(text)
        
# #         print("\n" + "="*50 + "\n")
        
# #         # 方法2: 获取文本及详细信息
# #         full_text, details = recognizer.recognize_with_confidence(
# #             image_path, min_confidence=0.6
# #         )
        
# #         print("带置信度过滤的识别结果:")
# #         print(full_text)
        
# #         print("\n详细信息:")
# #         for i, (bbox, text, confidence) in enumerate(details):
# #             print(f"{i+1}. 文本: '{text}', 置信度: {confidence:.3f}")
            
# #     except Exception as e:
# #         print(f"识别过程中出错: {e}")

# # if __name__ == "__main__":
# #     main()
# import easyocr
# import cv2
# import numpy as np
# from typing import List, Tuple
# from collections import Counter

# class TextRecognizer:
#     def __init__(self, languages=['ch_sim', 'en'], gpu=False):
#         """
#         初始化EasyOCR识别器
#         languages: 支持的语言列表
#         gpu: 是否使用GPU加速
#         """
#         self.reader = easyocr.Reader(languages, gpu=gpu)
    
#     def preprocess_image(self, image_path: str) -> np.ndarray:
#         """
#         图像预处理以提高识别准确率
#         """
#         # 读取图像
#         if isinstance(image_path, str):
#             image = cv2.imread(image_path)
#         else:
#             image = image_path
            
#         if image is None:
#             raise ValueError("无法读取图像文件")
            
#         # 转换为灰度图
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image
            
#         # 对比度增强
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(gray)
        
#         # 降噪
#         denoised = cv2.medianBlur(enhanced, 3)
        
#         return denoised
    
#     def recognize_text(self, image_path: str, 
#                       detail: int = 0, 
#                       preprocess: bool = True,
#                       min_confidence: float = 0.5) -> str:
#         """
#         识别图像中的文本并拼接成字符串
        
#         Args:
#             image_path: 图像路径或图像数组
#             detail: 0-返回字符串, 1-返回详细信息
#             preprocess: 是否进行图像预处理
#             min_confidence: 最小置信度阈值
#         """
#         # 图像预处理
#         if preprocess:
#             processed_image = self.preprocess_image(image_path)
#         else:
#             if isinstance(image_path, str):
#                 processed_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#             else:
#                 processed_image = image_path
        
#         # 文本识别
#         results = self.reader.readtext(processed_image, detail=detail)
        
#         if detail == 0:
#             # 直接返回拼接的文本
#             return ' '.join(results)
#         else:
#             # 过滤低置信度的结果并按位置排序
#             filtered_results = [
#                 result for result in results 
#                 if result[2] >= min_confidence
#             ]
            
#             # 按从上到下、从左到右的顺序排序
#             sorted_results = sorted(filtered_results, 
#                                   key=lambda x: (x[0][0][1], x[0][0][0]))
            
#             # 拼接文本
#             text_parts = [result[1] for result in sorted_results]
#             return ' '.join(text_parts)
    
#     def recognize_with_confidence(self, image_path: str, 
#                                 min_confidence: float = 0.5) -> Tuple[str, List]:
#         """
#         识别文本并返回文本字符串及详细信息
#         """
#         if isinstance(image_path, str):
#             image = cv2.imread(image_path)
#         else:
#             image = image_path
            
#         processed_image = self.preprocess_image(image)
        
#         # 获取详细信息
#         results = self.reader.readtext(processed_image, detail=1)
        
#         # 过滤和排序
#         filtered_results = [
#             result for result in results 
#             if result[2] >= min_confidence
#         ]
        
#         sorted_results = sorted(filtered_results, 
#                               key=lambda x: (x[0][0][1], x[0][0][0]))
        
#         # 拼接文本
#         text_parts = [result[1] for result in sorted_results]
#         full_text = ' '.join(text_parts)
        
#         return full_text, sorted_results
    
#     def extract_dominant_color(self, image_path: str, 
#                              color_tolerance: int = 40,
#                              ignore_colors: List[Tuple] = None) -> Tuple[int, int, int]:
#         """
#         提取图像中除黑色和白色之外占比最大的颜色
        
#         Args:
#             image_path: 图像路径或图像数组
#             color_tolerance: 颜色容差，用于颜色聚类
#             ignore_colors: 要忽略的颜色列表，默认为黑色和白色
            
#         Returns:
#             返回RGB格式的主色调，如果只有黑色和白色则返回黑色
#         """
#         # 读取图像
#         if isinstance(image_path, str):
#             image = cv2.imread(image_path)
#         else:
#             image = image_path
            
#         if image is None:
#             raise ValueError("无法读取图像文件")
        
#         # 转换为RGB格式
#         if len(image.shape) == 3:
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         else:
#             # 如果是灰度图，转换为RGB
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
#         # 设置要忽略的颜色（黑色和白色）
#         if ignore_colors is None:
#             ignore_colors = [(0, 0, 0), (255, 255, 255)]  # 黑色和白色
        
#         # 将图像重塑为一维数组
#         pixels = image_rgb.reshape(-1, 3)
        
#         # 过滤掉要忽略的颜色
#         filtered_pixels = []
#         for pixel in pixels:
#             ignore = False
#             for ignore_color in ignore_colors:
#                 # 计算颜色距离
#                 color_distance = np.sqrt(np.sum((pixel - ignore_color) ** 2))
#                 if color_distance <= color_tolerance:
#                     ignore = True
#                     break
#             if not ignore:
#                 filtered_pixels.append(tuple(pixel))
        
#         # 如果没有找到除黑色和白色之外的颜色，返回黑色
#         if not filtered_pixels:
#             return (0, 0, 0)
        
#         # 统计颜色频率
#         color_counter = Counter(filtered_pixels)
        
#         # 找到出现次数最多的颜色
#         dominant_color = color_counter.most_common(1)[0][0]
        
#         return dominant_color

# # 使用示例
# def main():
#     # 初始化识别器
#     recognizer = TextRecognizer(languages=['ch_sim', 'en'], gpu=False)
    
#     # 识别图像中的文本
#     image_path = 'temp_crops/legend_crop_0b11gpt6.jpg'  # 替换为你的图像路径
    
#     try:
#         # 方法1: 直接获取拼接的文本
#         text = recognizer.recognize_text(image_path)
#         print("识别结果:")
#         print(text)
        
#         print("\n" + "="*50 + "\n")
        
#         # 方法2: 获取文本及详细信息
#         full_text, details = recognizer.recognize_with_confidence(
#             image_path, min_confidence=0.6
#         )
        
#         print("带置信度过滤的识别结果:")
#         print(full_text)
        
#         print("\n详细信息:")
#         for i, (bbox, text, confidence) in enumerate(details):
#             print(f"{i+1}. 文本: '{text}', 置信度: {confidence:.3f}")
            
#         print("\n" + "="*50 + "\n")
        
#         # 新增功能：提取主色调
#         dominant_color = recognizer.extract_dominant_color(image_path)
#         print(f"图像主色调 (RGB): {dominant_color}")
        
#         # 可视化主色调
#         color_swatch = np.zeros((100, 100, 3), dtype=np.uint8)
#         color_swatch[:, :] = dominant_color
#         color_swatch_bgr = cv2.cvtColor(color_swatch, cv2.COLOR_RGB2BGR)
        
#         # 保存颜色样本
#         cv2.imwrite('dominant_color.jpg', color_swatch_bgr)
#         print("主色调样本已保存为 'dominant_color.jpg'")
            
#     except Exception as e:
#         print(f"识别过程中出错: {e}")

# if __name__ == "__main__":
#     main()
import easyocr
import cv2
import numpy as np
from typing import List, Tuple
from collections import Counter

class OCRLegendTextExtractor:
    def __init__(self, languages=['ch_sim', 'en'], gpu=True):
        """
        初始化EasyOCR识别器
        languages: 支持的语言列表
        gpu: 是否使用GPU加速
        """
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self.text_data = []
        self.color_data = []
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        图像预处理以提高识别准确率
        """
        # 读取图像
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
            
        if image is None:
            raise ValueError("无法读取图像文件")
            
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 降噪
        denoised = cv2.medianBlur(enhanced, 3)
        
        return denoised
    
    def recognize_text(self, image_path: str, 
                      detail: int = 0, 
                      preprocess: bool = True,
                      min_confidence: float = 0.5) -> str:
        """
        识别图像中的文本并拼接成字符串
        
        Args:
            image_path: 图像路径或图像数组
            detail: 0-返回字符串, 1-返回详细信息
            preprocess: 是否进行图像预处理
            min_confidence: 最小置信度阈值
        """
        # 图像预处理
        if preprocess:
            processed_image = self.preprocess_image(image_path)
        else:
            if isinstance(image_path, str):
                processed_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                processed_image = image_path
        
        # 文本识别
        results = self.reader.readtext(processed_image, detail=detail)
        
        if detail == 0:
            # 直接返回拼接的文本
            return ' '.join(results)
        else:
            # 过滤低置信度的结果并按位置排序
            filtered_results = [
                result for result in results 
                if result[2] >= min_confidence
            ]
            
            # 按从上到下、从左到右的顺序排序
            sorted_results = sorted(filtered_results, 
                                  key=lambda x: (x[0][0][1], x[0][0][0]))
            
            # 拼接文本
            text_parts = [result[1] for result in sorted_results]
            return ' '.join(text_parts)
    
    def recognize_with_confidence(self, image_path: str, 
                                min_confidence: float = 0.5) -> Tuple[str, List]:
        """
        识别文本并返回文本字符串及详细信息
        """
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
            
        processed_image = self.preprocess_image(image)
        
        # 获取详细信息
        results = self.reader.readtext(processed_image, detail=1)
        
        # 过滤和排序
        filtered_results = [
            result for result in results 
            if result[2] >= min_confidence
        ]
        
        sorted_results = sorted(filtered_results, 
                              key=lambda x: (x[0][0][1], x[0][0][0]))
        
        # 拼接文本
        text_parts = [result[1] for result in sorted_results]
        full_text = ' '.join(text_parts)
        
        return full_text, sorted_results
    
    def extract_dominant_color(self, image_path: str, 
                             color_tolerance: int = 40,
                             ignore_colors: List[Tuple] = None) -> Tuple[int, int, int]:
        """
        提取图像中除黑色和白色之外占比最大的颜色
        
        Args:
            image_path: 图像路径或图像数组
            color_tolerance: 颜色容差，用于颜色聚类
            ignore_colors: 要忽略的颜色列表，默认为黑色和白色
            
        Returns:
            返回RGB格式的主色调，如果只有黑色和白色则返回黑色
        """
        # 读取图像
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
            
        if image is None:
            raise ValueError("无法读取图像文件")
        
        # 转换为RGB格式
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # 如果是灰度图，转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 设置要忽略的颜色（黑色和白色）
        if ignore_colors is None:
            ignore_colors = [(0, 0, 0), (255, 255, 255)]  # 黑色和白色
        
        # 将图像重塑为一维数组
        pixels = image_rgb.reshape(-1, 3)
        
        # 过滤掉要忽略的颜色
        filtered_pixels = []
        for pixel in pixels:
            ignore = False
            for ignore_color in ignore_colors:
                # 计算颜色距离
                color_distance = np.sqrt(np.sum((pixel - ignore_color) ** 2))
                if color_distance <= color_tolerance:
                    ignore = True
                    break
            if not ignore:
                filtered_pixels.append(tuple(pixel))
        
        # 如果没有找到除黑色和白色之外的颜色，返回黑色
        if not filtered_pixels:
            return (0, 0, 0)
        
        # 统计颜色频率
        color_counter = Counter(filtered_pixels)
        
        # 找到出现次数最多的颜色
        dominant_color = color_counter.most_common(1)[0][0]
        
        return dominant_color

    def extract_legend(self, image_path: str) -> Tuple[List[str], List[str]]:
        """
        主函数：提取图例信息
        与legend_ocr.py保持相同的API
        
        Args:
            image_path: 图像路径
            
        Returns:
            texts: 文本列表
            colors: 对应的颜色十六进制值列表
        """
        try:
            # 提取主色调
            dominant_color_rgb = self.extract_dominant_color(image_path)
            
            # 识别文本
            text = self.recognize_text(image_path, min_confidence=0.5)
            
            # 转换为十六进制颜色
            color_hex = self.rgb_to_hex(dominant_color_rgb)
            
            # 返回与legend_ocr.py相同的格式
            if text.strip():
                return [text], [color_hex]
            else:
                # 如果没有识别到文本，只返回颜色
                return [], [color_hex]
                
        except Exception as e:
            print(f"提取图例时出错: {e}")
            return [], []

    def rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """
        将RGB颜色转换为十六进制
        
        Args:
            rgb: RGB颜色元组
            
        Returns:
            十六进制颜色字符串
        """
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

    def visualize_results(self, image_path: str, save_path: str = None):
        """
        可视化结果
        与legend_ocr.py保持相同的API
        
        Args:
            image_path: 图像路径
            save_path: 保存路径（可选）
        """
        import matplotlib.pyplot as plt
        plt.rc("font", family='Microsoft YaHei')
        
        # 提取图例信息
        texts, colors = self.extract_legend(image_path)
        
        # 读取并显示原图
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 创建可视化图像
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 显示原图
        axes[0].imshow(img_rgb)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 显示提取结果
        axes[1].axis('off')
        axes[1].set_title('提取的图例')
        
        if texts and colors:
            y_pos = 0.9
            for text, color in zip(texts, colors):
                axes[1].add_patch(plt.Rectangle((0.1, y_pos-0.05), 0.1, 0.1, 
                                          facecolor=color, transform=axes[1].transAxes))
                axes[1].text(0.25, y_pos, f"{text} - {color}", 
                      transform=axes[1].transAxes, fontsize=10, verticalalignment='center')
                y_pos -= 0.12
        else:
            axes[1].text(0.5, 0.5, "未检测到图例", 
                  transform=axes[1].transAxes, ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# 保留原有的TextRecognizer类以保持向后兼容
class TextRecognizer(OCRLegendTextExtractor):
    """为了向后兼容，保留原有类名"""
    pass


# 使用示例
def main():
    # 初始化识别器 - 使用新的类名
    extractor = OCRLegendTextExtractor(languages=['ch_sim', 'en'], gpu=False)
    
    # 识别图像中的文本
    image_path = 'temp_crops/legend_crop_0b11gpt6.jpg'  # 替换为你的图像路径
    
    try:
        # 方法1: 使用新的extract_legend API
        texts, colors = extractor.extract_legend(image_path)
        
        print("EasyOCR图例提取结果:")
        print("=" * 50)
        
        if texts and colors:
            for i, (text, color) in enumerate(zip(texts, colors), 1):
                print(f"{i}. 颜色: {color}, 文本: {text}")
        else:
            print("未检测到图例")
        
        print("\n" + "="*50 + "\n")
        
        # 方法2: 原有的详细识别功能
        full_text, details = extractor.recognize_with_confidence(
            image_path, min_confidence=0.6
        )
        
        print("带置信度过滤的识别结果:")
        print(full_text)
        
        print("\n详细信息:")
        for i, (bbox, text, confidence) in enumerate(details):
            print(f"{i+1}. 文本: '{text}', 置信度: {confidence:.3f}")
            
        print("\n" + "="*50 + "\n")
        
        # 方法3: 单独提取主色调
        dominant_color = extractor.extract_dominant_color(image_path)
        print(f"图像主色调 (RGB): {dominant_color}")
        print(f"图像主色调 (HEX): {extractor.rgb_to_hex(dominant_color)}")
        
        # 可视化结果
        extractor.visualize_results(image_path, "easyocr_legend_result.png")
            
    except Exception as e:
        print(f"识别过程中出错: {e}")

if __name__ == "__main__":
    main()