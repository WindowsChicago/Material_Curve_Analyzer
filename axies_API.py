import cv2
import numpy as np
import matplotlib.pyplot as plt
#plt.rc("font", family='Microsoft YaHei')
plt.rcParams['font.family'].insert(0,'WenQuanYi Micro Hei')
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
from scipy import signal
import easyocr
import re
from PIL import Image
import io
import os

# 初始化EasyOCR阅读器（支持英文和中文）
reader = easyocr.Reader(['ch_sim', 'en'])

# 导入X轴和Y轴处理函数
from axies_X import (
    read_image_pil,
    extract_x_coord_bidirectional,
    extract_text_num_x_axis_region,
    filter_merge_texts as filter_and_merge_texts_x,
    extract_x_axis_title,
    calculate_pixels_per_value,
    visualize_bidirectional_results_with_ocr as visualize_x_results
)

from axies_Y import (
    extract_y_axis_coordinates_bidirectional_auto,
    extract_text_and_numbers_from_y_axis_region,
    filter_and_merge_texts as filter_and_merge_texts_y,
    extract_y_axis_title,
    extract_rotated_y_axis_title,
    calculate_pixels_per_value_y,
    visualize_y_bidirectional_results_with_ocr as visualize_y_results
)

class AxesExtractor:
    """
    坐标轴提取器类，整合X轴和Y轴的处理功能
    """
    
    def __init__(self, debug=False):
        """
        初始化坐标轴提取器
        
        参数:
            debug: 是否启用调试模式（可视化结果）
        """
        self.debug = debug
        self.x_axis_info = None
        self.y_axis_info = None
        self.x_raw_data = None  # 缓存X轴原始数据
        self.y_raw_data = None  # 缓存Y轴原始数据
    
    def extract_axes_info(self, image_path):
        """
        提取图像中坐标轴的完整信息
        
        参数:
            image_path: 图像文件路径
            
        返回:
            axes_info: 包含坐标轴信息的字典
        """
        try:
            # 提取X轴信息
            x_info, x_raw_data = self._extract_x_axis_info(image_path)
            
            # 提取Y轴信息
            y_info, y_raw_data = self._extract_y_axis_info(image_path)
            
            # 缓存原始数据用于可视化
            self.x_raw_data = x_raw_data
            self.y_raw_data = y_raw_data
            
            # 整合结果
            axes_info = {
                'x_axis': x_info,
                'y_axis': y_info,
                'success': True,
                'message': '坐标轴提取成功'
            }
            
            # 保存信息供调试使用
            self.x_axis_info = x_info
            self.y_axis_info = y_info
            
            return axes_info
            
        except Exception as e:
            return {
                'x_axis': None,
                'y_axis': None,
                'success': False,
                'message': f'坐标轴提取失败: {str(e)}'
            }
    
    def _extract_x_axis_info(self, image_path):
        """
        提取X轴信息
        
        参数:
            image_path: 图像文件路径
            
        返回:
            x_info: X轴信息字典
            x_raw_data: X轴原始数据（用于可视化）
        """
        try:
            # 提取X轴坐标
            left_edge, right_edge, long_ticks, above_ticks, below_ticks = extract_x_coord_bidirectional(image_path)
            
            # 使用EasyOCR识别X轴区域的文本和数字
            numbers, texts = extract_text_num_x_axis_region(image_path)
            
            # 过滤和合并文本
            merged_texts = filter_and_merge_texts_x(texts, confidence_threshold=0.5, y_threshold=10)
            
            # 提取x轴标题
            x_axis_title, other_texts = extract_x_axis_title(merged_texts)
            
            # 计算X方向每像素对应的刻度值
            pixels_per_value, matched_pairs = calculate_pixels_per_value(long_ticks, numbers)
            
            # 构建X轴信息字典
            x_info = {
                'title': x_axis_title['content'] if x_axis_title else None,
                'title_position': x_axis_title['position'] if x_axis_title else None,
                'title_confidence': x_axis_title['confidence'] if x_axis_title else None,
                'left_limit': float(left_edge),
                'right_limit': float(right_edge),
                'pixels_per_value': float(pixels_per_value) if pixels_per_value else None,
                'long_ticks': [float(tick) for tick in long_ticks],
                'above_ticks': [float(tick) for tick in above_ticks],
                'below_ticks': [float(tick) for tick in below_ticks],
                'numbers': [
                    {
                        'value': float(num['value']),
                        'position': (float(num['position'][0]), float(num['position'][1])),
                        'confidence': float(num['confidence'])
                    } for num in numbers
                ],
                'matched_pairs': [
                    {
                        'tick_x': float(pair['tick_x']),
                        'num_value': float(pair['num_value']),
                        'distance': float(pair['distance'])
                    } for pair in matched_pairs
                ] if matched_pairs else [],
                'success': True
            }
            
            # 保存原始数据用于可视化
            x_raw_data = {
                'left_edge': left_edge,
                'right_edge': right_edge,
                'long_ticks': long_ticks,
                'above_ticks': above_ticks,
                'below_ticks': below_ticks,
                'numbers': numbers,
                'texts': texts,
                'merged_texts': merged_texts,
                'x_axis_title': x_axis_title,
                'other_texts': other_texts,
                'pixels_per_value': pixels_per_value,
                'matched_pairs': matched_pairs
            }
            
            return x_info, x_raw_data
            
        except Exception as e:
            error_info = {
                'title': None,
                'left_limit': None,
                'right_limit': None,
                'pixels_per_value': None,
                'success': False,
                'error': str(e)
            }
            return error_info, None
    
    def _extract_y_axis_info(self, image_path):
        """
        提取Y轴信息
        
        参数:
            image_path: 图像文件路径
            
        返回:
            y_info: Y轴信息字典
            y_raw_data: Y轴原始数据（用于可视化）
        """
        try:
            # 自动检测Y轴位置
            bottom_edge, top_edge, long_ticks, left_ticks, right_ticks, y_axis_position = extract_y_axis_coordinates_bidirectional_auto(image_path)
            
            # 使用EasyOCR识别Y轴区域的文本和数字
            numbers, texts = extract_text_and_numbers_from_y_axis_region(image_path, y_axis_position)
            
            # 过滤和合并文本
            merged_texts = filter_and_merge_texts_y(texts, confidence_threshold=0.5, x_threshold=10)
            
            # 提取y轴标题（传统方法）
            y_axis_title, other_texts = extract_y_axis_title(merged_texts, y_axis_position)
            
            # 提取旋转的y轴标题（新方法）
            rotated_y_axis_title = extract_rotated_y_axis_title(image_path, y_axis_position, numbers, texts)
            
            # 计算Y方向每像素对应的刻度值
            pixels_per_value, matched_pairs = calculate_pixels_per_value_y(long_ticks, numbers)
            
            # 优先使用旋转识别到的标题，如果没有则使用传统方法识别的标题
            final_title = rotated_y_axis_title if rotated_y_axis_title else y_axis_title
            
            # 构建Y轴信息字典
            y_info = {
                'title': final_title['content'] if final_title else None,
                'title_position': final_title['position'] if final_title else None,
                'title_confidence': final_title['confidence'] if final_title else None,
                'title_method': 'rotated' if rotated_y_axis_title else 'traditional' if y_axis_title else None,
                'axis_position': y_axis_position,
                'bottom_limit': float(bottom_edge),
                'top_limit': float(top_edge),
                'pixels_per_value': float(pixels_per_value) if pixels_per_value else None,
                'long_ticks': [float(tick) for tick in long_ticks],
                'left_ticks': [float(tick) for tick in left_ticks],
                'right_ticks': [float(tick) for tick in right_ticks],
                'numbers': [
                    {
                        'value': float(num['value']),
                        'position': (float(num['position'][0]), float(num['position'][1])),
                        'confidence': float(num['confidence'])
                    } for num in numbers
                ],
                'matched_pairs': [
                    {
                        'tick_y': float(pair['tick_y']),
                        'num_value': float(pair['num_value']),
                        'distance': float(pair['distance'])
                    } for pair in matched_pairs
                ] if matched_pairs else [],
                'success': True
            }
            
            # 保存原始数据用于可视化
            y_raw_data = {
                'bottom_edge': bottom_edge,
                'top_edge': top_edge,
                'long_ticks': long_ticks,
                'left_ticks': left_ticks,
                'right_ticks': right_ticks,
                'y_axis_position': y_axis_position,
                'numbers': numbers,
                'texts': texts,
                'merged_texts': merged_texts,
                'y_axis_title': y_axis_title,
                'other_texts': other_texts,
                'rotated_y_axis_title': rotated_y_axis_title,
                'pixels_per_value': pixels_per_value,
                'matched_pairs': matched_pairs
            }
            
            return y_info, y_raw_data
            
        except Exception as e:
            error_info = {
                'title': None,
                'axis_position': None,
                'bottom_limit': None,
                'top_limit': None,
                'pixels_per_value': None,
                'success': False,
                'error': str(e)
            }
            return error_info, None
    
    def get_formatted_results(self, axes_info):
        """
        获取格式化的结果（便于阅读和显示）
        
        参数:
            axes_info: extract_axes_info返回的结果
            
        返回:
            formatted: 格式化的结果字典
        """
        if not axes_info['success']:
            return {
                'success': False,
                'message': axes_info['message']
            }
        
        x_info = axes_info['x_axis']
        y_info = axes_info['y_axis']
        
        formatted = {
            'success': True,
            'x_axis': {
                'title': x_info['title'] if x_info['success'] else '未识别',
                'limits': {
                    'left': x_info['left_limit'] if x_info['success'] else None,
                    'right': x_info['right_limit'] if x_info['success'] else None
                },
                'scale': {
                    'pixels_per_value': x_info['pixels_per_value'] if x_info['success'] else None,
                    'units_per_pixel': 1.0 / x_info['pixels_per_value'] if x_info['success'] and x_info['pixels_per_value'] else None
                }
            },
            'y_axis': {
                'title': y_info['title'] if y_info['success'] else '未识别',
                'position': y_info['axis_position'] if y_info['success'] else None,
                'limits': {
                    'bottom': y_info['bottom_limit'] if y_info['success'] else None,
                    'top': y_info['top_limit'] if y_info['success'] else None
                },
                'scale': {
                    'pixels_per_value': y_info['pixels_per_value'] if y_info['success'] else None,
                    'units_per_pixel': 1.0 / y_info['pixels_per_value'] if y_info['success'] and y_info['pixels_per_value'] else None
                }
            },
            'coordinate_system': {
                'x_range': x_info['right_limit'] - x_info['left_limit'] if x_info['success'] else None,
                'y_range': y_info['top_limit'] - y_info['bottom_limit'] if y_info['success'] else None
            }
        }
        
        return formatted
    
    def visualize_results(self, image_path, axes_info=None):
        """
        可视化提取结果（仅在调试模式下可用）
        
        参数:
            image_path: 图像文件路径
            axes_info: 可选的坐标轴信息，如果为None则使用内部存储的信息
        """
        if not self.debug:
            print("可视化功能仅在调试模式下可用")
            return
        
        if axes_info is None:
            if not self.x_raw_data or not self.y_raw_data:
                print("没有可用的坐标轴信息进行可视化，请先调用 extract_axes_info()")
                return
        
        print("\n正在生成可视化结果...")
        
        try:
            # 可视化X轴结果
            if self.x_raw_data:
                print("显示X轴可视化结果...")
                visualize_x_results(
                    image_path, 
                    self.x_raw_data['left_edge'], 
                    self.x_raw_data['right_edge'], 
                    self.x_raw_data['long_ticks'],
                    self.x_raw_data['above_ticks'], 
                    self.x_raw_data['below_ticks'], 
                    self.x_raw_data['numbers'], 
                    self.x_raw_data['texts'],
                    self.x_raw_data['merged_texts'], 
                    self.x_raw_data['x_axis_title'], 
                    self.x_raw_data['other_texts'],
                    self.x_raw_data['pixels_per_value'], 
                    self.x_raw_data['matched_pairs']
                )
            else:
                print("X轴原始数据不可用，跳过X轴可视化")
            
            # 可视化Y轴结果
            if self.y_raw_data:
                print("显示Y轴可视化结果...")
                visualize_y_results(
                    image_path, 
                    self.y_raw_data['bottom_edge'], 
                    self.y_raw_data['top_edge'], 
                    self.y_raw_data['long_ticks'],
                    self.y_raw_data['left_ticks'], 
                    self.y_raw_data['right_ticks'], 
                    self.y_raw_data['y_axis_position'],
                    self.y_raw_data['numbers'], 
                    self.y_raw_data['texts'], 
                    self.y_raw_data['merged_texts'],
                    self.y_raw_data['y_axis_title'], 
                    self.y_raw_data['other_texts'],
                    self.y_raw_data['rotated_y_axis_title'], 
                    self.y_raw_data['pixels_per_value'], 
                    self.y_raw_data['matched_pairs']
                )
            else:
                print("Y轴原始数据不可用，跳过Y轴可视化")
                
        except Exception as e:
            print(f"可视化过程中出错: {e}")
    
    def save_results(self, axes_info, output_path):
        """
        保存提取结果到文件
        
        参数:
            axes_info: extract_axes_info返回的结果
            output_path: 输出文件路径
        """
        import json
        
        formatted = self.get_formatted_results(axes_info)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {output_path}")

# 简化接口函数
def extract_axes_info(image_path, debug=False):
    """
    简化接口：提取图像中坐标轴的完整信息
    
    参数:
        image_path: 图像文件路径
        debug: 是否启用调试模式
        
    返回:
        axes_info: 包含坐标轴信息的字典
    """
    extractor = AxesExtractor(debug=debug)
    return extractor.extract_axes_info(image_path)

def get_axes_parameters(image_path, debug=False):
    """
    简化接口：获取坐标轴的核心参数
    
    参数:
        image_path: 图像文件路径
        debug: 是否启用调试模式
        
    返回:
        parameters: 包含核心参数的字典
    """
    extractor = AxesExtractor(debug=debug)
    axes_info = extractor.extract_axes_info(image_path)
    return extractor.get_formatted_results(axes_info)

def extract_and_visualize(image_path, debug=True):
    """
    简化接口：提取坐标轴信息并显示可视化结果
    
    参数:
        image_path: 图像文件路径
        debug: 是否启用调试模式
        
    返回:
        axes_info: 包含坐标轴信息的字典
    """
    extractor = AxesExtractor(debug=debug)
    axes_info = extractor.extract_axes_info(image_path)
    
    if axes_info['success']:
        # 显示可视化结果
        extractor.visualize_results(image_path, axes_info)
    else:
        print(f"提取失败: {axes_info['message']}")
    
    return axes_info

# 使用示例
if __name__ == "__main__":
    # 测试代码
    image_path = "figs/22.tif"  # 替换为您的图像路径
    
    # 创建提取器实例
    extractor = AxesExtractor(debug=True)
    
    # 提取坐标轴信息
    axes_info = extractor.extract_axes_info(image_path)
    
    if axes_info['success']:
        # 获取格式化结果
        formatted = extractor.get_formatted_results(axes_info)
        
        # 打印核心信息
        print("=" * 50)
        print("坐标轴提取结果")
        print("=" * 50)
        
        print("\nX轴信息:")
        print(f"  标题: {formatted['x_axis']['title']}")
        print(f"  左极限: {formatted['x_axis']['limits']['left']:.2f}")
        print(f"  右极限: {formatted['x_axis']['limits']['right']:.2f}")
        if formatted['x_axis']['scale']['pixels_per_value']:
            print(f"  X方向刻度: {formatted['x_axis']['scale']['pixels_per_value']:.6f} 单位/像素")
        if formatted['x_axis']['scale']['units_per_pixel']:
            print(f"  X方向刻度: {formatted['x_axis']['scale']['units_per_pixel']:.6f} 像素/单位")
        
        print("\nY轴信息:")
        print(f"  标题: {formatted['y_axis']['title']}")
        print(f"  位置: {formatted['y_axis']['position']}")
        print(f"  下极限: {formatted['y_axis']['limits']['bottom']:.2f}")
        print(f"  上极限: {formatted['y_axis']['limits']['top']:.2f}")
        if formatted['y_axis']['scale']['pixels_per_value']:
            print(f"  Y方向刻度: {formatted['y_axis']['scale']['pixels_per_value']:.6f} 单位/像素")
        if formatted['y_axis']['scale']['units_per_pixel']:
            print(f"  Y方向刻度: {formatted['y_axis']['scale']['units_per_pixel']:.6f} 像素/单位")
        
        print("\n坐标系信息:")
        if formatted['coordinate_system']['x_range']:
            print(f"  X轴范围: {formatted['coordinate_system']['x_range']:.2f} 像素")
        if formatted['coordinate_system']['y_range']:
            print(f"  Y轴范围: {formatted['coordinate_system']['y_range']:.2f} 像素")
        
        # 保存结果
        extractor.save_results(axes_info, "axes_results.json")
        
        # 显示可视化结果
        print("\n显示可视化结果...")
        extractor.visualize_results(image_path, axes_info)
        
    else:
        print(f"提取失败: {axes_info['message']}")