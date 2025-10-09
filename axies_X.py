import cv2
import numpy as np
import matplotlib.pyplot as plt
#plt.rc("font", family='Microsoft YaHei')
from scipy import signal
import easyocr
import re
from PIL import Image
import io
import os

# 初始化EasyOCR阅读器（支持英文和中文）
reader = easyocr.Reader(['ch_sim', 'en'],gpu=True)

def read_image_with_pil(image_path):
    """
    使用PIL读取图像，解决OpenCV无法处理LZW压缩的问题
    
    参数:
        image_path: 图像文件路径
        
    返回:
        img: numpy数组格式的图像(BGR格式，与OpenCV兼容)
    """
    try:
        # 使用PIL读取图像
        pil_img = Image.open(image_path)
        
        # 转换为RGB模式（如果需要）
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # 转换为numpy数组
        img_array = np.array(pil_img)
        
        # PIL是RGB格式，OpenCV需要BGR格式
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_bgr
    except Exception as e:
        raise ValueError(f"无法使用PIL读取图像: {e}")

def merge_close_ticks(ticks, min_distance=5):
    """
    合并过于接近的刻度，坐标取平均
    
    参数:
        ticks: 刻度坐标列表
        min_distance: 最小距离阈值，小于此值的刻度将被合并
        
    返回:
        合并后的刻度坐标列表
    """
    if not ticks:
        return ticks
    
    merged_ticks = []
    current_group = [ticks[0]]
    
    for i in range(1, len(ticks)):
        if ticks[i] - current_group[-1] < min_distance:
            current_group.append(ticks[i])
        else:
            # 合并当前组内的刻度，取平均
            merged_ticks.append(np.mean(current_group))
            current_group = [ticks[i]]
    
    # 处理最后一组
    if current_group:
        merged_ticks.append(np.mean(current_group))
    
    return merged_ticks

def filter_and_merge_texts(texts, confidence_threshold=0.5, y_threshold=10):
    """
    过滤置信度较低的文本，并按行合并文本
    
    参数:
        texts: 原始文本列表
        confidence_threshold: 置信度阈值，低于此值的文本将被过滤
        y_threshold: Y坐标阈值，用于判断是否为同一行
        
    返回:
        merged_texts: 合并后的文本列表
    """
    # 过滤低置信度文本
    filtered_texts = [text for text in texts if text['confidence'] >= confidence_threshold]
    
    if not filtered_texts:
        return []
    
    # 按Y坐标排序
    filtered_texts.sort(key=lambda x: x['position'][1])
    
    # 按行分组
    text_groups = []
    current_group = [filtered_texts[0]]
    
    for i in range(1, len(filtered_texts)):
        current_y = filtered_texts[i]['position'][1]
        prev_y = current_group[-1]['position'][1]
        
        if abs(current_y - prev_y) <= y_threshold:
            current_group.append(filtered_texts[i])
        else:
            text_groups.append(current_group)
            current_group = [filtered_texts[i]]
    
    if current_group:
        text_groups.append(current_group)
    
    # 合并每行的文本
    merged_texts = []
    for group in text_groups:
        # 按X坐标排序
        group.sort(key=lambda x: x['position'][0])
        
        # 拼接文本内容
        merged_content = ' '.join([text['content'] for text in group])
        
        # 计算平均位置
        avg_x = np.mean([text['position'][0] for text in group])
        avg_y = np.mean([text['position'][1] for text in group])
        
        # 计算合并后的边界框
        all_bbox_points = []
        for text in group:
            all_bbox_points.extend(text['bbox'])
        
        # 计算平均置信度
        avg_confidence = np.mean([text['confidence'] for text in group])
        
        merged_texts.append({
            'content': merged_content,
            'position': (avg_x, avg_y),
            'bbox': all_bbox_points,
            'confidence': avg_confidence,
            'original_texts': group  # 保留原始文本信息
        })
    
    return merged_texts

def extract_x_axis_title(merged_texts):
    """
    从合并的文本中提取x轴标题（最靠下方的长字符串）
    
    参数:
        merged_texts: 合并后的文本列表
        
    返回:
        x_axis_title: x轴标题信息，如果没有则返回None
        other_texts: 其他文本信息
    """
    if not merged_texts:
        return None, []
    
    # 按Y坐标排序（从大到小，即从下到上）
    merged_texts.sort(key=lambda x: x['position'][1], reverse=True)
    
    # 最靠下方的文本作为x轴标题候选
    candidate_titles = []
    
    # 考虑多个候选，选择最下方且长度较长的文本
    for text in merged_texts:
        # 计算文本长度（字符数）
        text_length = len(text['content'])
        
        # 计算边界框宽度
        x_coords = [point[0] for point in text['bbox']]
        bbox_width = max(x_coords) - min(x_coords)
        
        candidate_titles.append({
            'text': text,
            'length': text_length,
            'bbox_width': bbox_width,
            'y_position': text['position'][1]
        })
    
    # 按位置（最下方）和长度综合排序
    if candidate_titles:
        # 优先选择最下方的文本，如果长度太短则考虑次下方的
        for candidate in candidate_titles:
            if candidate['length'] >= 2:  # 至少2个字符
                x_axis_title = candidate['text']
                other_texts = [text for text in merged_texts if text != x_axis_title]
                return x_axis_title, other_texts
        
        # 如果没有找到合适的标题，返回最下方的文本
        x_axis_title = candidate_titles[0]['text']
        other_texts = [text for text in merged_texts if text != x_axis_title]
        return x_axis_title, other_texts
    
    return None, merged_texts

def extract_x_axis_coordinates_bidirectional(image_path):
    """
    提取X轴边缘横坐标和长刻度横坐标，支持双向刻度检测
    
    参数:
        image_path: 图像文件路径
    
    返回:
        left_edge: 左边缘横坐标
        right_edge: 右边缘横坐标  
        long_ticks: 长刻度横坐标列表
    """
    
    # 使用PIL读取图像
    img = read_image_with_pil(image_path)
    if img is None:
        raise ValueError("无法读取图像文件")
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 获取图像尺寸
    height, width = gray.shape
    
    # 提取下四分之一区域作为ROI
    roi_height = height // 4
    roi_start = height - roi_height
    roi = gray[roi_start:height, 0:width]
    roi_height, roi_width = roi.shape
    
    # 二值化处理
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 水平投影，检测X轴直线
    horizontal_projection = np.sum(binary, axis=0)
    
    # 使用滑动窗口平滑投影曲线
    window_size = 5
    kernel = np.ones(window_size) / window_size
    smoothed_projection = np.convolve(horizontal_projection, kernel, mode='same')
    
    # 找到X轴主线的位置（投影值最大的行）
    x_axis_line = np.argmax(np.sum(binary, axis=1))
    
    # 提取X轴线上的像素
    x_axis_row = binary[x_axis_line, :]
    
    # 找到X轴左右边缘
    x_nonzero = np.where(x_axis_row > 0)[0]
    if len(x_nonzero) > 0:
        left_edge = x_nonzero[0]
        right_edge = x_nonzero[-1]
    else:
        # 如果没有找到连续直线，使用投影的边缘
        left_edge = np.where(smoothed_projection > np.max(smoothed_projection) * 0.1)[0][0]
        right_edge = np.where(smoothed_projection > np.max(smoothed_projection) * 0.1)[0][-1]
    
    # 检测双向刻度线
    # 在X轴上方和下方区域查找垂直的线段
    above_x_axis = binary[max(0, x_axis_line - 50):x_axis_line, :]
    below_x_axis = binary[x_axis_line + 1:min(x_axis_line + 50, roi_height), :]
    
    # 检测上方刻度线
    above_tick_candidates = []
    above_vertical_profiles = []
    
    for col in range(roi_width):
        col_data = above_x_axis[:, col]
        if np.any(col_data > 0):
            # 找到连续白色像素的长度
            white_pixels = np.where(col_data > 0)[0]
            if len(white_pixels) > 0:
                # 计算连续段的长度
                segments = []
                current_segment = [white_pixels[0]]
                for i in range(1, len(white_pixels)):
                    if white_pixels[i] == white_pixels[i-1] + 1:
                        current_segment.append(white_pixels[i])
                    else:
                        segments.append(current_segment)
                        current_segment = [white_pixels[i]]
                segments.append(current_segment)
                
                # 取最长的连续段
                longest_segment = max(segments, key=len)
                segment_length = len(longest_segment)
                
                if segment_length > 3:  # 过滤掉噪声
                    above_vertical_profiles.append((col, segment_length))
                    above_tick_candidates.append(col)
    
    # 检测下方刻度线
    below_tick_candidates = []
    below_vertical_profiles = []
    
    for col in range(roi_width):
        col_data = below_x_axis[:, col]
        if np.any(col_data > 0):
            # 找到连续白色像素的长度
            white_pixels = np.where(col_data > 0)[0]
            if len(white_pixels) > 0:
                # 计算连续段的长度
                segments = []
                current_segment = [white_pixels[0]]
                for i in range(1, len(white_pixels)):
                    if white_pixels[i] == white_pixels[i-1] + 1:
                        current_segment.append(white_pixels[i])
                    else:
                        segments.append(current_segment)
                        current_segment = [white_pixels[i]]
                segments.append(current_segment)
                
                # 取最长的连续段
                longest_segment = max(segments, key=len)
                segment_length = len(longest_segment)
                
                if segment_length > 3:  # 过滤掉噪声
                    below_vertical_profiles.append((col, segment_length))
                    below_tick_candidates.append(col)
    
    # 合并双向刻度线
    all_tick_candidates = list(set(above_tick_candidates + below_tick_candidates))
    all_vertical_profiles = above_vertical_profiles + below_vertical_profiles
    
    # 创建位置到长度的映射
    tick_length_map = {}
    for col, length in all_vertical_profiles:
        if col in tick_length_map:
            tick_length_map[col] = max(tick_length_map[col], length)
        else:
            tick_length_map[col] = length
    
    # 区分长短刻度
    if tick_length_map:
        lengths = list(tick_length_map.values())
        
        # 使用k-means简单分类长短刻度
        if len(lengths) > 1:
            # 按长度排序并找到自然断点
            sorted_lengths = sorted(lengths)
            differences = [sorted_lengths[i+1] - sorted_lengths[i] for i in range(len(sorted_lengths)-1)]
            
            if len(differences) > 0:
                max_diff_index = np.argmax(differences)
                threshold = (sorted_lengths[max_diff_index] + sorted_lengths[max_diff_index + 1]) / 2

                # 调参放宽条件：将阈值乘以一个小于1的系数，例如0.7，这样就会将一些较短的刻度也纳入长刻度
                threshold = threshold * 0.7
                
                long_ticks = [col for col in all_tick_candidates if tick_length_map[col] >= threshold]
            else:
                long_ticks = all_tick_candidates
        else:
            long_ticks = all_tick_candidates
    else:
        long_ticks = []
    
    # 过滤掉距离太近的刻度（去重）
    long_ticks = sorted(long_ticks)
    filtered_long_ticks = []
    min_tick_distance = 10  # 最小刻度间距
    
    for tick in long_ticks:
        if not filtered_long_ticks or tick - filtered_long_ticks[-1] >= min_tick_distance:
            filtered_long_ticks.append(tick)
    
    # 合并过于接近的刻度
    filtered_long_ticks = merge_close_ticks(filtered_long_ticks, min_distance=5)
    
    # 检查边缘和最近刻度是否过于接近，如果是则合并
    if filtered_long_ticks:
        # 检查左边缘和第一个刻度
        if abs(left_edge - filtered_long_ticks[0]) < 5:
            # 合并左边缘和第一个刻度，取平均
            merged_value = (left_edge + filtered_long_ticks[0]) / 2
            left_edge = merged_value
            filtered_long_ticks[0] = merged_value
        
        # 检查右边缘和最后一个刻度
        if abs(right_edge - filtered_long_ticks[-1]) < 5:
            # 合并右边缘和最后一个刻度，取平均
            merged_value = (right_edge + filtered_long_ticks[-1]) / 2
            right_edge = merged_value
            filtered_long_ticks[-1] = merged_value
    
    return left_edge, right_edge, filtered_long_ticks, above_tick_candidates, below_tick_candidates

def extract_text_and_numbers_from_x_axis_region(image_path):
    """
    使用EasyOCR识别X轴区域（下四分之一）的文本和数字
    
    参数:
        image_path: 图像文件路径
        
    返回:
        numbers: 数字列表，每个元素为字典，包含'value'和'position'
        texts: 文本列表，每个元素为字典，包含'content'和'position'
    """
    # 使用PIL读取图像
    img = read_image_with_pil(image_path)
    if img is None:
        raise ValueError("无法读取图像文件")
    
    # 获取图像尺寸
    height, width = img.shape[:2]
    
    # 提取下四分之一区域作为OCR识别区域
    roi_height = height // 4
    roi_start = height - roi_height
    roi_region = img[roi_start:height, 0:width]
    
    # 将ROI区域保存为临时文件
    temp_path = "temp_ocr_roi.png"
    cv2.imwrite(temp_path, roi_region)
    
    try:
        # 使用EasyOCR识别ROI区域的文本
        results = reader.readtext(temp_path)
    finally:
        # 删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    numbers = []
    texts = []
    
    # 正则表达式匹配数字（包括整数、小数、负数、科学计数法）
    number_pattern = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
    
    for result in results:
        # result格式: (bbox, text, confidence)
        bbox, text, confidence = result
        
        # 清理文本，去除空格和特殊字符
        cleaned_text = text.strip()
        
        # 尝试将文本转换为数字
        try:
            # 检查是否是纯数字格式
            if re.match(number_pattern, cleaned_text):
                value = float(cleaned_text)
                
                # 计算边界框的中心位置（相对于ROI区域）
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                
                # 将坐标转换回原图坐标系
                # x坐标不变，y坐标需要加上ROI的起始位置
                center_x_full = center_x
                center_y_full = center_y + roi_start
                
                # 同时转换整个边界框的坐标
                bbox_full = [(point[0], point[1] + roi_start) for point in bbox]
                
                numbers.append({
                    'value': value,
                    'position': (center_x_full, center_y_full),
                    'bbox': bbox_full,  # 完整边界框（相对于全图）
                    'confidence': confidence
                })
            else:
                # 如果不是数字，作为文本处理
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                
                # 将坐标转换回原图坐标系
                center_x_full = center_x
                center_y_full = center_y + roi_start
                
                # 同时转换整个边界框的坐标
                bbox_full = [(point[0], point[1] + roi_start) for point in bbox]
                
                texts.append({
                    'content': cleaned_text,
                    'position': (center_x_full, center_y_full),
                    'bbox': bbox_full,  # 完整边界框（相对于全图）
                    'confidence': confidence
                })
        except (ValueError, TypeError):
            # 转换失败，作为文本处理
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            
            # 将坐标转换回原图坐标系
            center_x_full = center_x
            center_y_full = center_y + roi_start
            
            # 同时转换整个边界框的坐标
            bbox_full = [(point[0], point[1] + roi_start) for point in bbox]
            
            texts.append({
                'content': cleaned_text,
                'position': (center_x_full, center_y_full),
                'bbox': bbox_full,  # 完整边界框（相对于全图）
                'confidence': confidence
            })
    
    return numbers, texts

def calculate_pixels_per_value(long_ticks, numbers, max_distance=15):
    """
    计算X方向每像素对应的刻度值
    
    参数:
        long_ticks: 长刻度横坐标列表
        numbers: 数字列表，每个元素包含'value'和'position'
        max_distance: 数字与刻度匹配的最大距离阈值
        
    返回:
        pixels_per_value: 每像素对应的刻度值
        matched_pairs: 匹配的刻度-数字对列表
    """
    matched_pairs = []
    
    # 对每个数字，找到最近的长刻度
    for number in numbers:
        num_x = number['position'][0]
        num_value = number['value']
        
        # 找到最近的长刻度
        min_distance = float('inf')
        nearest_tick = None
        
        for tick_x in long_ticks:
            distance = abs(tick_x - num_x)
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                nearest_tick = tick_x
        
        if nearest_tick is not None:
            matched_pairs.append({
                'tick_x': nearest_tick,
                'num_value': num_value,
                'distance': min_distance
            })
    
    # 如果没有找到匹配对，返回None
    if len(matched_pairs) < 2:
        return None, matched_pairs
    
    # 按刻度位置排序
    matched_pairs.sort(key=lambda x: x['tick_x'])
    
    # 计算所有相邻匹配对之间的像素值与数值比例
    ratios = []
    valid_pairs = []
    
    for i in range(len(matched_pairs) - 1):
        pair1 = matched_pairs[i]
        pair2 = matched_pairs[i + 1]
        
        # 计算像素差和数值差
        pixel_diff = pair2['tick_x'] - pair1['tick_x']
        value_diff = pair2['num_value'] - pair1['num_value']
        
        # 避免除以零
        if pixel_diff != 0:
            ratio = value_diff / pixel_diff
            ratios.append(ratio)
            valid_pairs.append((pair1, pair2))
    
    # 如果没有有效的比例，返回None
    if not ratios:
        return None, matched_pairs
    
    # 计算平均比例
    avg_ratio = np.mean(ratios)
    
    return avg_ratio, matched_pairs

def visualize_bidirectional_results_with_ocr(image_path, left_edge, right_edge, long_ticks, 
                                           above_ticks, below_ticks, numbers, texts, 
                                           merged_texts=None, x_axis_title=None, other_texts=None,
                                           pixels_per_value=None, matched_pairs=None):
    """
    可视化双向刻度检测结果和OCR识别结果
    """
    img = cv2.imread(image_path)
    if img is None:
        img = read_image_with_pil(image_path)
    
    height, width = img.shape[:2]
    roi_height = height // 4
    roi_start = height - roi_height
    
    # 在图像上绘制结果
    result_img = img.copy()
    
    # 绘制X轴边缘
    cv2.line(result_img, (int(left_edge), roi_start + roi_height//2), 
             (int(right_edge), roi_start + roi_height//2), (0, 255, 0), 2)
    
    # 绘制左边缘标记
    cv2.circle(result_img, (int(left_edge), roi_start + roi_height//2), 5, (255, 0, 0), -1)
    
    # 绘制右边缘标记  
    cv2.circle(result_img, (int(right_edge), roi_start + roi_height//2), 5, (255, 0, 0), -1)
    
    # 绘制上方刻度
    for tick in above_ticks:
        cv2.line(result_img, (tick, roi_start + roi_height//2 - 20), 
                (tick, roi_start + roi_height//2), (255, 255, 0), 2)
    
    # 绘制下方刻度
    for tick in below_ticks:
        cv2.line(result_img, (tick, roi_start + roi_height//2), 
                (tick, roi_start + roi_height//2 + 20), (255, 165, 0), 2)
    
    # 绘制长刻度（用红色突出显示）
    for tick in long_ticks:
        # 检查是上方还是下方刻度
        if tick in above_ticks:
            cv2.line(result_img, (int(tick), roi_start + roi_height//2 - 30), 
                    (int(tick), roi_start + roi_height//2), (0, 0, 255), 3)
        else:
            cv2.line(result_img, (int(tick), roi_start + roi_height//2), 
                    (int(tick), roi_start + roi_height//2 + 30), (0, 0, 255), 3)
        cv2.circle(result_img, (int(tick), roi_start + roi_height//2), 3, (0, 0, 255), -1)
    
    # 绘制OCR识别的数字和文本
    # 绘制数字（用绿色框和文本）
    for number in numbers:
        bbox = number['bbox']
        value = number['value']
        confidence = number['confidence']
        
        # 绘制边界框
        pts = np.array(bbox, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(result_img, [pts], True, (0, 255, 0), 2)
        
        # 在框上方显示数值
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        text_position = (int(min_x), int(min_y) - 5)
        cv2.putText(result_img, f"{value:.4f}", text_position, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 绘制原始文本（用浅蓝色框和文本）
    for text_item in texts:
        bbox = text_item['bbox']
        content = text_item['content']
        confidence = text_item['confidence']
        
        # 绘制边界框
        pts = np.array(bbox, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(result_img, [pts], True, (173, 216, 230), 1)  # 浅蓝色
        
        # 在框上方显示文本内容
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        text_position = (int(min_x), int(min_y) - 5)
        cv2.putText(result_img, content, text_position, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (173, 216, 230), 1)
    
    # 绘制合并后的文本（用深蓝色框和文本）
    if merged_texts:
        for text_item in merged_texts:
            bbox = text_item['bbox']
            content = text_item['content']
            confidence = text_item['confidence']
            
            # 绘制边界框
            pts = np.array(bbox, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(result_img, [pts], True, (0, 0, 255), 2)  # 红色
            
            # 在框上方显示文本内容
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            text_position = (int(min_x), int(min_y) - 10)
            cv2.putText(result_img, content, text_position, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # 绘制x轴标题（用紫色突出显示）
    if x_axis_title:
        bbox = x_axis_title['bbox']
        content = x_axis_title['content']
        
        # 绘制边界框
        pts = np.array(bbox, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(result_img, [pts], True, (255, 0, 255), 3)  # 紫色
        
        # 在框上方显示标题内容
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        text_position = (int(min_x), int(min_y) - 15)
        cv2.putText(result_img, f"X轴标题: {content}", text_position, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 3)
    
    # 绘制匹配的刻度-数字对（用紫色连线）
    if matched_pairs:
        for pair in matched_pairs:
            tick_x = pair['tick_x']
            num_value = pair['num_value']
            
            # 找到对应的数字位置
            for number in numbers:
                if abs(number['position'][0] - tick_x) <= 15 and abs(number['value'] - num_value) < 1e-6:
                    num_x, num_y = number['position']
                    
                    # 绘制连线
                    cv2.line(result_img, 
                            (int(tick_x), roi_start + roi_height//2),
                            (int(num_x), int(num_y)),
                            (255, 0, 255), 2)
                    
                    # 在连线上标注比例
                    if pixels_per_value is not None:
                        mid_x = (tick_x + num_x) / 2
                        mid_y = (roi_start + roi_height//2 + num_y) / 2
                        cv2.putText(result_img, f"匹配", 
                                   (int(mid_x), int(mid_y)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    break
    
    # 绘制ROI区域边界
    cv2.rectangle(result_img, (0, roi_start), (width, height), (128, 128, 128), 2)
    cv2.putText(result_img, "OCR识别区域", (10, roi_start + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
    
    # 在图像上显示每像素对应的刻度值
    if pixels_per_value is not None:
        info_text = f"X方向刻度: {pixels_per_value:.6f} 单位/像素"
        cv2.putText(result_img, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 显示结果
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('原图')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title('双向刻度检测和OCR识别结果')
    plt.axis('off')
    
    # 显示ROI区域
    roi_img = img[roi_start:height, :]
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
    plt.title('X轴ROI区域（OCR识别区域）')
    plt.axis('off')
    
    # 绘制刻度分布图
    plt.subplot(2, 2, 4)
    if above_ticks:
        plt.scatter(above_ticks, [1]*len(above_ticks), c='yellow', label='上方刻度', s=50)
    if below_ticks:
        plt.scatter(below_ticks, [0]*len(below_ticks), c='orange', label='下方刻度', s=50)
    if long_ticks:
        plt.scatter(long_ticks, [0.5]*len(long_ticks), c='red', label='长刻度', s=100, marker='x')
    
    # 绘制OCR识别结果的位置
    if numbers:
        number_x = [num['position'][0] for num in numbers]
        plt.scatter(number_x, [1.2]*len(numbers), c='green', label='识别数字', s=80, marker='s')
    
    if texts:
        text_x = [text['position'][0] for text in texts]
        plt.scatter(text_x, [1.4]*len(texts), c='lightblue', label='原始文本', s=60, marker='^')
    
    if merged_texts:
        merged_text_x = [text['position'][0] for text in merged_texts]
        plt.scatter(merged_text_x, [1.6]*len(merged_texts), c='blue', label='合并文本', s=80, marker='o')
    
    if x_axis_title:
        plt.scatter([x_axis_title['position'][0]], [1.8], c='purple', label='X轴标题', s=120, marker='*')
    
    # 绘制匹配的刻度-数字对
    if matched_pairs:
        matched_tick_x = [pair['tick_x'] for pair in matched_pairs]
        plt.scatter(matched_tick_x, [2.0]*len(matched_pairs), c='magenta', label='匹配对', s=100, marker='D')
    
    plt.axvline(x=left_edge, color='blue', linestyle='--', label='X轴边缘')
    plt.axvline(x=right_edge, color='blue', linestyle='--')
    plt.xlabel('横坐标')
    plt.yticks([0, 0.5, 1, 1.2, 1.4, 1.6, 1.8, 2.0], 
               ['下方刻度', '长刻度', '上方刻度', '识别数字', '原始文本', '合并文本', 'X轴标题', '匹配对'])
    plt.title('刻度和OCR识别分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 替换为您的图像路径
    image_path = "1.jpg"
    
    try:
        # 提取坐标
        left_edge, right_edge, long_ticks, above_ticks, below_ticks = extract_x_axis_coordinates_bidirectional(image_path)
        
        # 使用EasyOCR识别X轴区域的文本和数字
        numbers, texts = extract_text_and_numbers_from_x_axis_region(image_path)
        
        # 过滤和合并文本
        merged_texts = filter_and_merge_texts(texts, confidence_threshold=0.5, y_threshold=10)
        
        # 提取x轴标题
        x_axis_title, other_texts = extract_x_axis_title(merged_texts)
        
        # 计算X方向每像素对应的刻度值
        pixels_per_value, matched_pairs = calculate_pixels_per_value(long_ticks, numbers)
        
        # 打印结果
        print(f"X轴左边缘横坐标: {left_edge}")
        print(f"X轴右边缘横坐标: {right_edge}")
        print(f"长刻度横坐标: {long_ticks}")
        print(f"上方刻度横坐标: {above_ticks}")
        print(f"下方刻度横坐标: {below_ticks}")
        print(f"检测到 {len(long_ticks)} 个长刻度")
        print(f"检测到 {len(above_ticks)} 个上方刻度")
        print(f"检测到 {len(below_ticks)} 个下方刻度")
        
        print("\nOCR识别结果:")
        print("数字:")
        for i, number in enumerate(numbers):
            print(f"  {i+1}. 数值: {number['value']}, 位置: ({number['position'][0]:.2f}, {number['position'][1]:.2f}), 置信度: {number['confidence']:.4f}")
        
        print("\n原始文本:")
        for i, text in enumerate(texts):
            print(f"  {i+1}. 内容: '{text['content']}', 位置: ({text['position'][0]:.2f}, {text['position'][1]:.2f}), 置信度: {text['confidence']:.4f}")
        
        print("\n合并后的文本:")
        for i, text in enumerate(merged_texts):
            print(f"  {i+1}. 内容: '{text['content']}', 位置: ({text['position'][0]:.2f}, {text['position'][1]:.2f}), 置信度: {text['confidence']:.4f}")
        
        if x_axis_title:
            print(f"\nX轴标题: '{x_axis_title['content']}'")
            print(f"  位置: ({x_axis_title['position'][0]:.2f}, {x_axis_title['position'][1]:.2f})")
            print(f"  置信度: {x_axis_title['confidence']:.4f}")
        else:
            print("\n未检测到X轴标题")
        
        # 输出每像素对应的刻度值
        if pixels_per_value is not None:
            print(f"\nX方向每像素对应的刻度值: {pixels_per_value:.6f} 单位/像素")
            print(f"匹配的刻度-数字对数量: {len(matched_pairs)}")
            for i, pair in enumerate(matched_pairs):
                print(f"  匹配对 {i+1}: 刻度位置={pair['tick_x']:.2f}, 数值={pair['num_value']}, 距离={pair['distance']:.2f}")
        else:
            print("\n无法计算X方向每像素对应的刻度值，需要至少两个匹配的刻度-数字对")
        
        # 可视化结果
        visualize_bidirectional_results_with_ocr(image_path, left_edge, right_edge, long_ticks, 
                                               above_ticks, below_ticks, numbers, texts, 
                                               merged_texts, x_axis_title, other_texts,
                                               pixels_per_value, matched_pairs)
        
    except Exception as e:
        print(f"处理图像时出错: {e}")