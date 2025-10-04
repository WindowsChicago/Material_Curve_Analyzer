import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rc("font", family='Microsoft YaHei')
from scipy import signal
import easyocr
import re
from PIL import Image
import os

# 初始化EasyOCR阅读器（支持英文和中文）
reader = easyocr.Reader(['ch_sim', 'en'])

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

def filter_and_merge_texts(texts, confidence_threshold=0.5, x_threshold=10):
    """
    过滤置信度较低的文本，并按列合并文本
    
    参数:
        texts: 原始文本列表
        confidence_threshold: 置信度阈值，低于此值的文本将被过滤
        x_threshold: X坐标阈值，用于判断是否为同一列
        
    返回:
        merged_texts: 合并后的文本列表
    """
    # 过滤低置信度文本
    filtered_texts = [text for text in texts if text['confidence'] >= confidence_threshold]
    
    if not filtered_texts:
        return []
    
    # 按X坐标排序
    filtered_texts.sort(key=lambda x: x['position'][0])
    
    # 按列分组
    text_groups = []
    current_group = [filtered_texts[0]]
    
    for i in range(1, len(filtered_texts)):
        current_x = filtered_texts[i]['position'][0]
        prev_x = current_group[-1]['position'][0]
        
        if abs(current_x - prev_x) <= x_threshold:
            current_group.append(filtered_texts[i])
        else:
            text_groups.append(current_group)
            current_group = [filtered_texts[i]]
    
    if current_group:
        text_groups.append(current_group)
    
    # 合并每列的文本
    merged_texts = []
    for group in text_groups:
        # 按Y坐标排序（从上到下）
        group.sort(key=lambda x: x['position'][1])
        
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

def extract_rotated_y_axis_title(image_path, y_axis_position, numbers, texts):
    """
    专门提取旋转的Y轴标题
    通过旋转图像区域使标题变为正常方向，然后进行识别
    
    参数:
        image_path: 图像文件路径
        y_axis_position: Y轴位置（"left"或"right"）
        numbers: 已识别的数字列表（用于排除数字区域）
        texts: 已识别的文本列表（用于排除已识别区域）
        
    返回:
        y_axis_title: Y轴标题信息，如果没有则返回None
    """
    # 使用PIL读取图像
    img = read_image_with_pil(image_path)
    if img is None:
        raise ValueError("无法读取图像文件")
    
    # 获取图像尺寸
    height, width = img.shape[:2]
    
    # 根据Y轴位置确定标题区域
    if y_axis_position == "left":
        # 左侧Y轴，标题通常在左侧中间位置
        title_roi_width = width // 8
        title_roi_height = height // 3
        title_roi_x = 0
        title_roi_y = (height - title_roi_height) // 2
    else:
        # 右侧Y轴，标题通常在右侧中间位置
        title_roi_width = width // 8
        title_roi_height = height // 3
        title_roi_x = width - title_roi_width
        title_roi_y = (height - title_roi_height) // 2
    
    # 提取标题区域
    title_roi = img[title_roi_y:title_roi_y+title_roi_height, 
                   title_roi_x:title_roi_x+title_roi_width]
    
    # 将标题区域顺时针旋转90度，使旋转的标题变为正常方向
    rotated_title_roi = cv2.rotate(title_roi, cv2.ROTATE_90_CLOCKWISE)
    
    # 将旋转后的区域保存为临时文件
    temp_path = "temp_y_title_rotated.png"
    cv2.imwrite(temp_path, rotated_title_roi)
    
    try:
        # 使用EasyOCR识别旋转后的标题区域
        title_results = reader.readtext(temp_path)
    finally:
        # 删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # 处理识别结果
    title_candidates = []
    
    # 正则表达式匹配数字（用于排除数字）
    number_pattern = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
    
    for result in title_results:
        bbox, text, confidence = result
        
        # 清理文本
        cleaned_text = text.strip()
        
        # 排除数字和短文本（标题通常是较长的文本）
        if (not re.match(number_pattern, cleaned_text) and 
            len(cleaned_text) > 1 and  # 排除单个字符
            confidence >= 0.5):  # 置信度阈值
        
            # 计算边界框的中心位置（相对于旋转后的ROI区域）
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            
            # 将坐标转换回原图坐标系
            # 由于图像被旋转了，需要反向转换坐标
            if y_axis_position == "left":
                # 对于左侧Y轴，旋转前的坐标转换
                orig_x = title_roi_x + (title_roi_height - center_y)
                orig_y = title_roi_y + center_x
            else:
                # 对于右侧Y轴，旋转前的坐标转换
                orig_x = title_roi_x + center_y
                orig_y = title_roi_y + (title_roi_width - center_x)
            
            # 转换整个边界框的坐标
            bbox_orig = []
            for point in bbox:
                if y_axis_position == "left":
                    orig_point_x = title_roi_x + (title_roi_height - point[1])
                    orig_point_y = title_roi_y + point[0]
                else:
                    orig_point_x = title_roi_x + point[1]
                    orig_point_y = title_roi_y + (title_roi_width - point[0])
                bbox_orig.append((orig_point_x, orig_point_y))
            
            title_candidates.append({
                'content': cleaned_text,
                'position': (orig_x, orig_y),
                'bbox': bbox_orig,
                'confidence': confidence
            })
    
    # 选择最可能的标题
    if title_candidates:
        # 按置信度和文本长度综合评分
        for candidate in title_candidates:
            # 评分公式：置信度 * 文本长度
            candidate['score'] = candidate['confidence'] * len(candidate['content'])
        
        # 选择评分最高的候选作为标题
        best_candidate = max(title_candidates, key=lambda x: x['score'])
        return best_candidate
    
    return None

def extract_y_axis_title(merged_texts, y_axis_position):
    """
    从合并的文本中提取y轴标题（最靠外侧的长字符串）
    
    参数:
        merged_texts: 合并后的文本列表
        y_axis_position: Y轴位置（"left"或"right"）
        
    返回:
        y_axis_title: y轴标题信息，如果没有则返回None
        other_texts: 其他文本信息
    """
    if not merged_texts:
        return None, []
    
    # 根据Y轴位置选择标题候选
    if y_axis_position == "left":
        # 对于左侧Y轴，标题通常在最左侧
        merged_texts.sort(key=lambda x: x['position'][0])  # 按X坐标排序（从小到大）
    else:
        # 对于右侧Y轴，标题通常在最右侧
        merged_texts.sort(key=lambda x: x['position'][0], reverse=True)  # 按X坐标排序（从大到小）
    
    # 考虑多个候选，选择最外侧且长度较长的文本
    candidate_titles = []
    
    for text in merged_texts:
        # 计算文本长度（字符数）
        text_length = len(text['content'])
        
        # 计算边界框高度
        y_coords = [point[1] for point in text['bbox']]
        bbox_height = max(y_coords) - min(y_coords)
        
        candidate_titles.append({
            'text': text,
            'length': text_length,
            'bbox_height': bbox_height,
            'x_position': text['position'][0]
        })
    
    # 按位置（最外侧）和长度综合排序
    if candidate_titles:
        # 优先选择最外侧的文本，如果长度太短则考虑次外侧的
        for candidate in candidate_titles:
            if candidate['length'] >= 2:  # 至少2个字符
                y_axis_title = candidate['text']
                other_texts = [text for text in merged_texts if text != y_axis_title]
                return y_axis_title, other_texts
        
        # 如果没有找到合适的标题，返回最外侧的文本
        y_axis_title = candidate_titles[0]['text']
        other_texts = [text for text in merged_texts if text != y_axis_title]
        return y_axis_title, other_texts
    
    return None, merged_texts

def extract_y_axis_coordinates_bidirectional_auto(image_path):
    """
    自动检测Y轴位置并提取坐标，支持双向刻度检测
    
    参数:
        image_path: 图像文件路径
    
    返回:
        bottom_edge: 下边缘纵坐标
        top_edge: 上边缘纵坐标  
        long_ticks: 长刻度纵坐标列表
        left_ticks: 左侧刻度纵坐标列表
        right_ticks: 右侧刻度纵坐标列表
        y_axis_position: Y轴在图像中的位置（"left"或"right"）
    """
    
    # 使用PIL读取图像
    img = read_image_with_pil(image_path)
    if img is None:
        raise ValueError("无法读取图像文件")
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 获取图像尺寸
    height, width = gray.shape
    
    # 尝试在左侧和右侧都检测Y轴
    left_roi_width = width // 4
    right_roi_width = width // 4
    
    left_roi = gray[0:height, 0:left_roi_width]
    right_roi = gray[0:height, width - right_roi_width:width]
    
    # 二值化处理
    _, left_binary = cv2.threshold(left_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, right_binary = cv2.threshold(right_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 计算左右区域的垂直投影强度
    left_vertical_strength = np.sum(np.sum(left_binary, axis=0))
    right_vertical_strength = np.sum(np.sum(right_binary, axis=0))
    
    # 确定Y轴在哪一侧
    if left_vertical_strength > right_vertical_strength:
        # Y轴在左侧
        roi = left_binary
        roi_offset_x = 0
        y_axis_position = "left"
    else:
        # Y轴在右侧
        roi = right_binary
        roi_offset_x = width - right_roi_width
        y_axis_position = "right"
    
    roi_height, roi_width = roi.shape
    
    # 垂直投影，检测Y轴直线
    vertical_projection = np.sum(roi, axis=1)
    
    # 使用滑动窗口平滑投影曲线
    window_size = 5
    kernel = np.ones(window_size) / window_size
    smoothed_projection = np.convolve(vertical_projection, kernel, mode='same')
    
    # 找到Y轴主线的位置（投影值最大的列）
    y_axis_col = np.argmax(np.sum(roi, axis=0))
    
    # 提取Y轴线上的像素
    y_axis_column = roi[:, y_axis_col]
    
    # 找到Y轴上下边缘
    y_nonzero = np.where(y_axis_column > 0)[0]
    if len(y_nonzero) > 0:
        top_edge = y_nonzero[0]
        bottom_edge = y_nonzero[-1]
    else:
        # 如果没有找到连续直线，使用投影的边缘
        nonzero_indices = np.where(smoothed_projection > np.max(smoothed_projection) * 0.1)[0]
        if len(nonzero_indices) > 0:
            top_edge = nonzero_indices[0]
            bottom_edge = nonzero_indices[-1]
        else:
            top_edge = 0
            bottom_edge = roi_height - 1
    
    # 检测双向刻度线
    if y_axis_position == "left":
        # Y轴在左侧，刻度线在右侧和左侧
        left_of_y_axis = roi[0:roi_height, max(0, y_axis_col - 50):y_axis_col]
        right_of_y_axis = roi[0:roi_height, y_axis_col + 1:min(y_axis_col + 50, roi_width)]
    else:
        # Y轴在右侧，刻度线在左侧和右侧
        left_of_y_axis = roi[0:roi_height, y_axis_col + 1:min(y_axis_col + 50, roi_width)]
        right_of_y_axis = roi[0:roi_height, max(0, y_axis_col - 50):y_axis_col]
    
    # 检测左侧刻度线
    left_tick_candidates = []
    left_horizontal_profiles = []
    
    for row in range(roi_height):
        row_data = left_of_y_axis[row, :]
        if np.any(row_data > 0):
            # 找到连续白色像素的长度
            white_pixels = np.where(row_data > 0)[0]
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
                    left_horizontal_profiles.append((row, segment_length))
                    left_tick_candidates.append(row)
    
    # 检测右侧刻度线
    right_tick_candidates = []
    right_horizontal_profiles = []
    
    for row in range(roi_height):
        row_data = right_of_y_axis[row, :]
        if np.any(row_data > 0):
            # 找到连续白色像素的长度
            white_pixels = np.where(row_data > 0)[0]
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
                    right_horizontal_profiles.append((row, segment_length))
                    right_tick_candidates.append(row)
    
    # 合并双向刻度线
    all_tick_candidates = list(set(left_tick_candidates + right_tick_candidates))
    all_horizontal_profiles = left_horizontal_profiles + right_horizontal_profiles
    
    # 创建位置到长度的映射
    tick_length_map = {}
    for row, length in all_horizontal_profiles:
        if row in tick_length_map:
            tick_length_map[row] = max(tick_length_map[row], length)
        else:
            tick_length_map[row] = length
    
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
                
                long_ticks = [row for row in all_tick_candidates if tick_length_map[row] >= threshold]
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
        # 检查上边缘和第一个刻度
        if abs(top_edge - filtered_long_ticks[0]) < 5:
            # 合并上边缘和第一个刻度，取平均
            merged_value = (top_edge + filtered_long_ticks[0]) / 2
            top_edge = merged_value
            filtered_long_ticks[0] = merged_value
        
        # 检查下边缘和最后一个刻度
        if abs(bottom_edge - filtered_long_ticks[-1]) < 5:
            # 合并下边缘和最后一个刻度，取平均
            merged_value = (bottom_edge + filtered_long_ticks[-1]) / 2
            bottom_edge = merged_value
            filtered_long_ticks[-1] = merged_value
    
    # 调整坐标到原图坐标系
    bottom_edge_global = bottom_edge
    top_edge_global = top_edge
    long_ticks_global = filtered_long_ticks
    left_ticks_global = left_tick_candidates
    right_ticks_global = right_tick_candidates
    
    return bottom_edge_global, top_edge_global, long_ticks_global, left_ticks_global, right_ticks_global, y_axis_position

def extract_text_and_numbers_from_y_axis_region(image_path, y_axis_position):
    """
    使用EasyOCR识别Y轴区域（左或右四分之一）的文本和数字
    
    参数:
        image_path: 图像文件路径
        y_axis_position: Y轴位置（"left"或"right"）
        
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
    
    # 根据Y轴位置提取相应的ROI区域
    roi_width = width // 4
    if y_axis_position == "left":
        # Y轴在左侧，提取左四分之一区域
        roi_region = img[0:height, 0:roi_width]
        roi_offset_x = 0
    else:
        # Y轴在右侧，提取右四分之一区域
        roi_region = img[0:height, width - roi_width:width]
        roi_offset_x = width - roi_width
    
    # 将ROI区域保存为临时文件
    temp_path = "temp_ocr_roi_y.png"
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
                # x坐标需要加上ROI的偏移量，y坐标不变
                center_x_full = center_x + roi_offset_x
                center_y_full = center_y
                
                # 同时转换整个边界框的坐标
                bbox_full = [(point[0] + roi_offset_x, point[1]) for point in bbox]
                
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
                center_x_full = center_x + roi_offset_x
                center_y_full = center_y
                
                # 同时转换整个边界框的坐标
                bbox_full = [(point[0] + roi_offset_x, point[1]) for point in bbox]
                
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
            center_x_full = center_x + roi_offset_x
            center_y_full = center_y
            
            # 同时转换整个边界框的坐标
            bbox_full = [(point[0] + roi_offset_x, point[1]) for point in bbox]
            
            texts.append({
                'content': cleaned_text,
                'position': (center_x_full, center_y_full),
                'bbox': bbox_full,  # 完整边界框（相对于全图）
                'confidence': confidence
            })
    
    return numbers, texts

def calculate_pixels_per_value_y(long_ticks, numbers, max_distance=15):
    """
    计算Y方向每像素对应的刻度值
    
    参数:
        long_ticks: 长刻度纵坐标列表
        numbers: 数字列表，每个元素包含'value'和'position'
        max_distance: 数字与刻度匹配的最大距离阈值
        
    返回:
        pixels_per_value: 每像素对应的刻度值
        matched_pairs: 匹配的刻度-数字对列表
    """
    matched_pairs = []
    
    # 对每个数字，找到最近的长刻度
    for number in numbers:
        num_y = number['position'][1]
        num_value = number['value']
        
        # 找到最近的长刻度
        min_distance = float('inf')
        nearest_tick = None
        
        for tick_y in long_ticks:
            distance = abs(tick_y - num_y)
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                nearest_tick = tick_y
        
        if nearest_tick is not None:
            matched_pairs.append({
                'tick_y': nearest_tick,
                'num_value': num_value,
                'distance': min_distance
            })
    
    # 如果没有找到匹配对，返回None
    if len(matched_pairs) < 2:
        return None, matched_pairs
    
    # 按刻度位置排序
    matched_pairs.sort(key=lambda x: x['tick_y'])
    
    # 计算所有相邻匹配对之间的像素值与数值比例
    ratios = []
    valid_pairs = []
    
    for i in range(len(matched_pairs) - 1):
        pair1 = matched_pairs[i]
        pair2 = matched_pairs[i + 1]
        
        # 计算像素差和数值差
        pixel_diff = pair2['tick_y'] - pair1['tick_y']
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

def visualize_y_bidirectional_results_with_ocr(image_path, bottom_edge, top_edge, long_ticks, 
                                             left_ticks, right_ticks, y_axis_position, 
                                             numbers, texts, merged_texts=None, y_axis_title=None, other_texts=None,
                                             rotated_y_axis_title=None, pixels_per_value=None, matched_pairs=None):
    """
    可视化Y轴双向刻度检测结果和OCR识别结果
    """
    img = cv2.imread(image_path)
    if img is None:
        img = read_image_with_pil(image_path)
    
    height, width = img.shape[:2]
    roi_width = width // 4
    
    # 根据Y轴位置确定ROI区域
    if y_axis_position == "left":
        roi_start_x = 0
        roi_end_x = roi_width
        y_axis_x = roi_width // 2
    else:
        roi_start_x = width - roi_width
        roi_end_x = width
        y_axis_x = roi_start_x + roi_width // 2
    
    # 在图像上绘制结果
    result_img = img.copy()
    
    # 绘制Y轴边缘
    cv2.line(result_img, (y_axis_x, int(top_edge)), 
             (y_axis_x, int(bottom_edge)), (0, 255, 0), 2)
    
    # 绘制上边缘标记
    cv2.circle(result_img, (y_axis_x, int(top_edge)), 5, (255, 0, 0), -1)
    
    # 绘制下边缘标记  
    cv2.circle(result_img, (y_axis_x, int(bottom_edge)), 5, (255, 0, 0), -1)
    
    # 绘制左侧刻度
    for tick in left_ticks:
        cv2.line(result_img, (y_axis_x - 20, tick), 
                (y_axis_x, tick), (255, 255, 0), 2)
    
    # 绘制右侧刻度
    for tick in right_ticks:
        cv2.line(result_img, (y_axis_x, tick), 
                (y_axis_x + 20, tick), (255, 165, 0), 2)
    
    # 绘制长刻度（用红色突出显示）
    for tick in long_ticks:
        # 检查是左侧还是右侧刻度
        if tick in left_ticks:
            cv2.line(result_img, (y_axis_x - 30, int(tick)), 
                    (y_axis_x, int(tick)), (0, 0, 255), 3)
        else:
            cv2.line(result_img, (y_axis_x, int(tick)), 
                    (y_axis_x + 30, int(tick)), (0, 0, 255), 3)
        cv2.circle(result_img, (y_axis_x, int(tick)), 3, (0, 0, 255), -1)
    
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
        
        # 在框左侧显示数值
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        text_position = (int(min_x) - 5, int(min_y))
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
        
        # 在框左侧显示文本内容
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        text_position = (int(min_x) - 5, int(min_y))
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
            
            # 在框左侧显示文本内容
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            text_position = (int(min_x) - 10, int(min_y))
            cv2.putText(result_img, content, text_position, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # 绘制y轴标题（用紫色突出显示）
    if y_axis_title:
        bbox = y_axis_title['bbox']
        content = y_axis_title['content']
        
        # 绘制边界框
        pts = np.array(bbox, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(result_img, [pts], True, (255, 0, 255), 3)  # 紫色
        
        # 在框左侧显示标题内容
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        text_position = (int(min_x) - 15, int(min_y))
        cv2.putText(result_img, f"Y轴标题: {content}", text_position, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 3)
    
    # 绘制旋转识别到的y轴标题（用橙色突出显示）
    if rotated_y_axis_title:
        bbox = rotated_y_axis_title['bbox']
        content = rotated_y_axis_title['content']
        
        # 绘制边界框
        pts = np.array(bbox, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(result_img, [pts], True, (0, 165, 255), 3)  # 橙色
        
        # 在框左侧显示标题内容
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        text_position = (int(min_x) - 15, int(min_y) - 20)
        cv2.putText(result_img, f"旋转识别标题: {content}", text_position, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 3)
    
    # 绘制匹配的刻度-数字对（用紫色连线）
    if matched_pairs:
        for pair in matched_pairs:
            tick_y = pair['tick_y']
            num_value = pair['num_value']
            
            # 找到对应的数字位置
            for number in numbers:
                if abs(number['position'][1] - tick_y) <= 15 and abs(number['value'] - num_value) < 1e-6:
                    num_x, num_y = number['position']
                    
                    # 绘制连线
                    cv2.line(result_img, 
                            (y_axis_x, int(tick_y)),
                            (int(num_x), int(num_y)),
                            (255, 0, 255), 2)
                    
                    # 在连线上标注比例
                    if pixels_per_value is not None:
                        mid_x = (y_axis_x + num_x) / 2
                        mid_y = (tick_y + num_y) / 2
                        cv2.putText(result_img, f"匹配", 
                                   (int(mid_x), int(mid_y)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    break
    
    # 绘制ROI区域边界
    cv2.rectangle(result_img, (roi_start_x, 0), (roi_end_x, height), (128, 128, 128), 2)
    cv2.putText(result_img, "OCR识别区域", (roi_start_x + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
    
    # 在图像上显示每像素对应的刻度值
    if pixels_per_value is not None:
        info_text = f"Y方向刻度: {pixels_per_value:.6f} 单位/像素"
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
    plt.title('Y轴双向刻度检测和OCR识别结果')
    plt.axis('off')
    
    # 显示ROI区域
    roi_img = img[0:height, roi_start_x:roi_end_x]
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
    plt.title('Y轴ROI区域（OCR识别区域）')
    plt.axis('off')
    
    # 绘制刻度分布图
    plt.subplot(2, 2, 4)
    if left_ticks:
        plt.scatter([1]*len(left_ticks), left_ticks, c='yellow', label='左侧刻度', s=50)
    if right_ticks:
        plt.scatter([0]*len(right_ticks), right_ticks, c='orange', label='右侧刻度', s=50)
    if long_ticks:
        plt.scatter([0.5]*len(long_ticks), long_ticks, c='red', label='长刻度', s=100, marker='x')
    
    # 绘制OCR识别结果的位置
    if numbers:
        number_y = [num['position'][1] for num in numbers]
        plt.scatter([1.2]*len(numbers), number_y, c='green', label='识别数字', s=80, marker='s')
    
    if texts:
        text_y = [text['position'][1] for text in texts]
        plt.scatter([1.4]*len(texts), text_y, c='lightblue', label='原始文本', s=60, marker='^')
    
    if merged_texts:
        merged_text_y = [text['position'][1] for text in merged_texts]
        plt.scatter([1.6]*len(merged_texts), merged_text_y, c='blue', label='合并文本', s=80, marker='o')
    
    if y_axis_title:
        plt.scatter([1.8], [y_axis_title['position'][1]], c='purple', label='Y轴标题', s=120, marker='*')
    
    if rotated_y_axis_title:
        plt.scatter([2.0], [rotated_y_axis_title['position'][1]], c='orange', label='旋转识别标题', s=120, marker='D')
    
    # 绘制匹配的刻度-数字对
    if matched_pairs:
        matched_tick_y = [pair['tick_y'] for pair in matched_pairs]
        plt.scatter([2.2]*len(matched_pairs), matched_tick_y, c='magenta', label='匹配对', s=100, marker='+')
    
    plt.axhline(y=bottom_edge, color='blue', linestyle='--', label='Y轴边缘')
    plt.axhline(y=top_edge, color='blue', linestyle='--')
    plt.ylabel('纵坐标')
    plt.xticks([0, 0.5, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2], 
               ['右侧刻度', '长刻度', '左侧刻度', '识别数字', '原始文本', '合并文本', 'Y轴标题', '旋转标题', '匹配对'])
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
        # 自动检测Y轴位置
        print("自动检测Y轴位置")
        bottom_edge, top_edge, long_ticks, left_ticks, right_ticks, y_axis_position = extract_y_axis_coordinates_bidirectional_auto(image_path)
        
        # 使用EasyOCR识别Y轴区域的文本和数字
        numbers, texts = extract_text_and_numbers_from_y_axis_region(image_path, y_axis_position)
        
        # 过滤和合并文本
        merged_texts = filter_and_merge_texts(texts, confidence_threshold=0.5, x_threshold=10)
        
        # 提取y轴标题（传统方法）
        y_axis_title, other_texts = extract_y_axis_title(merged_texts, y_axis_position)
        
        # 提取旋转的y轴标题（新方法）
        rotated_y_axis_title = extract_rotated_y_axis_title(image_path, y_axis_position, numbers, texts)
        
        # 计算Y方向每像素对应的刻度值
        pixels_per_value, matched_pairs = calculate_pixels_per_value_y(long_ticks, numbers)
        
        # 打印结果
        print(f"Y轴位置: {y_axis_position}")
        print(f"Y轴下边缘纵坐标: {bottom_edge}")
        print(f"Y轴上边缘纵坐标: {top_edge}")
        print(f"长刻度纵坐标: {long_ticks}")
        print(f"左侧刻度纵坐标: {left_ticks}")
        print(f"右侧刻度纵坐标: {right_ticks}")
        print(f"检测到 {len(long_ticks)} 个长刻度")
        print(f"检测到 {len(left_ticks)} 个左侧刻度")
        print(f"检测到 {len(right_ticks)} 个右侧刻度")
        
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
        
        if y_axis_title:
            print(f"\nY轴标题（传统方法）: '{y_axis_title['content']}'")
            print(f"  位置: ({y_axis_title['position'][0]:.2f}, {y_axis_title['position'][1]:.2f})")
            print(f"  置信度: {y_axis_title['confidence']:.4f}")
        else:
            print("\n未检测到Y轴标题（传统方法）")
        
        if rotated_y_axis_title:
            print(f"\nY轴标题（旋转识别）: '{rotated_y_axis_title['content']}'")
            print(f"  位置: ({rotated_y_axis_title['position'][0]:.2f}, {rotated_y_axis_title['position'][1]:.2f})")
            print(f"  置信度: {rotated_y_axis_title['confidence']:.4f}")
        else:
            print("\n未检测到Y轴标题（旋转识别）")
        
        # 输出每像素对应的刻度值
        if pixels_per_value is not None:
            print(f"\nY方向每像素对应的刻度值: {pixels_per_value:.6f} 单位/像素")
            print(f"匹配的刻度-数字对数量: {len(matched_pairs)}")
            for i, pair in enumerate(matched_pairs):
                print(f"  匹配对 {i+1}: 刻度位置={pair['tick_y']:.2f}, 数值={pair['num_value']}, 距离={pair['distance']:.2f}")
        else:
            print("\n无法计算Y方向每像素对应的刻度值，需要至少两个匹配的刻度-数字对")
        
        # 可视化结果
        visualize_y_bidirectional_results_with_ocr(image_path, bottom_edge, top_edge, long_ticks, 
                                                 left_ticks, right_ticks, y_axis_position,
                                                 numbers, texts, merged_texts, y_axis_title, other_texts,
                                                 rotated_y_axis_title, pixels_per_value, matched_pairs)
        
    except Exception as e:
        print(f"处理图像时出错: {e}")