import cv2
import numpy as np
import os
import glob

onnx_model_path = "best-SimAM-M2.onnx"
input_shape = (640, 640)

# 检查CUDA可用性
def check_cuda_available():
    try:
        # 尝试创建CUDA后端网络
        net = cv2.dnn.readNetFromONNX(onnx_model_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        # 测试一个简单的前向传播
        test_blob = cv2.dnn.blobFromImage(np.ones((640, 640, 3), dtype=np.uint8), 1/255.0, input_shape, swapRB=True, crop=False)
        net.setInput(test_blob)
        _ = net.forward()
        
        print("CUDA backend is available and working")
        return True
    except Exception as e:
        print(f"CUDA not available: {e}")
        print("Falling back to CPU")
        return False

# 初始化网络
net = cv2.dnn.readNetFromONNX(onnx_model_path)

# 根据CUDA可用性设置后端
cuda_available = check_cuda_available()
if cuda_available:
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using CUDA for inference")
    except:
        print("Failed to set CUDA backend, using CPU instead")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
else:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU for inference")

# 更新类别标签，只关注legend_box
model_classify = ["legend_box"]

def extract_dominant_color(region):
    """
    提取图像区域中的主要颜色
    
    Args:
        region: 图像区域 (numpy数组)
    
    Returns:
        dominant_color: 主要颜色 (B, G, R)
        percentage: 该颜色在区域中的占比
    """
    if region.size == 0:
        return None, 0
        
    # 将图像区域转换为一维数组
    pixels = region.reshape(-1, 3)
    
    # 使用K-means聚类找到主要颜色
    try:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels.astype(np.float32), min(2, len(pixels)), None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # 计算每个聚类的像素数量
        unique, counts = np.unique(labels, return_counts=True)
        
        # 找到像素数量最多的聚类
        dominant_cluster = unique[np.argmax(counts)]
        dominant_color = centers[dominant_cluster].astype(int)
        percentage = np.max(counts) / len(labels)
        
        return dominant_color, percentage
    except Exception as e:
        print(f"Error in color extraction: {e}")
        return None, 0

def remove_background_color(img, target_color, tolerance=10):
    """
    从图像中移除特定颜色
    
    Args:
        img: 原始图像
        target_color: 目标颜色 (B, G, R)
        tolerance: 颜色容差
    
    Returns:
        result_img: 处理后的图像
    """
    # 创建目标颜色的掩码
    lower_bound = np.array([max(0, c - tolerance) for c in target_color])
    upper_bound = np.array([min(255, c + tolerance) for c in target_color])
    
    mask = cv2.inRange(img, lower_bound, upper_bound)
    
    # 将目标颜色替换为白色
    result_img = img.copy()
    result_img[mask > 0] = [255, 255, 255]
    
    return result_img

def clean_image_background(img_path, threshold=0.5, clean_background=True):
    """
    清除图像背景颜色，不绘制检测框
    
    Args:
        img_path: 图像路径
        threshold: 检测阈值
        clean_background: 是否清除背景
    
    Returns:
        cleaned_img: 清除背景后的图像
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return img
    
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, input_shape, swapRB=True, crop=False)
    net.setInput(blob)

    output = net.forward()
    output = output.transpose((0, 2, 1))

    height, width, _ = img.shape
    x_factor, y_factor = width / input_shape[0], height / input_shape[1]

    scores, boxes = [], []
    for i in range(output[0].shape[0]):
        box = output[0][i]
        _, _, _, max_idx = cv2.minMaxLoc(box[4:])
        class_id = max_idx[1]
        score = box[4:][class_id]
        
        # 只处理legend_box类别
        if class_id < len(model_classify) and score > threshold:
            scores.append(score)
            x, y, w, h = box[0].item(), box[1].item(), box[2].item(), box[3].item()
            x = int((x - 0.5 * w) * x_factor)
            y = int((y - 0.5 * h) * y_factor)
            w = int(w * x_factor)
            h = int(h * y_factor)
            box = np.array([x, y, w, h])
            boxes.append(box)

    # 检查是否有检测结果
    if len(boxes) == 0:
        print("No legend_box objects detected for background cleaning")
        return img.copy()

    indexes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
    
    cleaned_img = img.copy()
    
    if len(indexes) > 0:
        if isinstance(indexes, tuple) or (hasattr(indexes, 'shape') and len(indexes.shape) > 1):
            indexes = indexes.flatten()
        
        for i in indexes:
            # 确保索引在范围内
            if i >= len(scores) or i >= len(boxes):
                continue
                
            score, box = scores[i], boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            
            # 确保坐标在图像范围内
            x = max(0, min(x, width-1))
            y = max(0, min(y, height-1))
            w = max(1, min(w, width-x))
            h = max(1, min(h, height-y))
            
            # 提取图例框区域
            legend_box_region = img[y:y+h, x:x+w]
            
            if clean_background and legend_box_region.size > 0:
                # 提取主要颜色
                dominant_color, percentage = extract_dominant_color(legend_box_region)
                
                if dominant_color is not None:
                    # 检查是否需要清除背景
                    black_threshold = 30
                    white_threshold = 225
                    
                    is_black = all(c < black_threshold for c in dominant_color)
                    is_white = all(c > white_threshold for c in dominant_color)
                    
                    print(f"Background cleaning - Dominant color: {dominant_color}, Percentage: {percentage:.2f}, Is black: {is_black}, Is white: {is_white}")
                    
                    # 如果主要颜色不是黑色或白色，并且占比超过50%，则清除该颜色
                    if not is_black and not is_white and percentage > 0.5:
                        print(f"Removing background color: {dominant_color}")
                        cleaned_img = remove_background_color(cleaned_img, dominant_color)
    
    return cleaned_img

def recognize_and_clean(img_path, threshold=0.5, save_output=False, output_dir=None, display=False, clean_background=True):
    """
    识别legend_box并清除背景
    
    Args:
        img_path: 图像路径
        threshold: 检测阈值
        save_output: 是否保存输出
        output_dir: 输出目录
        display: 是否显示结果
        clean_background: 是否清除背景
    
    Returns:
        result_boxes: 检测到的图例框信息
        cleaned_img: 清除背景后的图像 (如果clean_background为True)
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return [], None
        
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, input_shape, swapRB=True, crop=False)
    net.setInput(blob)

    output = net.forward()
    output = output.transpose((0, 2, 1))

    height, width, _ = img.shape
    x_factor, y_factor = width / input_shape[0], height / input_shape[1]

    classifys, scores, boxes = [], [], []
    for i in range(output[0].shape[0]):
        box = output[0][i]
        _, _, _, max_idx = cv2.minMaxLoc(box[4:])
        class_id = max_idx[1]
        score = box[4:][class_id]
        
        # 只处理legend_box类别
        if class_id < len(model_classify) and score > threshold:
            scores.append(score)
            classifys.append(model_classify[int(class_id)])
            x, y, w, h = box[0].item(), box[1].item(), box[2].item(), box[3].item()
            x = int((x - 0.5 * w) * x_factor)
            y = int((y - 0.5 * h) * y_factor)
            w = int(w * x_factor)
            h = int(h * y_factor)
            box = np.array([x, y, w, h])
            boxes.append(box)

    # 检查是否有检测结果
    if len(boxes) == 0:
        print("No legend_box objects detected")
        return [], img.copy()

    indexes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
    
    # 处理NMSBoxes返回值的不同格式
    result_boxes = []
    cleaned_img = img.copy()
    
    if len(indexes) > 0:
        if isinstance(indexes, tuple) or (hasattr(indexes, 'shape') and len(indexes.shape) > 1):
            indexes = indexes.flatten()
        
        for i in indexes:
            # 确保索引在范围内
            if i >= len(classifys) or i >= len(scores) or i >= len(boxes):
                continue
                
            classify, score, box = classifys[i], scores[i], boxes[i]
            print(f"Detected: {classify}, Score: {score:.4f}, Box: {box}")
            x, y, w, h = box[0], box[1], box[2], box[3]
            
            # 只处理legend_box
            if classify == "legend_box":
                # 确保坐标在图像范围内
                x = max(0, min(x, width-1))
                y = max(0, min(y, height-1))
                w = max(1, min(w, width-x))
                h = max(1, min(h, height-y))
                
                # 提取图例框区域
                legend_box_region = img[y:y+h, x:x+w]
                
                if clean_background and legend_box_region.size > 0:
                    # 提取主要颜色
                    dominant_color, percentage = extract_dominant_color(legend_box_region)
                    
                    if dominant_color is not None:
                        # 检查是否需要清除背景
                        black_threshold = 30
                        white_threshold = 225
                        
                        is_black = all(c < black_threshold for c in dominant_color)
                        is_white = all(c > white_threshold for c in dominant_color)
                        
                        print(f"Dominant color: {dominant_color}, Percentage: {percentage:.2f}, Is black: {is_black}, Is white: {is_white}")
                        
                        # 如果主要颜色不是黑色或白色，并且占比超过50%，则清除该颜色
                        if not is_black and not is_white and percentage > 0.5:
                            print(f"Removing background color: {dominant_color}")
                            cleaned_img = remove_background_color(cleaned_img, dominant_color)
                
                # 绘制检测框（使用蓝色）
                color = (255, 0, 0)  # 蓝色
                cv2.rectangle(cleaned_img, (x, y), (x + w, y + h), color, 3)
                label = f'{classify}: {score:.2f}'
                cv2.putText(cleaned_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                result_boxes.append((classify, score, box))
    else:
        print("No legend_box objects detected after NMS")

    # 保存或显示结果
    if save_output and output_dir:
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        
        # 从输入路径提取文件名
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_detected{ext}")
        
        # 保存图像
        cv2.imwrite(output_path, cleaned_img)
        print(f"Saved result to: {output_path}")
    
    if display:
        # 显示图像
        predict = cv2.resize(cleaned_img, (1600, 900))
        cv2.imshow("img", predict)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result_boxes, cleaned_img

def process_folder_clean(input_folder, output_folder, threshold=0.5, image_extensions=('*.jpg', '*.jpeg', '*.png', '*.bmp'), display=False, clean_background=True):
    """
    处理文件夹中的所有图像，并清除图例框背景
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        threshold: 检测阈值
        image_extensions: 支持的图像扩展名
        display: 是否显示每张图像
        clean_background: 是否清除背景
    
    Returns:
        all_results: 所有图像的检测结果
    """
    # 查找所有图像文件
    image_files = []
    for extension in image_extensions:
        pattern = os.path.join(input_folder, extension)
        image_files.extend(glob.glob(pattern))
        # 同时查找大写扩展名
        pattern_upper = os.path.join(input_folder, extension.upper())
        image_files.extend(glob.glob(pattern_upper))
    
    # 去重
    image_files = list(set(image_files))
    
    print(f"Found {len(image_files)} images in {input_folder}")
    
    all_results = {}
    for i, img_path in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {img_path}")
        results, _ = recognize_and_clean(img_path, threshold, save_output=True, output_dir=output_folder, display=display, clean_background=clean_background)
        all_results[img_path] = results
    
    return all_results

# API函数 - 便于其他程序调用
def detect_and_clean_legend_boxes(img_path, threshold=0.5, clean_background=True):
    """
    API函数：检测图例框并清除背景
    
    Args:
        img_path: 图像路径
        threshold: 检测阈值
        clean_background: 是否清除背景
    
    Returns:
        result_boxes: 检测到的图例框信息列表，每个元素为(类别, 置信度, [x, y, w, h])
        cleaned_img: 清除背景后的图像 (如果clean_background为True)
    """
    return recognize_and_clean(img_path, threshold, save_output=False, output_dir=None, display=False, clean_background=clean_background)

def clean_image_background_only(img_path, threshold=0.5, clean_background=True):
    """
    API函数：仅清除图像背景，不返回检测框信息
    
    Args:
        img_path: 图像路径
        threshold: 检测阈值
        clean_background: 是否清除背景
    
    Returns:
        cleaned_img: 清除背景后的图像
    """
    return clean_image_background(img_path, threshold, clean_background)

def batch_detect_and_clean_legend_boxes(input_folder, output_folder, threshold=0.5, clean_background=True):
    """
    API函数：批量检测图例框并清除背景
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        threshold: 检测阈值
        clean_background: 是否清除背景
    
    Returns:
        all_results: 所有图像的检测结果字典，键为图像路径，值为检测结果列表
    """
    return process_folder_clean(input_folder, output_folder, threshold, display=False, clean_background=clean_background)

if __name__ == '__main__':
    # 单张图像测试
    # results, cleaned_img = detect_and_clean_legend_boxes('3.jpg', 0.3, clean_background=True)
    
    # 文件夹处理示例
    input_dir = "figs"  # 输入文件夹路径
    output_dir = "output_images"  # 输出文件夹路径
    
    # 确保输入文件夹存在
    if os.path.exists(input_dir):
        results = batch_detect_and_clean_legend_boxes(input_dir, output_dir, threshold=0.3, clean_background=True)
        print(f"\nProcessing completed. Results saved to {output_dir}")
        
        # 打印汇总信息
        total_detections = sum(len(detections) for detections in results.values())
        print(f"Total legend_box detections: {total_detections}")
        for img_path, detections in results.items():
            if detections:
                print(f"{os.path.basename(img_path)}: {len(detections)} legend_box detections")
    else:
        print(f"Input directory {input_dir} does not exist. Using single image mode.")
        # 回退到单张图像模式
        results, cleaned_img = detect_and_clean_legend_boxes('3.jpg', 0.3, clean_background=True)
        print(f"Detected {len(results)} legend_box(es)")