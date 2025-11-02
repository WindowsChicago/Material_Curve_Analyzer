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

# 更新类别标签，支持legend和legend_box
model_classify = ["legend", "legend_box"]

def recognize(img_path, threshold=0.5, save_output=False, output_dir=None, display=False):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return []
        
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
        if (score > threshold):
            scores.append(score)
            classifys.append(model_classify[int(class_id)])
            x, y, w, h = box[0].item(), box[1].item(), box[2].item(), box[3].item()
            x = int((x - 0.5 * w) * x_factor)
            y = int((y - 0.5 * h) * y_factor)
            w = int(w * x_factor)
            h = int(h * y_factor)
            box = np.array([x, y, w, h])
            boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
    
    # 处理NMSBoxes返回值的不同格式
    result_boxes = []
    if len(indexes) > 0:
        if isinstance(indexes, tuple) or len(indexes.shape) > 1:
            indexes = indexes.flatten()
        
        for i in indexes:
            classify, score, box = classifys[i], scores[i], boxes[i]
            print(f"Detected: {classify}, Score: {score:.4f}, Box: {box}")
            x, y, w, h = box[0], box[1], box[2], box[3]
            
            # 根据类别使用不同颜色
            color = (0, 255, 0) if classify == "legend" else (255, 0, 0)  # legend用绿色，legend_box用蓝色
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            label = f'{classify}: {score:.2f}'
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            result_boxes.append((classify, score, box))
    else:
        print("No objects detected")

    # 保存或显示结果
    if save_output and output_dir:
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        
        # 从输入路径提取文件名
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_detected{ext}")
        
        # 保存图像
        cv2.imwrite(output_path, img)
        print(f"Saved result to: {output_path}")
    
    if display:
        # 显示图像
        predict = cv2.resize(img, (1600, 900))
        cv2.imshow("img", predict)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result_boxes

def process_folder(input_folder, output_folder, threshold=0.5, image_extensions=('*.jpg', '*.jpeg', '*.png', '*.bmp'), display=False):
    """
    处理文件夹中的所有图像
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        threshold: 检测阈值
        image_extensions: 支持的图像扩展名
        display: 是否显示每张图像
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
        results = recognize(img_path, threshold, save_output=True, output_dir=output_folder, display=display)
        all_results[img_path] = results
    
    return all_results

if __name__ == '__main__':
    # 单张图像测试
    # recognize('3.jpg', 0.3, display=True)
    
    # 文件夹处理示例
    input_dir = "figs"  # 输入文件夹路径
    output_dir = "output_images"  # 输出文件夹路径
    
    # 确保输入文件夹存在
    if os.path.exists(input_dir):
        results = process_folder(input_dir, output_dir, threshold=0.3, display=False)
        print(f"\nProcessing completed. Results saved to {output_dir}")
        
        # 打印汇总信息
        total_detections = sum(len(detections) for detections in results.values())
        print(f"Total detections: {total_detections}")
        for img_path, detections in results.items():
            if detections:
                print(f"{os.path.basename(img_path)}: {len(detections)} detections")
    else:
        print(f"Input directory {input_dir} does not exist. Using single image mode.")
        # 回退到单张图像模式
        recognize('3.jpg', 0.3, display=True)