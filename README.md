# Material_Curve_Analyzer

##### by ZYL

#### 开发日志：

* RC1:基于M3,为legend模块引入EasyOCR，合并axies\_reunion.py和axies\_API.py,优化legend标签的识别方案，是上传git的最初版本
* RC2:加入对linux的支持，转移开发环境到linux，加入requirements.txt以便于配置环境,修改了gitignore屏蔽更多类型的无用文件
* RC3:加入读取文件夹的功能，对于对数轴导致的报错问题暂时采用忽略的方式处理,为OCR功能加入CUDA加速，大幅提升了运行速度，修改了结果文件输出的格式为比赛要求的格式
* RC4:引入多线程设计（HyperThread V1和HyperPipeline V1）,大幅提升了运行速度，在测试机上运行时间缩短了约28%（4分10秒->3分03秒，works=4），修复了数轴识别匹配时无法正常识别负号的问题，为yolo部分引入了CUDA加速
* RC5:修改文件输出，删除了sheet2（改为另外输出一个txt文件）、修改了单元格格式以解决裁判系统报错的问题，修改了文件输出顺序，改进了多线程设计（HyperThread V2，删除了HyperPipeLine模式），合并了系统算法说明和README.md,此为初赛的最终版本
* RC6:修改文件输出格式为复赛格式，更新图例识别模型为复赛版本，加入了引入了CBAM、SimAM和公版三种模型，曲线提取器新增颜色宽容设计，初步修复了坐标轴在复赛数据集上的匹配问题，由于存在问题暂时禁用了超线程
* RC7:删除大量无用代码，删除传统视觉的OCR,此为复赛第一次提交的版本
* RC8:修复了图形界面调试时在Linux下无法正常显示中文的问题，引入带图例框版本的模型（暂未使用），重写了曲线提取逻辑，目前对部分曲线识别效果提升较大，但是代价是大幅增加了图例被识别到的问题，并且识别黑色曲线会识别为坐标轴并大幅降低了运行速度，以及颜色宽容还得调
* RC9:曲线识别加入对图例区域的覆盖，防止识别到图例区域
#### 整体设计：

* 采用高度模块化设计，每个组件均可单独运行调试，方便进行改进
* 传统视觉与神经网络融合设计，保证了高准确性和运行效率

# 算法说明：

## 一、系统概述

本系统是一个基于计算机视觉和深度学习的科学图像数据处理平台，专门用于从科学图表图像中自动提取坐标轴信息、识别图例、分离曲线数据，并将像素坐标转换为实际物理坐标。系统采用模块化设计，支持多线程并行处理，能够高效处理大批量科学图像数据。

## 二、主要原理

### 2.1 系统架构原理

系统采用分层处理架构，包含四个核心处理层：

**数据输入层**：支持多种图像格式（JPG、JPEG、TIF、TIFF、PNG），自动检测输入图像质量并进行预处理。

**特征提取层**：
- 坐标轴检测模块：基于边缘检测和投影分析
- 图例识别模块：基于YOLO目标检测和OCR文本识别
- 曲线分离模块：基于颜色聚类和空间连续性分析

**坐标转换层**：建立像素坐标系到物理坐标系的映射关系，实现数据标准化。

**数据输出层**：生成结构化Excel数据，包含原始信息和转换后的物理坐标。

### 2.2 坐标轴检测算法原理

#### 2.2.1 X轴检测算法

**ROI区域定位**：
```python
# 提取下四分之一区域作为X轴检测区域
roi_height = height // 4
roi_start = height - roi_height
roi = gray[roi_start:height, 0:width]
```

**边缘检测与投影分析**：
- 使用OTSU自适应二值化处理ROI区域
- 水平投影分析确定X轴基线位置：
  ```python
  horizontal_projection = np.sum(binary, axis=0)
  smoothed_projection = np.convolve(horizontal_projection, kernel, mode='same')
  x_axis_line = np.argmax(np.sum(binary, axis=1))
  ```

**双向刻度检测**：
- 在X轴上方和下方分别检测垂直线段
- 基于线段长度分类长短刻度
- 使用K-means聚类自动确定长短刻度阈值

**OCR数字识别与匹配**：
```python
# 使用EasyOCR识别X轴区域文本
numbers, texts = extract_text_num_x_axis_region(image_path)
# 数字与刻度位置匹配
pixels_per_value, matched_pairs = calculate_pixels_per_value(long_ticks, numbers)
```

#### 2.2.2 Y轴检测算法

**自适应位置检测**：
```python
# 同时检测左右两侧区域，选择投影强度较大的一侧作为Y轴
left_vertical_strength = np.sum(np.sum(left_binary, axis=0))
right_vertical_strength = np.sum(np.sum(right_binary, axis=0))
y_axis_position = "left" if left_vertical_strength > right_vertical_strength else "right"
```

**旋转标题识别**：
针对垂直书写的Y轴标题，采用图像旋转技术：
```python
# 将标题区域顺时针旋转90度，使文本变为正常方向
rotated_title_roi = cv2.rotate(title_roi, cv2.ROTATE_90_CLOCKWISE)
# 识别旋转后的文本
title_results = reader.readtext(rotated_title_roi)
```

### 2.3 图例检测与识别原理

#### 2.3.1 基于YOLO的图例区域检测

**模型架构**：
- 使用ONNX格式的预训练YOLO模型
- 输入尺寸：640×640像素
- 输出格式：边界框坐标、置信度、类别

**检测流程**：
```python
# 预处理
blob = cv2.dnn.blobFromImage(img, 1/255.0, input_shape, swapRB=True, crop=False)
net.setInput(blob)
# 前向传播
output = net.forward()
output = output.transpose((0, 2, 1))
# 后处理与非极大值抑制
indexes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
```

#### 2.3.2 基于EasyOCR的文本识别

**图像预处理**：
- CLAHE对比度限制自适应直方图均衡化
- 中值滤波降噪
- 灰度转换

**多语言识别**：
```python
reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
results = reader.readtext(processed_image, detail=1)
```

**文本后处理**：
- 置信度过滤（默认阈值0.5）
- 位置排序（从上到下，从左到右）
- 文本合并与清理

#### 2.3.3 主色调提取算法

**颜色空间分析**：
```python
# 转换为RGB颜色空间
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 过滤黑色和白色
filtered_pixels = [pixel for pixel in pixels 
                  if not any(np.sqrt(np.sum((pixel - ignore_color) ** 2)) <= color_tolerance 
                  for ignore_color in [(0, 0, 0), (255, 255, 255)])]
# 统计颜色频率
color_counter = Counter(filtered_pixels)
dominant_color = color_counter.most_common(1)[0][0]
```

### 2.4 曲线提取原理

#### 2.4.1 基于颜色的曲线分离

**颜色掩码创建**：
```python
def find_color_mask(self, target_color, tolerance=40):
    target_rgb = self.hex_to_rgb(target_color)
    target_array = np.array(target_rgb, dtype=np.uint8)
    # 计算颜色欧氏距离
    color_diff = np.sqrt(np.sum((self.image_rgb.astype(np.float32) - target_array) ** 2, axis=2))
    mask = color_diff < tolerance
    return mask.astype(np.uint8) * 255
```

**坐标轴与文本过滤**：
- 基于轮廓分析过滤大面积区域（可能是坐标轴）
- 基于位置信息过滤边缘区域
- 保留中等面积的连续区域（曲线特征）

#### 2.4.2 曲线点聚类与排序

**DBSCAN空间聚类**：
```python
# 去除离散噪声点
clustering = DBSCAN(eps=5, min_samples=5).fit(points)
main_cluster_mask = clustering.labels_ == 0
points = points[main_cluster_mask]
```

**角度排序算法**：
```python
def sort_curve_points(self, points):
    center = np.mean(points, axis=0)
    # 计算各点相对于中心的角度
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]
```

#### 2.4.3 曲线重采样

**基于弧长的参数化插值**：
```python
def resample_curve(self, points, num_points=128):
    sorted_points = self.sort_curve_points(points)
    # 计算累积弧长
    distances = np.cumsum(np.sqrt(np.sum(np.diff(sorted_points, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)
    normalized_distances = distances / distances[-1]
    
    # 线性插值
    fx = interpolate.interp1d(normalized_distances, sorted_points[:, 0], kind='linear')
    fy = interpolate.interp1d(normalized_distances, sorted_points[:, 1], kind='linear')
    
    new_distances = np.linspace(0, 1, num_points)
    new_x = fx(new_distances)
    new_y = fy(new_distances)
    
    return np.column_stack((new_x, new_y))
```

### 2.5 坐标转换原理

**线性映射模型**：
```
实际横坐标 = (像素横坐标 - X轴左极限) × X方向每像素对应的刻度值
实际纵坐标 = (像素纵坐标 - Y轴底部极限值) × Y方向每像素对应的刻度值
```

数学表达式：
$$x_{actual} = (x_{pixel} - x_{left}) \times scale_x$$
$$y_{actual} = (y_{pixel} - y_{bottom}) \times scale_y$$

其中：
- $scale_x = \frac{\Delta value_x}{\Delta pixel_x}$ （X方向单位/像素）
- $scale_y = \frac{\Delta value_y}{\Delta pixel_y}$ （Y方向单位/像素）

### 2.6 多线程处理原理

#### 2.6.1 线程安全设计

**资源隔离**：
```python
def create_image_cache_dir(image_path, base_cache_dir="image_caches"):
    # 为每个图像创建独立的缓存目录
    cache_dir = os.path.join(base_cache_dir, f"{image_id}_{uuid.uuid4().hex[:8]}")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir
```

**线程安全输出**：
```python
print_lock = threading.Lock()

def thread_safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)
```

#### 2.6.2 并行处理架构

**ThreadPoolExecutor模式**：
```python
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_image = {executor.submit(process_single_image, image_path): os.path.basename(image_path) 
                      for image_path in image_files}
    
    for future in concurrent.futures.as_completed(future_to_image):
        result = future.result()
        # 处理结果收集
```

## 三、创新点说明

### 3.1 算法创新

**1. 双向刻度检测技术**
传统方法通常只检测单方向刻度，本系统创新性地实现了双向刻度检测，能够准确识别X轴上下方和Y轴左右侧的刻度线，提高了刻度检测的鲁棒性。

**2. 自适应坐标轴定位**
通过左右区域投影强度对比自动确定Y轴位置，无需预设坐标轴位置，适应多种图表布局。

**3. 旋转文本识别技术**
针对垂直书写的Y轴标题，采用图像旋转+OCR识别的方法，解决了传统OCR对旋转文本识别率低的问题。

**4. 多模态特征融合**
将颜色特征、空间特征、文本特征深度融合，实现准确的图例-曲线关联。

### 3.2 工程创新

**1. 容错性设计**
- 坐标轴提取失败时自动使用默认值
- 曲线提取失败时跳过当前曲线继续处理
- 多层次异常处理机制

**2. 缓存隔离架构**
每个处理线程拥有独立的缓存目录，避免多线程环境下的资源冲突，提高系统稳定性。

**3. 自适应参数调整**
- 自动确定长短刻度分类阈值
- 动态调整颜色容差参数
- 自适应OCR置信度阈值

**4. 内存友好设计**
- 流式处理大数据集
- 及时清理中间结果
- 可控的线程数量

### 3.3 输出格式创新

**结构化数据存储**：
```python
excel_data = {
    'DOI': "图像名称 (ID)",
    'X-label': "X轴物理量",
    'Y-label': "Y轴物理量", 
    'sample': "曲线标识",
    'Values': "标准化坐标数据",
    'note': "备注信息"
}
```

这种格式既保留了原始信息，又提供了机器可读的结构化数据，便于后续数据分析。

## 四、代码逻辑详解

### 4.1 训练逻辑

#### 4.1.1 YOLO模型训练

**数据准备**：
- 收集科学图表图像数据集
- 使用LabelImg等工具标注图例区域
- 数据增强：旋转、缩放、颜色变换

**训练配置**：
```yaml
model_type: YOLOv5
input_size: 640x640
batch_size: 16
epochs: 100
optimizer: Adam
learning_rate: 0.001
```

**模型导出**：
训练完成后将PyTorch模型转换为ONNX格式，提高推理效率并减少依赖。

#### 4.1.2 OCR模型使用

系统使用预训练的EasyOCR模型，支持中文和英文识别，无需额外训练。

### 4.2 推理逻辑

#### 4.2.1 单图像处理流程

```python
def process_single_image(image_path):
    # 1. 坐标轴信息提取
    axes_info = extract_axes_info(image_path)
    
    # 2. 曲线数据提取  
    curve_results = extract_curves(image_path)
    
    # 3. 坐标转换
    transformed_curves = coordinate_transformation(curve_results, axes_info)
    
    # 4. 数据格式化
    excel_data = format_to_excel(transformed_curves, axes_info)
    
    return excel_data
```

#### 4.2.2 批量处理流程

```python
def process_all_images_multithreaded(figs_folder, output_file, max_workers):
    # 1. 扫描图像文件
    image_files = scan_image_files(figs_folder)
    
    # 2. 创建线程池
    with ThreadPoolExecutor(max_workers) as executor:
        # 3. 提交任务
        futures = [executor.submit(process_single_image, img_path) 
                  for img_path in image_files]
        
        # 4. 收集结果
        results = collect_results(futures)
    
    # 5. 生成输出
    generate_excel_output(results, output_file)
```

### 4.3 外推逻辑

#### 4.3.1 模型泛化能力

**数据分布外推**：
- 通过数据增强提高模型对不同图表风格的适应性
- 使用多尺度特征提取应对不同分辨率的图像
- 颜色归一化处理减少光照变化影响

**异常处理机制**：
```python
try:
    result = processing_pipeline(image_path)
except AxisDetectionError:
    result = fallback_axis_detection(image_path)
except CurveExtractionError:
    result = alternative_curve_extraction(image_path)
except Exception as e:
    log_error(e)
    result = None
```

#### 4.3.2 参数自适应

**动态阈值调整**：
```python
# 根据图像特性自动调整参数
if image_quality < threshold:
    color_tolerance *= 1.5
    ocr_confidence_threshold *= 0.8
```

## 五、性能优化策略

### 5.1 计算优化

**1. 区域限制处理**
只处理感兴趣的图像区域，减少计算量：
- X轴：仅处理底部1/4区域
- Y轴：仅处理左侧或右侧1/4区域
- 图例：仅在检测到的边界框内处理

**2. 早停机制**
当关键步骤失败时及时终止当前图像处理，避免不必要的计算。

**3. 内存优化**
及时释放中间结果，控制同时处理的图像数量。

### 5.2 I/O优化

**1. 缓存策略**
为每个线程创建独立缓存目录，避免磁盘I/O冲突。

**2. 批量写入**
收集所有结果后一次性写入Excel文件，减少文件操作次数。

**3. 流式处理**
支持大规模图像数据集的处理，无需一次性加载所有数据。

## 六、参考文献

### 6.1 计算机视觉基础

[1] Bradski, G., & Kaehler, A. (2008). Learning OpenCV: Computer vision with the OpenCV library. O'Reilly Media.

[2] Szeliski, R. (2010). Computer vision: algorithms and applications. Springer Science & Business Media.

### 6.2 目标检测

[3] Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition.

[4] Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). YOLOv4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934.

### 6.3 文本识别

[5] Baek, J., Kim, G., Lee, J., Park, S., Han, D., Yun, S., ... & Lee, H. (2019). What is wrong with scene text recognition model comparisons? dataset and model analysis. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

### 6.4 图像处理

[6] Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE transactions on systems, man, and cybernetics, 9(1), 62-66.

[7] Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. Graphics gems, 474-485.

### 6.5 数据提取相关

[8] Choudhury, S., & Mitra, P. (2021). Automatic data extraction from scientific plots. In Proceedings of the 2021 IEEE International Conference on Image Processing.

[9] Savva, M., Kong, N., Chhajta, A., Fei-Fei, L., Agrawala, M., & Heer, J. (2011). Revision: Automated classification, analysis and redesign of chart images. In Proceedings of the 24th annual ACM symposium on User interface software and technology.

## 七、应用前景与扩展

### 7.1 应用场景

**科学研究**：自动化处理实验数据图表，加速科研进程
**工业检测**：生产线质量监控图表分析
**金融分析**：技术分析图表数据提取
**教育领域**：试题图表数据自动化处理

### 7.2 技术扩展方向

**1. 支持更多图表类型**
- 三维图表
- 极坐标图表  
- 对数坐标图表

**2. 智能图表理解**
- 自动识别图表类型
- 理解图表语义信息
- 生成图表描述

**3. 实时处理能力**
- 流式图像处理
- 在线学习适应新图表风格
- 云端部署与API服务

## 八、总结

本系统通过深度融合计算机视觉、深度学习和传统图像处理技术，实现了科学图像数据的高效自动化提取。系统在算法设计上具有多项创新，在工程实现上考虑了实用性、稳定性和扩展性。通过模块化设计和多线程优化，系统能够满足大规模科学数据处理的需求，为科研工作者提供了强有力的工具支持。

系统的成功开发不仅解决了科学数据提取的实际问题，也为相关领域的技术发展提供了有价值的参考。随着人工智能技术的不断进步，该系统有望在更多领域发挥重要作用，推动科学研究的数字化、智能化进程。
