# main.py
import json
import os
import pandas as pd
import threading
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from axies_API import AxesExtractor
from curve_API import CurveAnalysisAPI

# 线程锁，用于保护共享资源
file_lock = threading.Lock()
data_lock = threading.Lock()

def natural_sort_key(filename):
    """
    自然排序键函数，按数字大小排序而不是字符串排序
    例如: 1.jpg, 2.jpg, 10.jpg 而不是 1.jpg, 10.jpg, 2.jpg
    """
    # 使用正则表达式提取文件名中的数字部分
    numbers = re.findall(r'\d+', filename)
    if numbers:
        # 返回数字的整数值用于排序
        return int(numbers[0])
    else:
        # 如果没有数字，返回原文件名
        return filename

def safe_format(value, format_spec=".2f"):
    """
    安全格式化函数，处理None值
    
    参数:
        value: 要格式化的值
        format_spec: 格式说明符
        
    返回:
        格式化后的字符串
    """
    if value is None:
        return "None"
    try:
        return format(value, format_spec)
    except (TypeError, ValueError):
        return str(value)

def format_coordinate_value(value):
    """
    格式化坐标值，与extractor.xlsx保持一致
    使用科学计数法格式，保持与参考文件相同的精度
    """
    if value is None:
        return "0"
    try:
        # 对于小数值使用科学计数法，保持与extractor.xlsx相同的格式
        if abs(value) < 0.01 and value != 0:
            return "{:.5E}".format(value).replace('E-0', 'E-').replace('E+0', 'E+').replace('E-', 'E-')
        else:
            return "{:.5f}".format(value).rstrip('0').rstrip('.')
    except (TypeError, ValueError):
        return str(value)

def process_single_image(image_path, image_index, total_images):
    """
    处理单张图像，提取坐标轴信息和曲线数据，并进行坐标转换
    
    参数:
        image_path: 图像文件路径
        image_index: 图像索引
        total_images: 总图像数量
        
    返回:
        处理结果字典，包含Excel数据和其他信息
    """
    thread_name = threading.current_thread().name
    print("=" * 60)
    print(f"线程 {thread_name}: 开始处理图像 ({image_index}/{total_images}): {os.path.basename(image_path)}")
    print("=" * 60)
    
    # 验证图像文件存在
    if not os.path.exists(image_path):
        print(f"线程 {thread_name}: 错误: 图像文件不存在: {image_path}")
        return None
    
    try:
        # 步骤1: 提取坐标轴信息
        print(f"\n线程 {thread_name}: 步骤1: 提取坐标轴信息...")
        axes_extractor = AxesExtractor(debug=False)
        axes_info = axes_extractor.extract_axes_info(image_path)
        
        if not axes_info['success']:
            print(f"线程 {thread_name}: 坐标轴提取失败: {axes_info['message']}")
            return None
        
        # 获取格式化后的坐标轴参数
        axes_params = axes_extractor.get_formatted_results(axes_info)
        
        # 提取关键参数，使用安全默认值
        x_left = axes_params['x_axis']['limits']['left'] if axes_params['x_axis']['limits']['left'] is not None else 0
        x_right = axes_params['x_axis']['limits']['right'] if axes_params['x_axis']['limits']['right'] is not None else 0
        x_title = axes_params['x_axis']['title'] if axes_params['x_axis']['title'] is not None else "未识别"
        x_pixels_per_value = axes_params['x_axis']['scale']['pixels_per_value'] if axes_params['x_axis']['scale']['pixels_per_value'] is not None else 0
        
        y_bottom = axes_params['y_axis']['limits']['bottom'] if axes_params['y_axis']['limits']['bottom'] is not None else 0
        y_top = axes_params['y_axis']['limits']['top'] if axes_params['y_axis']['limits']['top'] is not None else 0
        y_title = axes_params['y_axis']['title'] if axes_params['y_axis']['title'] is not None else "未识别"
        y_pixels_per_value = axes_params['y_axis']['scale']['pixels_per_value'] if axes_params['y_axis']['scale']['pixels_per_value'] is not None else 0
        
        print(f"线程 {thread_name}: 坐标轴信息提取完成:")
        print(f"  线程 {thread_name}: X轴: 左极限={safe_format(x_left)}, 右极限={safe_format(x_right)}, 标题='{x_title}', 刻度={safe_format(x_pixels_per_value, '.6f')} 单位/像素")
        print(f"  线程 {thread_name}: Y轴: 下极限={safe_format(y_bottom)}, 上极限={safe_format(y_top)}, 标题='{y_title}', 刻度={safe_format(y_pixels_per_value, '.6f')} 单位/像素")
        
        # 检查必要的坐标轴参数是否有效
        if x_pixels_per_value == 0 or y_pixels_per_value == 0:
            print(f"线程 {thread_name}: 警告: 坐标轴刻度值为0，可能影响坐标转换精度")
        
        # 步骤2: 提取曲线数据
        print(f"\n线程 {thread_name}: 步骤2: 提取曲线数据...")
        curve_api = CurveAnalysisAPI()
        curve_results = curve_api.analyze_image(
            image_path=image_path,
            yolo_threshold=0.3,
            curve_points=128,
            color_tolerance=40,
            visualize=False
        )
        
        if 'error' in curve_results:
            print(f"线程 {thread_name}: 曲线提取失败: {curve_results['error']}")
            return None
        
        print(f"线程 {thread_name}: 曲线数据提取完成，找到 {curve_results['total_legends']} 条曲线")
        
        # 步骤3: 坐标转换
        print(f"\n线程 {thread_name}: 步骤3: 进行坐标转换...")
        transformed_curves = []
        
        for curve in curve_results['curves']:
            legend_name = curve['legend_name']
            original_points = curve['points']
            
            print(f"  线程 {thread_name}: 处理图例 '{legend_name}'，原始点数: {len(original_points)}")
            
            # 坐标转换
            transformed_points = []
            for point in original_points:
                try:
                    # 严格按照指定的公式进行坐标转换
                    # 横坐标转换: (点的横坐标 - x坐标轴左极限) × x方向每像素对应的刻度值
                    new_x = (point[0] - x_left) * x_pixels_per_value
                    
                    # 纵坐标转换: (点的纵坐标 - y坐标轴底部极限值) × y方向每像素对应的刻度值  
                    new_y = (point[1] - y_bottom) * y_pixels_per_value
                    
                    transformed_points.append([float(new_x), float(new_y)])
                except Exception as e:
                    print(f"    线程 {thread_name}: 坐标转换错误: {e}, 跳过该点")
                    continue
            
            if transformed_points:
                transformed_curve = {
                    'legend_name': legend_name,
                    'original_points_count': len(original_points),
                    'transformed_points': transformed_points
                }
                
                transformed_curves.append(transformed_curve)
                print(f"    线程 {thread_name}: 坐标转换完成，转换后点数: {len(transformed_points)}")
            else:
                print(f"    线程 {thread_name}: 警告: 图例 '{legend_name}' 没有成功转换的点")
        
        if not transformed_curves:
            print(f"线程 {thread_name}: 警告: 没有成功转换任何曲线数据")
            return None
        
        # 步骤4: 构建Excel数据 - 修改为与extractor.xlsx完全一致的格式
        print(f"\n线程 {thread_name}: 步骤4: 构建Excel数据...")
        excel_data = []
        
        image_name = os.path.basename(image_path)
        image_id = os.path.splitext(image_name)[0]
        
        for curve in transformed_curves:
            # 将坐标点列表转换为与extractor.xlsx相同的格式: x值\ty值\n
            points_lines = []
            for point in curve['transformed_points']:
                x_str = format_coordinate_value(point[0])
                y_str = format_coordinate_value(point[1])
                points_lines.append(f"{x_str}\t{y_str}")
            
            points_str = '<br>'.join(points_lines)  # 使用<br>作为换行符，与extractor.xlsx一致
            
            # 只保留必要的列，与extractor.xlsx结构完全一致
            row_data = {
                'DOI': image_name,
                'figure_index': '',  # 空着不填
                'figure_title': '',  # 空着不填
                'X-label': x_title,
                'Y-label': y_title,
                'sample': curve['legend_name'],
                'Values': points_str,
                'note': ''  # 空着不填
            }
            
            excel_data.append(row_data)
        
        print(f"线程 {thread_name}: 图像 {image_name} 处理完成，生成 {len(excel_data)} 行数据")
        
        return {
            'excel_data': excel_data,
            'image_name': image_name,
            'image_path': image_path,
            'x_title': x_title,
            'y_title': y_title,
            'curves_count': len(transformed_curves),
            'image_id': image_id,
            'image_index': image_index
        }
    
    except Exception as e:
        print(f"线程 {thread_name}: 处理图像时发生异常: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_all_images(figs_folder="figs", output_file="extractor.xlsx", max_workers=8):
    """
    处理figs文件夹中的所有图片，并将结果保存到同一个Excel文件
    
    参数:
        figs_folder: 包含图片的文件夹路径
        output_file: 输出Excel文件名
        max_workers: 最大线程数，None表示使用CPU核心数
    """
    # 检查figs文件夹是否存在
    if not os.path.exists(figs_folder):
        print(f"错误: 文件夹 '{figs_folder}' 不存在")
        return
    
    # 支持的图片格式
    supported_formats = {'.jpg', '.jpeg', '.tif', '.tiff', '.png'}
    
    # 获取所有图片文件
    image_files = []
    for file in os.listdir(figs_folder):
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext in supported_formats:
            image_files.append(os.path.join(figs_folder, file))
    
    # 使用自然排序而不是字符串排序
    image_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    if not image_files:
        print(f"在文件夹 '{figs_folder}' 中没有找到支持的图片文件")
        print(f"支持的格式: {', '.join(supported_formats)}")
        return
    
    print(f"找到 {len(image_files)} 个图片文件 (按数字顺序排序):")
    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")
    
    # 使用线程池处理所有图片
    all_excel_data = []
    processed_results = []
    failed_images = []
    
    print(f"\n{'='*80}")
    print(f"开始多线程处理，使用 {max_workers if max_workers else '默认'} 个线程")
    print(f"{'='*80}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_image = {
            executor.submit(process_single_image, image_path, i+1, len(image_files)): image_path 
            for i, image_path in enumerate(image_files)
        }
        
        # 收集结果
        for future in as_completed(future_to_image):
            image_path = future_to_image[future]
            image_name = os.path.basename(image_path)
            
            try:
                result = future.result()
                if result:
                    # 使用线程锁保护共享数据
                    with data_lock:
                        all_excel_data.extend(result['excel_data'])
                        processed_results.append(result)
                    print(f"✓ 成功处理: {image_name}")
                else:
                    with data_lock:
                        failed_images.append(image_name)
                    print(f"✗ 处理失败: {image_name}")
            except Exception as e:
                with data_lock:
                    failed_images.append(image_name)
                print(f"✗ 处理异常: {image_name} - {e}")
    
    # 按照自然排序顺序对处理结果进行排序
    processed_results.sort(key=lambda x: natural_sort_key(x['image_name']))
    all_excel_data_sorted = []
    
    # 重新构建排序后的Excel数据
    for result in processed_results:
        all_excel_data_sorted.extend(result['excel_data'])
    
    if not all_excel_data_sorted:
        print("没有成功处理任何图片")
        return
    
    # 创建DataFrame并保存为Excel - 移除多余的格式设置
    print(f"\n{'='*80}")
    print("保存所有结果到Excel文件...")
    print(f"{'='*80}")
    
    try:
        # 创建与extractor.xlsx完全一致的DataFrame结构
        df = pd.DataFrame(all_excel_data_sorted, columns=[
            'DOI', 'figure_index', 'figure_title', 
            'X-label', 'Y-label', 'sample', 'Values', 'note'
        ])
        
        # 简化的Excel保存，不添加额外格式
        df.to_excel(output_file, sheet_name='Sheet1', index=False)
        print(f"✓ Excel文件保存成功: {output_file}")
        
    except Exception as e:
        print(f"✗ 保存Excel文件失败: {e}")
        return
    
    # 显示处理统计信息
    print_processing_summary(processed_results, failed_images)
    
    return {
        'all_excel_data': all_excel_data_sorted,
        'processed_results': processed_results,
        'failed_images': failed_images,
        'total_images': len(processed_results),
        'total_curves': len(all_excel_data_sorted)
    }

def print_processing_summary(processed_results, failed_images):
    """
    打印处理汇总信息
    
    参数:
        processed_results: 处理结果列表
        failed_images: 处理失败的图片列表
    """
    print("\n" + "=" * 80)
    print("处理汇总信息")
    print("=" * 80)
    
    print(f"成功处理的图片数量: {len(processed_results)}")
    
    if failed_images:
        print(f"处理失败的图片数量: {len(failed_images)}")
        print("失败的图片:")
        for img in failed_images:
            print(f"  - {img}")
    
    total_curves = sum(result['curves_count'] for result in processed_results)
    print(f"总曲线数量: {total_curves}")
    
    if processed_results:
        print("\n各图片处理详情:")
        print("-" * 100)
        print(f"{'图片名称':<30} {'X轴标题':<20} {'Y轴标题':<20} {'曲线数量':<10}")
        print("-" * 100)
        
        for result in processed_results:
            print(f"{result['image_name']:<30} {result['x_title']:<20} {result['y_title']:<20} {result['curves_count']:<10}")

if __name__ == "__main__":
    try:
        # 处理figs文件夹中的所有图片
        result = process_all_images(figs_folder="figs", output_file="extractor.xlsx", max_workers=8)
            
    except Exception as e:
        print(f"程序执行过程中出错: {e}")
        import traceback
        traceback.print_exc()