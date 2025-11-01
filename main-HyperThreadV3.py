#V6：引入异步容器式多线程并发设计
# main.py
import json
import os
import pandas as pd
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from axies_API import AxesExtractor
from curve_API import CurveAnalysisAPI

# 线程本地存储，避免数据冲突
thread_local = threading.local()

def natural_sort_key(s):
    """
    自然排序键函数，用于按数字顺序排序文件名
    
    参数:
        s: 字符串
        
    返回:
        排序键
    """
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]

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

def format_coordinates_for_excel(points):
    """
    将坐标点格式化为与extractor.xlsx相同的格式
    使用制表符分隔x和y值，换行分隔不同点
    
    参数:
        points: 坐标点列表 [[x1, y1], [x2, y2], ...]
        
    返回:
        格式化后的坐标字符串
    """
    formatted_points = []
    for point in points:
        x, y = point
        # 使用制表符分隔x和y值
        formatted_point = f"{x}\t{y}"
        formatted_points.append(formatted_point)
    
    # 使用换行符连接所有点
    return '\n'.join(formatted_points)

def write_extra_content_to_txt(extra_content, output_file="extra_content.txt"):
    """
    将多余内容写入txt文件
    
    参数:
        extra_content: 要写入的额外内容
        output_file: 输出txt文件名
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(extra_content)
        print(f"✓ 额外内容已保存到: {output_file}")
    except Exception as e:
        print(f"✗ 保存额外内容失败: {e}")

def get_thread_local_extractor():
    """
    获取线程本地的坐标轴提取器实例，避免多线程冲突
    
    返回:
        AxesExtractor实例
    """
    if not hasattr(thread_local, 'axes_extractor'):
        thread_local.axes_extractor = AxesExtractor(debug=False)
    return thread_local.axes_extractor

def get_thread_local_curve_api():
    """
    获取线程本地的曲线分析API实例，避免多线程冲突
    
    返回:
        CurveAnalysisAPI实例
    """
    if not hasattr(thread_local, 'curve_api'):
        thread_local.curve_api = CurveAnalysisAPI()
    return thread_local.curve_api

def process_single_image_thread_safe(image_path, image_index, total_images):
    """
    线程安全的单张图像处理函数
    
    参数:
        image_path: 图像文件路径
        image_index: 图像索引
        total_images: 总图像数量
        
    返回:
        处理结果字典，包含Excel数据和其他信息
    """
    thread_id = threading.get_ident()
    print(f"[线程{thread_id}] 开始处理图像 ({image_index}/{total_images}): {os.path.basename(image_path)}")
    
    # 验证图像文件存在
    if not os.path.exists(image_path):
        error_msg = f"[线程{thread_id}] 错误: 图像文件不存在: {image_path}"
        print(error_msg)
        return {'error': error_msg, 'image_path': image_path}
    
    try:
        # 步骤1: 提取坐标轴信息（使用线程本地实例）
        axes_extractor = get_thread_local_extractor()
        axes_info = axes_extractor.extract_axes_info(image_path)
        
        if not axes_info['success']:
            error_msg = f"[线程{thread_id}] 坐标轴提取失败: {axes_info['message']}"
            print(error_msg)
            return {'error': error_msg, 'image_path': image_path}
        
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
        
        print(f"[线程{thread_id}] 坐标轴信息提取完成:")
        print(f"[线程{thread_id}]   X轴: 左极限={safe_format(x_left)}, 右极限={safe_format(x_right)}, 标题='{x_title}'")
        print(f"[线程{thread_id}]   Y轴: 下极限={safe_format(y_bottom)}, 上极限={safe_format(y_top)}, 标题='{y_title}'")
        
        # 检查必要的坐标轴参数是否有效
        if x_pixels_per_value == 0 or y_pixels_per_value == 0:
            print(f"[线程{thread_id}] 警告: 坐标轴刻度值为0，可能影响坐标转换精度")
        
        # 步骤2: 提取曲线数据（使用线程本地实例）
        curve_api = get_thread_local_curve_api()
        curve_results = curve_api.analyze_image(
            image_path=image_path,
            yolo_threshold=0.3,
            curve_points=128,
            color_tolerance=40,
            visualize=False
        )
        
        if 'error' in curve_results:
            error_msg = f"[线程{thread_id}] 曲线提取失败: {curve_results['error']}"
            print(error_msg)
            return {'error': error_msg, 'image_path': image_path}
        
        print(f"[线程{thread_id}] 曲线数据提取完成，找到 {curve_results['total_legends']} 条曲线")
        
        # 步骤3: 坐标转换
        transformed_curves = []
        
        for curve in curve_results['curves']:
            legend_name = curve['legend_name']
            original_points = curve['points']
            
            print(f"[线程{thread_id}]   处理图例 '{legend_name}'，原始点数: {len(original_points)}")
            
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
                    print(f"[线程{thread_id}]     坐标转换错误: {e}, 跳过该点")
                    continue
            
            if transformed_points:
                transformed_curve = {
                    'legend_name': legend_name,
                    'original_points_count': len(original_points),
                    'transformed_points': transformed_points
                }
                
                transformed_curves.append(transformed_curve)
                print(f"[线程{thread_id}]     坐标转换完成，转换后点数: {len(transformed_points)}")
            else:
                print(f"[线程{thread_id}]     警告: 图例 '{legend_name}' 没有成功转换的点")
        
        if not transformed_curves:
            error_msg = f"[线程{thread_id}] 警告: 没有成功转换任何曲线数据"
            print(error_msg)
            return {'error': error_msg, 'image_path': image_path}
        
        # 步骤4: 构建Excel数据
        excel_data = []
        
        image_name = os.path.basename(image_path)
        
        for curve in transformed_curves:
            # 将坐标点列表转换为与extractor.xlsx相同的格式
            points_str = format_coordinates_for_excel(curve['transformed_points'])
            
            row_data = {
                'DOI': image_name,  # 只保留文件名，去掉(id)
                'figure_index': '',  # 空着不填
                'figure_title': '',  # 空着不填
                'X-label': x_title,
                'Y-label': y_title,
                'sample': curve['legend_name'],
                'Values': points_str,
                'note': ''  # 空着不填
            }
            
            excel_data.append(row_data)
        
        print(f"[线程{thread_id}] 图像 {image_name} 处理完成，生成 {len(excel_data)} 行数据")
        
        return {
            'excel_data': excel_data,
            'image_name': image_name,
            'image_path': image_path,
            'x_title': x_title,
            'y_title': y_title,
            'curves_count': len(transformed_curves),
            'processing_log': f"图像: {image_name}\nX轴: {x_title}, Y轴: {y_title}\n曲线数量: {len(transformed_curves)}\n",
            'success': True
        }
    
    except Exception as e:
        error_msg = f"[线程{thread_id}] 处理图像时发生异常: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {'error': error_msg, 'image_path': image_path}

def process_all_images(figs_folder="figs", output_file="extractor.xlsx", max_workers=None):
    """
    使用多线程并发处理figs文件夹中的所有图片，并将结果保存到同一个Excel文件
    
    参数:
        figs_folder: 包含图片的文件夹路径
        output_file: 输出Excel文件名
        max_workers: 最大线程数，None表示自动设置
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
    
    # 使用自然排序按数字顺序排序
    image_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    if not image_files:
        print(f"在文件夹 '{figs_folder}' 中没有找到支持的图片文件")
        print(f"支持的格式: {', '.join(supported_formats)}")
        return
    
    print(f"找到 {len(image_files)} 个图片文件 (按数字顺序排序):")
    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")
    
    print(f"\n开始多线程并发处理，最大线程数: {max_workers or '自动'}")
    
    # 使用线程池并发处理所有图片
    all_excel_data = []
    processed_results = []
    failed_images = []
    extra_content = "图像处理详细信息\n" + "=" * 50 + "\n\n"
    
    # 创建任务字典，用于保持原始顺序
    tasks = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        for i, image_path in enumerate(image_files, 1):
            future = executor.submit(process_single_image_thread_safe, image_path, i, len(image_files))
            tasks[future] = image_path
        
        # 按照任务完成顺序收集结果
        completed_count = 0
        for future in as_completed(tasks):
            image_path = tasks[future]
            image_name = os.path.basename(image_path)
            
            try:
                result = future.result()
                completed_count += 1
                
                if result.get('success', False):
                    # 保存成功结果，但先不添加到最终列表（后面会按原始顺序排序）
                    processed_results.append({
                        'result': result,
                        'original_index': image_files.index(image_path),
                        'image_name': image_name
                    })
                    print(f"[主线程] ✓ 完成 ({completed_count}/{len(image_files)}): {image_name}")
                else:
                    failed_images.append(image_name)
                    error_msg = result.get('error', '未知错误')
                    extra_content += f"处理失败: {image_name} - {error_msg}\n\n"
                    print(f"[主线程] ✗ 失败 ({completed_count}/{len(image_files)}): {image_name} - {error_msg}")
                    
            except Exception as e:
                completed_count += 1
                failed_images.append(image_name)
                error_msg = f"任务执行异常: {e}"
                extra_content += f"处理异常: {image_name} - {error_msg}\n\n"
                print(f"[主线程] ✗ 异常 ({completed_count}/{len(image_files)}): {image_name} - {error_msg}")
    
    # 按照原始自然排序顺序重新排序处理结果
    processed_results.sort(key=lambda x: x['original_index'])
    
    # 按照正确顺序构建最终数据
    for item in processed_results:
        result = item['result']
        all_excel_data.extend(result['excel_data'])
        extra_content += result['processing_log'] + "\n"
    
    if not all_excel_data:
        print("没有成功处理任何图片")
        return
    
    # 创建DataFrame并保存为Excel
    print(f"\n{'='*80}")
    print("保存所有结果到Excel文件...")
    print(f"{'='*80}")
    
    try:
        df = pd.DataFrame(all_excel_data)
        
        # 设置Excel写入器，确保格式与extractor.xlsx一致
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
            
            # 获取工作表
            #worksheet = writer.sheets['Sheet1']
            
            # 不再设置固定列宽，让Excel自动调整
            
            # # 设置单元格格式为文本，确保坐标显示正确
            # for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row, min_col=7, max_col=7):
            #     for cell in row:
            #         cell.number_format = '@'  # 文本格式
        
        print(f"✓ Excel文件保存成功: {output_file}")
        
    except Exception as e:
        print(f"✗ 保存Excel文件失败: {e}")
        return
    
    # 添加处理统计信息到额外内容
    extra_content += "\n" + "=" * 50 + "\n"
    extra_content += "处理汇总信息\n"
    extra_content += f"成功处理的图片数量: {len(processed_results)}\n"
    extra_content += f"处理失败的图片数量: {len(failed_images)}\n"
    extra_content += f"总曲线数量: {sum(item['result']['curves_count'] for item in processed_results)}\n"
    
    if failed_images:
        extra_content += "\n失败的图片:\n"
        for img in failed_images:
            extra_content += f"  - {img}\n"
    
    # 将额外内容保存到txt文件
    write_extra_content_to_txt(extra_content, "processing_details.txt")
    
    # 显示处理统计信息
    print_processing_summary(processed_results, failed_images)
    
    return {
        'all_excel_data': all_excel_data,
        'processed_results': [item['result'] for item in processed_results],
        'failed_images': failed_images,
        'total_images': len(processed_results),
        'total_curves': len(all_excel_data)
    }

def print_processing_summary(processed_results, failed_images):
    """
    打印处理汇总信息
    
    参数:
        processed_results: 处理结果列表（已排序）
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
    
    total_curves = sum(item['result']['curves_count'] for item in processed_results)
    print(f"总曲线数量: {total_curves}")
    
    if processed_results:
        print("\n各图片处理详情 (按自然文件名排序):")
        print("-" * 100)
        print(f"{'图片名称':<30} {'X轴标题':<20} {'Y轴标题':<20} {'曲线数量':<10}")
        print("-" * 100)
        
        for item in processed_results:
            result = item['result']
            print(f"{result['image_name']:<30} {result['x_title']:<20} {result['y_title']:<20} {result['curves_count']:<10}")

if __name__ == "__main__":
    try:
        # 处理figs文件夹中的所有图片，使用多线程并发
        # 可以设置max_workers参数来控制并发线程数，None表示自动设置
        result = process_all_images(figs_folder="figs", output_file="extractor.xlsx", max_workers=8)
        
    except Exception as e:
        print(f"程序执行过程中出错: {e}")
        import traceback
        traceback.print_exc()