# main.py
import json
import os
import pandas as pd
import re
from axies_API import AxesExtractor
from curve_API_canary import CurveAnalysisAPI

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

def parse_filename_components(filename):
    """
    解析文件名，提取DOI、Figure和Sub-figure信息
    
    参数:
        filename: 文件名（带或不带扩展名）
        
    返回:
        tuple: (doi, figure, sub_figure)
    """
    # 移除文件扩展名
    name_without_ext = os.path.splitext(filename)[0]
    
    # 按"_"分割文件名
    parts = name_without_ext.split('_')
    
    # 提取DOI（第一个"_"之前的内容）
    doi = parts[0] if len(parts) > 0 else ""
    
    # 提取Figure（第一个"_"到第二个"_"之间的内容）
    figure = parts[1] if len(parts) > 1 else "/"
    
    # 提取Sub-figure（第二个"_"到文件扩展名之前的内容）
    sub_figure = parts[2] if len(parts) > 2 else "/"
    
    # 如果Sub-figure为空，则设置为"/"
    if sub_figure == "":
        sub_figure = "/"
    
    return doi, figure, sub_figure

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

def process_single_image(image_path):
    """
    处理单张图像，提取坐标轴信息和曲线数据，并进行坐标转换
    
    参数:
        image_path: 图像文件路径
        
    返回:
        处理结果字典，包含Excel数据和其他信息
    """
    print("=" * 60)
    print("开始处理图像:", image_path)
    print("=" * 60)
    
    # 验证图像文件存在
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        return None
    
    # 步骤1: 提取坐标轴信息
    print("\n步骤1: 提取坐标轴信息...")
    axes_extractor = AxesExtractor(debug=False)
    axes_info = axes_extractor.extract_axes_info(image_path)
    
    if not axes_info['success']:
        print(f"坐标轴提取失败: {axes_info['message']}")
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
    
    print("坐标轴信息提取完成:")
    print(f"  X轴: 左极限={safe_format(x_left)}, 右极限={safe_format(x_right)}, 标题='{x_title}', 刻度={safe_format(x_pixels_per_value, '.6f')} 单位/像素")
    print(f"  Y轴: 下极限={safe_format(y_bottom)}, 上极限={safe_format(y_top)}, 标题='{y_title}', 刻度={safe_format(y_pixels_per_value, '.6f')} 单位/像素")
    
    # 检查必要的坐标轴参数是否有效
    if x_pixels_per_value == 0 or y_pixels_per_value == 0:
        print("警告: 坐标轴刻度值为0，可能影响坐标转换精度")
    
    # 步骤2: 提取曲线数据
    print("\n步骤2: 提取曲线数据...")
    curve_api = CurveAnalysisAPI()
    curve_results = curve_api.analyze_image(
        image_path=image_path,
        yolo_threshold=0.3,
        curve_points=128,
        color_tolerance=40,
        visualize=True
    )
    
    if 'error' in curve_results:
        print(f"曲线提取失败: {curve_results['error']}")
        return None
    
    print(f"曲线数据提取完成，找到 {curve_results['total_legends']} 条曲线")
    
    # 步骤3: 坐标转换
    print("\n步骤3: 进行坐标转换...")
    transformed_curves = []
    
    for curve in curve_results['curves']:
        legend_name = curve['legend_name']
        original_points = curve['points']
        
        print(f"  处理图例 '{legend_name}'，原始点数: {len(original_points)}")
        
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
                print(f"    坐标转换错误: {e}, 跳过该点")
                continue
        
        if transformed_points:
            transformed_curve = {
                'legend_name': legend_name,
                'original_points_count': len(original_points),
                'transformed_points': transformed_points
            }
            
            transformed_curves.append(transformed_curve)
            print(f"    坐标转换完成，转换后点数: {len(transformed_points)}")
        else:
            print(f"    警告: 图例 '{legend_name}' 没有成功转换的点")
    
    if not transformed_curves:
        print("警告: 没有成功转换任何曲线数据")
        return None
    
    # 步骤4: 构建Excel数据
    print("\n步骤4: 构建Excel数据...")
    excel_data = []
    
    image_name = os.path.basename(image_path)
    
    # 解析文件名组件
    doi, figure, sub_figure = parse_filename_components(image_name)
    
    print(f"文件名解析结果: DOI='{doi}', Figure='{figure}', Sub-figure='{sub_figure}'")
    
    for curve in transformed_curves:
        # 将坐标点列表转换为与extractor.xlsx相同的格式
        points_str = format_coordinates_for_excel(curve['transformed_points'])
        
        row_data = {
            'DOI': doi,  # 只显示文件名中第一个"_"之前的内容
            'Figure': figure,  # 显示第一个"_"到第二个"_"之间的内容
            'Sub-figure': sub_figure,  # 显示第二个"_"到文件扩展名之前的内容，如果没有则显示"/"
            'X-label': x_title,
            'Y-label': y_title,
            'Legend': curve['legend_name'],
            'Values': points_str,
            'note': ''  # 空着不填
        }
        
        excel_data.append(row_data)
    
    print(f"图像 {image_name} 处理完成，生成 {len(excel_data)} 行数据")
    
    return {
        'excel_data': excel_data,
        'image_name': image_name,
        'x_title': x_title,
        'y_title': y_title,
        'curves_count': len(transformed_curves),
        'processing_log': f"图像: {image_name}\nX轴: {x_title}, Y轴: {y_title}\n曲线数量: {len(transformed_curves)}\n"
    }

def process_all_images(figs_folder="figs", output_file="extractor.xlsx"):
    """
    处理figs文件夹中的所有图片，并将结果保存到同一个Excel文件
    
    参数:
        figs_folder: 包含图片的文件夹路径
        output_file: 输出Excel文件名
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
    
    # 处理所有图片并收集数据
    all_excel_data = []
    processed_results = []
    failed_images = []
    extra_content = "图像处理详细信息\n" + "=" * 50 + "\n\n"
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n{'='*80}")
        print(f"处理第 {i}/{len(image_files)} 个图片: {os.path.basename(image_path)}")
        print(f"{'='*80}")
        
        try:
            result = process_single_image(image_path)
            if result:
                all_excel_data.extend(result['excel_data'])
                processed_results.append(result)
                extra_content += result['processing_log'] + "\n"
                print(f"✓ 成功处理: {os.path.basename(image_path)}")
            else:
                failed_images.append(os.path.basename(image_path))
                extra_content += f"处理失败: {os.path.basename(image_path)}\n\n"
                print(f"✗ 处理失败: {os.path.basename(image_path)}")
        except Exception as e:
            failed_images.append(os.path.basename(image_path))
            extra_content += f"处理异常: {os.path.basename(image_path)} - {e}\n\n"
            print(f"✗ 处理异常: {os.path.basename(image_path)} - {e}")
        
        print(f"\n完成第 {i}/{len(image_files)} 个图片的处理")
    
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
            worksheet = writer.sheets['Sheet1']
            
            # 设置与示例文件完全一致的列宽
            # A列（DOI）：约40字符
            worksheet.column_dimensions['A'].width = 40
            # # B列（Figure）：约10字符
            # worksheet.column_dimensions['B'].width = 10
            # # C列（Sub-figure）：约12字符
            # worksheet.column_dimensions['C'].width = 12
            # # D列（X-label）：约20字符
            # worksheet.column_dimensions['D'].width = 20
            # # E列（Y-label）：约20字符
            # worksheet.column_dimensions['E'].width = 20
            # # F列（Legend）：约10字符
            # worksheet.column_dimensions['F'].width = 10
            # # G列（Values）：约50字符（需要更宽以容纳坐标数据）
            # worksheet.column_dimensions['G'].width = 50
            # # H列（note）：约10字符
            # worksheet.column_dimensions['H'].width = 10
            
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
    extra_content += f"总曲线数量: {sum(result['curves_count'] for result in processed_results)}\n"
    
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
        'processed_results': processed_results,
        'failed_images': failed_images,
        'total_images': len(processed_results),
        'total_curves': len(all_excel_data)
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
        # # 处理figs文件夹中的所有图片
        result = process_all_images(figs_folder="fig2", output_file="extractor.xlsx")
        #处理根目录下的单张图片
        #result = process_all_images(figs_folder="./", output_file="extractor.xlsx")

        
    except Exception as e:
        print(f"程序执行过程中出错: {e}")
        import traceback
        traceback.print_exc()