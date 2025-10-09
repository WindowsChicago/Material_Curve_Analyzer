# main.py
import json
import os
from axies_API import AxesExtractor
from curve_API import CurveAnalysisAPI

def main(image_path):
    """
    主函数：处理图像，提取坐标轴信息和曲线数据，并进行坐标转换
    
    参数:
        image_path: 图像文件路径
    """
    print("=" * 60)
    print("开始处理图像:", image_path)
    print("=" * 60)
    
    # 验证图像文件存在
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        return
    
    # 步骤1: 提取坐标轴信息
    print("\n步骤1: 提取坐标轴信息...")
    axes_extractor = AxesExtractor(debug=False)
    axes_info = axes_extractor.extract_axes_info(image_path)
    
    if not axes_info['success']:
        print(f"坐标轴提取失败: {axes_info['message']}")
        return
    
    # 获取格式化后的坐标轴参数
    axes_params = axes_extractor.get_formatted_results(axes_info)
    
    # 提取关键参数
    x_left = axes_params['x_axis']['limits']['left']
    x_right = axes_params['x_axis']['limits']['right']
    x_title = axes_params['x_axis']['title']
    x_pixels_per_value = axes_params['x_axis']['scale']['pixels_per_value']
    
    y_bottom = axes_params['y_axis']['limits']['bottom']
    y_top = axes_params['y_axis']['limits']['top']
    y_title = axes_params['y_axis']['title']
    y_pixels_per_value = axes_params['y_axis']['scale']['pixels_per_value']
    
    print("坐标轴信息提取完成:")
    print(f"  X轴: 左极限={x_left:.2f}, 右极限={x_right:.2f}, 标题='{x_title}', 刻度={x_pixels_per_value:.6f} 单位/像素")
    print(f"  Y轴: 下极限={y_bottom:.2f}, 上极限={y_top:.2f}, 标题='{y_title}', 刻度={y_pixels_per_value:.6f} 单位/像素")
    
    # 步骤2: 提取曲线数据
    print("\n步骤2: 提取曲线数据...")
    curve_api = CurveAnalysisAPI()
    curve_results = curve_api.analyze_image(
        image_path=image_path,
        yolo_threshold=0.3,
        curve_points=128,
        color_tolerance=40,
        visualize=False
    )
    
    if 'error' in curve_results:
        print(f"曲线提取失败: {curve_results['error']}")
        return
    
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
            # 严格按照指定的公式进行坐标转换
            # 横坐标转换: (点的横坐标 - x坐标轴左极限) × x方向每像素对应的刻度值
            new_x = (point[0] - x_left) * x_pixels_per_value
            
            # 纵坐标转换: (点的纵坐标 - y坐标轴底部极限值) × y方向每像素对应的刻度值  
            new_y = (point[1] - y_bottom) * y_pixels_per_value
            
            transformed_points.append([float(new_x), float(new_y)])
        
        transformed_curve = {
            'legend_name': legend_name,
            'original_points_count': len(original_points),
            'transformed_points': transformed_points
        }
        
        transformed_curves.append(transformed_curve)
        print(f"    坐标转换完成，转换后点数: {len(transformed_points)}")
    
    # 步骤4: 构建最终结果
    print("\n步骤4: 构建最终结果...")
    final_result = {
        'image_name': os.path.basename(image_path),
        'x_axis_title': x_title,
        'y_axis_title': y_title,
        'coordinate_parameters': {
            'x_left_limit': float(x_left),
            'x_right_limit': float(x_right),
            'x_pixels_per_value': float(x_pixels_per_value),
            'y_bottom_limit': float(y_bottom),
            'y_top_limit': float(y_top),
            'y_pixels_per_value': float(y_pixels_per_value)
        },
        'curves': transformed_curves
    }
    
    # 步骤5: 输出结果
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    
    print(f"图像名称: {final_result['image_name']}")
    print(f"X轴标题: {final_result['x_axis_title']}")
    print(f"Y轴标题: {final_result['y_axis_title']}")
    print(f"曲线数量: {len(final_result['curves'])}")
    
    for i, curve in enumerate(final_result['curves']):
        print(f"\n曲线 {i+1}:")
        print(f"  图例名称: {curve['legend_name']}")
        print(f"  转换后点数: {len(curve['transformed_points'])}")
        print(f"  前5个转换后的坐标点:")
        
        for j, point in enumerate(curve['transformed_points'][:5]):
            print(f"    点 {j+1}: ({point[0]:.4f}, {point[1]:.4f})")
        
        if len(curve['transformed_points']) > 5:
            print(f"    ... (显示前5个，共{len(curve['transformed_points'])}个点)")
    
    # 保存结果到JSON文件
    output_filename = os.path.splitext(final_result['image_name'])[0] + '_result.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n完整结果已保存到: {output_filename}")
    
    return final_result

def print_detailed_coordinates(result, max_points=10):
    """
    打印详细的坐标点信息
    
    参数:
        result: 主函数返回的结果字典
        max_points: 每条曲线显示的最大点数
    """
    print("\n" + "=" * 60)
    print("详细坐标点信息")
    print("=" * 60)
    
    for i, curve in enumerate(result['curves']):
        print(f"\n曲线 {i+1}: {curve['legend_name']}")
        print(f"坐标点 (显示前{min(max_points, len(curve['transformed_points']))}个):")
        
        for j, point in enumerate(curve['transformed_points'][:max_points]):
            print(f"  点 {j+1:3d}: X = {point[0]:12.6f}, Y = {point[1]:12.6f}")
        
        if len(curve['transformed_points']) > max_points:
            print(f"  ... (共{len(curve['transformed_points'])}个点)")

if __name__ == "__main__":
    # 设置图像路径
    image_path = "2.jpg"  # 替换为您的图像路径
    
    try:
        # 执行主程序
        result = main(image_path)
        
        # 可选：打印详细坐标信息
        if result:
            print_detailed_coordinates(result, max_points=5)
            
    except Exception as e:
        print(f"程序执行过程中出错: {e}")