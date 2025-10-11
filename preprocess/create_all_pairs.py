import os
import itertools
import subprocess
import argparse

def main(args):
    """
    自动扫描指定目录中的.edgelist文件，
    并为所有可能的两两组合运行preprocess_pairwise.py脚本。
    """
    input_dir = args.input_dir
    output_dir = args.output_dir
    script_path = args.script_path
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- 正在扫描目录: {input_dir} ---")
    
    # 找到所有.edgelist文件
    edgelist_files = [f for f in os.listdir(input_dir) if f.endswith('.edgelist')]
    
    if len(edgelist_files) < 2:
        print("错误：在指定目录中未能找到至少两个.edgelist文件。")
        return
        
    print(f"发现 {len(edgelist_files)} 个网络文件: {', '.join(edgelist_files)}")
    
    # 生成所有唯一的两两组合
    file_pairs = list(itertools.combinations(edgelist_files, 2))
    
    print(f"\n--- 将为以下 {len(file_pairs)} 个组合生成对齐任务数据集 ---")
    
    # 3. 为每个组合运行预处理脚本
    for i, (file1, file2) in enumerate(file_pairs):
        layer1_path = os.path.join(input_dir, file1)
        layer2_path = os.path.join(input_dir, file2)
        
        # 创建一个清晰的输出文件名
        name1 = os.path.splitext(file1)[0].replace('_network', '')
        name2 = os.path.splitext(file2)[0].replace('_network', '')
        output_filename = f"{name1}_vs_{name2}.npz"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\n[{i+1}/{len(file_pairs)}] 正在处理: {name1} vs {name2}")
        print(f"  -> 输出文件: {output_path}")
        
        # 构建并执行命令行命令
        command = [
            'python',
            script_path,
            '--layer1', layer1_path,
            '--layer2', layer2_path,
            '--output_file', output_path,
            '--split_ratios', *args.split_ratios
        ]
        
        try:
            # 运行子进程并等待其完成
            subprocess.run(command, check=True, text=True)
            print(f"  ✅ 处理成功！")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 处理失败！命令 '{' '.join(command)}' 返回错误: \n{e}")
        except FileNotFoundError:
            print(f"  ❌ 错误: 找不到脚本 '{script_path}'。请确保路径正确。")
            break

    print("\n--- 所有任务处理完毕 ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="自动为目录中的所有.edgelist文件对创建网络对齐数据集。")
    parser.add_argument('--input_dir', type=str, default='datasets/higgs', help='包含.edgelist文件的输入目录的路径。')
    parser.add_argument('--output_dir', type=str, default='datasets/higgs', help='保存生成的.npz文件的输出目录的路径。')
    parser.add_argument('--script_path', type=str, default='dataset_preprocess/preprocess_pairwise.py',help='preprocess_pairwise.py脚本的路径。')
    parser.add_argument('--split_ratios', type=str, nargs=3, default=['0.2', '0.1', '0.7'],
                        help='训练、验证、测试集的划分比例。')
    
    args = parser.parse_args()
    main(args)