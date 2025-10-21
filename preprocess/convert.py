import os
import pandas as pd
import numpy as np
import argparse

def load_attrs(attr_file):
    """读取节点属性文件，返回矩阵"""
    df = pd.read_csv(attr_file, sep=None, engine='python', header=None)
    return df.values.astype(np.float32)

def load_edges(edge_file, num_nodes=None):
    """读取边文件，返回标准格式 (2, E)，并自动清洗"""
    edges = pd.read_csv(edge_file, sep=None, engine='python', header=None).values

    # 修正编号从 1 开始的情况
    if edges.min() == 1:
        edges = edges - 1

    # 过滤越界节点
    if num_nodes is not None:
        before = len(edges)
        edges = edges[(edges[:, 0] < num_nodes) & (edges[:, 1] < num_nodes)]
        after = len(edges)
        if before != after:
            print(f"⚠️  {edge_file} 中有 {before - after} 条边越界，已自动清除。")

    # 转换为 (2, E)
    return edges.T.astype(np.int64)

def convert_dataset(dataset_name, input_dir, output_path):
    """主函数：将原始文件转为 LA 框架 npz 格式"""
    attr1_file = os.path.join(input_dir, f"{dataset_name}attr1.csv")
    attr2_file = os.path.join(input_dir, f"{dataset_name}attr2.csv")
    edge1_file = os.path.join(input_dir, f"{dataset_name}1.edges")
    edge2_file = os.path.join(input_dir, f"{dataset_name}2.edges")
    pair_file = os.path.join(input_dir, f"{dataset_name}.csv")

    print(f"--- 正在处理数据集: {dataset_name} ---")

    # 1. 读取属性
    x1 = load_attrs(attr1_file)
    x2 = load_attrs(attr2_file)

    # 2. 读取边
    edge_index1 = load_edges(edge1_file, num_nodes=x1.shape[0])
    edge_index2 = load_edges(edge2_file, num_nodes=x2.shape[0])

    # 3. 读取锚点对
    pairs = pd.read_csv(pair_file, sep=None, engine='python', header=None).values
    if pairs.min() == 1:
        pairs = pairs - 1
    pairs = pairs.astype(np.int64)

    # 4. 打印基本信息
    print(f"图1: 节点 {x1.shape[0]}，边 {edge_index1.shape[1]}")
    print(f"图2: 节点 {x2.shape[0]}，边 {edge_index2.shape[1]}")
    print(f"锚点对: {pairs.shape[0]}")

    # 5. 保存 npz 文件（LA 格式）
    np.savez_compressed(
        output_path,
        x1=x1,
        x2=x2,
        edge_index1=edge_index1,
        edge_index2=edge_index2,
        pos_pairs=pairs,
    )

    print(f"✅ 已保存到 {output_path}")
    print(f"文件结构：{list(np.load(output_path).keys())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset to LA-compatible npz format.")
    parser.add_argument("--data", type=str, required=True, help="数据集名称，如 am-td 或 fb-tt")
    parser.add_argument("--input_dir", type=str, default="./", help="原始数据文件所在目录")
    parser.add_argument("--output", type=str, default="./output", help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, f"{args.data}.npz")
    convert_dataset(args.data, args.input_dir, output_path)