import argparse

def get_args():
    parser = argparse.ArgumentParser(description="ESA-based Node Embedding Model")

    # --- 数据相关参数 ---
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument('--data_path', type=str, default='datasets/ACM-DBLP_0.2.npz',help='输入数据文件的路径 (.npz格式)')

    # --- 模型架构参数 ---
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--embed_dim', type=int, default=128,help='最终输出的节点嵌入维度')
    model_group.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 64, 128],help='每个注意力层的隐藏维度列表')
    model_group.add_argument('--num_heads', type=int, nargs='+', default=[4, 4, 8],help='每个注意力层的头数列表')
    model_group.add_argument('--layer_types', type=str, nargs='+', default=['M', 'M', 'S'],choices=['M', 'S'],
                             help="注意力层的类型序列 ('M' for Masked, 'S' for Standard)")
    model_group.add_argument('--no_mlp', action='store_true',help='如果设置，则不在注意力块后使用MLP')
    model_group.add_argument('--dropout', type=float, default=0.1,help='模型中使用的dropout率')

    # --- 运行参数 ---
    run_group = parser.add_argument_group('Runtime Parameters')
    run_group.add_argument('--device', type=str, default='auto',help="运行设备 ('auto', 'cpu', 'cuda')")

    args = parser.parse_args()

    # --- 参数校验 ---
    if len(args.hidden_dims) != len(args.num_heads) or len(args.hidden_dims) != len(args.layer_types):
        raise ValueError("`hidden_dims`, `num_heads`, 和 `layer_types` 的长度必须一致。")
    
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return args

if __name__ == '__main__':
    # 这是一个简单的测试，用来展示如何使用这个配置文件
    import torch
    args = get_args()
    print("解析后的参数配置:")
    for key, value in vars(args).items():
        print(f"  - {key}: {value}")