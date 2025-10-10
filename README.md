# Attention-and-LLM-for-alignment

## 
python main_train.py \
    --data_path datasets/Douban_0.2.npz \
    --split_ratios 0.2 0.1 0.7 \
    --seed 2024 \
    --embed_dim 256 \
    --hidden_dims 128 256 \
    --num_heads 4 8 \
    --layer_types M S \
    --dropout 0.15 \
    --lr 0.0008 \
    --epochs 800 \
    --margin 0.6 \
    --neg_sample_k 30 \
    --device cuda