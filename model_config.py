TAMNet_Config = {
    # --- 核心模式设置 ---
    # 可选 'residual' (预测绝对残差) 或 'factor' (预测时变缩放因子)
    'correction_mode': 'factor',
    'use_mamba': True,          # 必须关闭 Mamba
    'use_transformer': False,     # 开启 Transformer
    'transformer_heads': 4,
    'transformer_layers': 2,

    # --- 动态分支 ---
    'dyn_hidden_size': 64,
    'lstm_layers': 2,
    'mamba_d_state': 16,
    'mamba_d_conv': 4,
    'mamba_expand': 2,

    # --- 静态分支 ---
    'sta_fetch_embed_dim': 32,
    'sta_scalar_embed_dim': 16,

    # --- 融合与预测 ---
    'fusion_hidden_dim': 64,
    'dropout': 0.1,

    # --- 训练参数 ---
    'lr': 1e-4,
    'batch_size': 256,
    'n_epochs': 50,
    'huber_slope': 1.5
}