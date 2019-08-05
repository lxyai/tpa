config = {
    "attention_len": 50,
    "batch_size": 32,
    "file_out": "log.out",
    "learning_rate": 0.001,
    "max_gradient_norm": 5.0,
    "num_epoch": 20,
    "highway": 16,
    "decay": 1000,
    "num_units": 64,
    "num_layers": 3,
    "dropout": 0.2,
    "model_dir": "ckpt",
    "data_config": {
        "y": {
            "range": [7, 7],
            "key": [str(i) for i in range(321)]
        },
        "x": {
            "range": [-49, 0],
            "key": [str(i) for i in range(321)]
        },
        "shuffle": True,
        "num_epoch": 20,
        "batch_size": 32,
        "step": 1,
        "split": [0, 0.6, 0.8, 1],
        "normalization": "global"
    }
}