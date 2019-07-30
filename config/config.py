config = {
    "attention_len": 40,
    "batch_size": 16,
    "output_size": 1,
    "dropout": 0.2,
    "file_out": "log.out",
    "horizon": 7,
    "learning_rate": 0.0001,
    "max_gradient_norm": 5.0,
    "num_epoch": 50,
    "num_layers": 3,
    "num_units": 64,
    "highway": 0,
    "decay": 1000,
    "model_dir": "../ckpt1",
    "data_config": {
        "y": {
            "range": [1, 7],
            "key": ['1']
        },
        "x": {
            "range": [-39, 0],
            "key": [str(i) for i in range(1, 138)]
        },
        "z": {
            "range": [-39, 14],
            "key": [str(i) for i in range(2, 72)]
        },
        "shuffle": True,
        "num_epoch": 50,
        "batch_size": 16,
        "step": 1,
        "split": [0, 0.8, 1, 1]
    }
}