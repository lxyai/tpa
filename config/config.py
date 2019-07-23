config = {
  "attention_len": 48,
  "batch_size": 32,
  "output_size": 321,
  "dropout": 0.2,
  "file_out": "log.out",
  "horizon": 3,
  "learning_rate": 0.003,
  "max_gradient_norm": 5.0,
  "num_epoch": 20,
  "num_layers": 3,
  "num_units": 48,
  "num_highway": 48,
  "data_config": {
    "y": {
      "range": [3,3],
      "key": [i for i in range(321)]
    },
    "x": {
      "range": [-47, 0],
      "key": [i for i in range(321)]
    },
    "z": None,
    "shuffle": True,
    "num_epoch": 20,
    "batch_size": 32,
    "step": 1,
    "split": [0, 0.6, 0.8, 1]
  }
}