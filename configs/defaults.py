
SEEDS = [0, 1, 3]

PADDING_IDX = 0

MULTILABEL_THRESHOLD = 0.5

DEFAULTS = {
    "batch_size": 8,
    "bert":  "bert-base-uncased", #"diptanu/fBERT", 
    "clip_value": 1.0,
    "dropout": 0.1,
    "embed_dim": 256,
    "encoder": "gru",
    "epochs": 5,
    "hidden_dim": 128,
    "lr": 1e-4,
    "main_only_epochs": 0,
    "num_layers": 5,
    "num_restarts": 1,
    "patience": 2,
    "print_every": 100,
    "save_path": "go",
    "stress_weight": -1,
    "tolerance": 1e-4,
    "trials": 5
}