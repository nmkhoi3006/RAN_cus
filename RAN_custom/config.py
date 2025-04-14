def get_config():
    return {
        "root": "./aio-hutech",
        "batch_size": 64,
        "shuffle": True,
        "num_workers": 2,
        "lr": 10**-4,
        "num_epochs": 100,
        "num_classes": 4,
        "version": "l",

        "order": 8,
        "save_model": "./runs",
        "pre_train_best": f"./runs/save_7/best.pth",
        "pre_train_last": f"./runs/save_7/last.pth",

        #===============Model==================
        "num_blocks": 4,
        "num_channels": (64, 128, 256, 512),
        "num_heads": (8, 8, 8, 8),
        "dropout": 0.5,
        "label_smoothing": 0.1
    }