import wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="ClassifierProject",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "CNN",
    "dataset": "CIFAR-10",
    "batch_size": 8,
    "epochs": 4,
    }
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')