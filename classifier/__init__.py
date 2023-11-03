import wandb

batchSize = 4
epochs = 2
learning_rate = 0.001

wandb.init(
    # set the wandb project where this run will be logged
    project="ClassifierProject",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "CNN",
    "dataset": "CIFAR-10",
    "batch_size": batchSize,
    "epochs": epochs,
    }
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')