import classifier.data
import classifier.model
import classifier.train
import classifier
import pytorch_lightning as L
import wandb
from classifier.train import trainer, dm

wandb.init(
    project="pytorch_classification",

    config={
    "learning_rate": classifier.learning_rate,
    "dataset": "CIFAR-10",
    "batch_size": classifier.batch_size,
    "epochs": classifier.epochs,
    }
)
net = classifier.neural_network.Net(len(classifier.classes))

trainer.fit(net,dm)

trainer.validate(net, dm)

trainer.test(net, dm)

classifier.neural_network.accuracy_for_classes(net, dm.test_dataloader())

wandb.finish()

classifier.neural_network.saveNet(net,'')