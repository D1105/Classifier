import classifier.data
import classifier.model
import classifier.train
import classifier

net = classifier.model.Net()

print('Train started')
trainloader = classifier.data.trainloading()
classifier.train.train(net, trainloader)

print('Test started')
testloader = classifier.data.testloading()
classifier.model.testNet(net, testloader)


classifier.model.accuracy_for_classes(net, testloader)