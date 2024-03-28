#version 1.0
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
resnet101 = models.resnet101(pretrained=True)
num_ftrs = resnet101.fc.in_features
resnet101.fc = nn.Linear(num_ftrs, 10)
def train_model(model, criterion, optimizer, num_epochs=1):
    # Before using CUDA, check if it's available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)


    # Lists to keep track of progress
    train_loss, train_accuracy = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in trainloader:

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = running_corrects.double() / len(trainloader.dataset)

        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Accuracy: {epoch_acc}')

    return model, train_loss, train_accuracy
criterion = nn.CrossEntropyLoss()

optimizer_adam = optim.Adam(resnet101.parameters(), lr=0.001)
model_adam, train_loss_adam, train_accuracy_adam = train_model(resnet101, criterion, optimizer_adam)

optimizer_adagrad = optim.Adagrad(resnet101.parameters(), lr=0.001)
model_adagrad, train_loss_adagrad, train_accuracy_adagrad = train_model(resnet101, criterion, optimizer_adagrad)

optimizer_adadelta = optim.Adadelta(resnet101.parameters(),lr=0.001)
model_adadelta, train_loss_adadelta, train_accuracy_adadelta = train_model(resnet101, criterion, optimizer_adadelta)
def evaluate_model(model, testloader):
    model.eval()
    #model = model.to('cuda')
    top5_correct = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            #inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _, top5_preds = outputs.topk(5, 1, True, True)
            top5_correct += torch.sum(top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds)))

    top5_acc = top5_correct.double() / len(testloader.dataset)
    return top5_acc

# Evaluate for each model and optimizer
top5_acc_adam = evaluate_model(model_adam, testloader)
print(f'Top-5 Accuracy with Adam: {top5_acc_adam}')

# Plot the training loss and accuracy
def plot_metrics(train_loss, train_accuracy, title=''):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.title(f'Training Loss {title}')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.title(f'Training Accuracy {title}')
    plt.legend()
    plt.show()

plot_metrics(train_loss_adam, train_accuracy_adam, 'with Adam')
top5_acc_adagrad = evaluate_model(model_adagrad, testloader)
print(f'Top-5 Accuracy with Adagrad: {top5_acc_adagrad}')

plot_metrics(train_loss_adagrad, train_accuracy_adagrad, 'with Adagrad')
top5_acc_adadelta = evaluate_model(model_adadelta, testloader)
print(f'Top-5 Accuracy with Adadelta: {top5_acc_adadelta}')

plot_metrics(train_loss_adadelta, train_accuracy_adagrad, 'with Adadelta')
