from tqdm import tqdm
from tqdm import trange

import torch

def train(net, trainloader, optimizer, loss_function, scheduler, device, epochs, testloader):
    t = tqdm(range(epochs))
    for epoch in t:  # loop over the dataset multiple times
        train_loss = train_single_epoch(net, trainloader, optimizer, loss_function, device)
        scheduler.step()
        test_accuracy = test(net, testloader, device)
        t.set_description(f'epoch: {epoch + 1}, train loss: {train_loss:.4f} test acc {test_accuracy * 100:.2f}%', refresh=True)

def train_single_epoch(net, trainloader, optimizer, loss_function, device):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
    train_loss = running_loss / len(trainloader)
    return train_loss
    
def test(net, testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.to(device)).cpu()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy