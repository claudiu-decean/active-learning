import torch

def train(net, trainloader, optimizer, loss_function, scheduler, device, epochs=50, testloader=None):
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = train_single_epoch(net, trainloader, optimizer, loss_function, device, epochs=50)
        scheduler.step()
        print(f'epoch: {epoch + 1}, training loss: {train_loss}')
        if testloader is not None:
            test_accuracy = test(net, testloader, device)
            print(f'test accuracy {test_accuracy}')
    print('Finished Training')

def train_single_epoch(net, trainloader, optimizer, loss_function, device, epochs=50):
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