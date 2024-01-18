import torch
from torch.data.utils import DataLoader

# Calculate the accuracy to evaluate the model
def accuracy(dataset, model):
    with torch.no_grad():
        correct = 0
        total = 0
        dataloader = DataLoader(dataset)
        for images, labels in dataloader:
            images = images.view(-1, 16 * 16)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()

    print(
        "Accuracy of the model : {:.2f} %".format(100 * correct.item() / len(dataset))
    )