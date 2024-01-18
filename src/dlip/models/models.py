import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        # We allocate space for the weights
        self.l1 = nn.Linear(input_size, 100)
        self.l2 = nn.Linear(100, output_size)
        # Input size is 16*16, output size should be the same with the number of classes

    def forward(self, inputs):  # Called when we apply the network
        h = F.relu(
            self.l1(inputs)
        )  # You can put anything, as long as its Pytorch functions
        outputs = F.softmax(
            self.l2(h), dim=1
        )  # Use softmax as the activation function for the last layer
        return outputs


def load_model(path_checkpoint, modelClass: torch.nn.Module):
    model = modelClass()
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model


def save_model(path_checkpoint, model):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        path_checkpoint,
    )
