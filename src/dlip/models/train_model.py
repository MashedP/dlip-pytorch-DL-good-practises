import logging
import os

import hydra
import torch
from dlip.data.data import load_dataset
from dlip.models.models import LinearModel, save_model
from hydra import utils
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split


def train(num_epochs, batch_size, criterion, optimizer, model, dataset):
    train_error = []
    train_loader = DataLoader(dataset, batch_size, shuffle=True)
    model.train()  # Indicates to the network we are in training mode
    for epoch in range(num_epochs):
        epoch_average_loss = 0.0
        for images, labels in train_loader:
            y_pre = model(images.view(batch_size, -1))
            # reshape the inputs from [N, img_shape, img_shape] to [N, img_shape*img_shape]

            # One-hot encoding or labels so as to calculate MSE error:
            labels_one_hot = torch.FloatTensor(batch_size, 10)
            labels_one_hot.zero_()
            labels_one_hot.scatter_(1, labels.view(-1, 1), 1)

            loss = criterion(y_pre, labels_one_hot)  # Real number
            optimizer.zero_grad()  # Set all the parameters gradient to 0
            loss.backward()  # Computes  dloss/da for every parameter a which has requires_grad=True
            optimizer.step()  # Updates the weights
            epoch_average_loss += loss.item() * batch_size / len(dataset)
        train_error.append(epoch_average_loss)
        logging.info(
            "Epoch [{}/{}], Loss: {:.4f}".format(
                epoch + 1, num_epochs, epoch_average_loss
            )
        )
    return train_error


@hydra.main(config_path="../conf", config_name="train_model")
# This decorator add the parameter "cfg" to the launch function
# the cfg object is an instance of the DictConfig class. You can think of it as a dictionnary , when dic['key'] is accessible as the( dict.key)
# cfg is loaded from the yaml file at path ../conf/train_model.yaml
def launch(cfg: DictConfig):
    working_dir = os.getcwd()
    dataset = load_dataset(utils.get_original_cwd() + cfg.data_path)
    train_set, val_set = random_split(dataset, [6000, 1291])
    model = LinearModel(16 * 16, 10)

    # Use mean squared loss function
    criterion = torch.nn.MSELoss()

    # Use SGD optimizer with a learning rate of 0.01
    # It is initialized on our model
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    train(cfg.num_epochs, cfg.batch_size, criterion, optimizer, model, train_set)
    logging.info(f"Checkpoint is saved at {working_dir}")
    save_model(working_dir + "/checkpoint.pt", model)

    # TODO Maybe improve the logging of the training loop ?
    # TODO Vizualisation methods ?


if __name__ == "__main__":
    launch()
