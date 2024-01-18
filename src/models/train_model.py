import hydra

import mlflow
import mlflow.pytorch

from torch.utils.data import DataLoader
from omegaconf import DictConfig
## TODO Improve and make self-sustained

@hydra.main(version_base=None, config_path="../conf", config_name="train_model") 
# This decorator add the parameter "cfg" to the launch function 
# the cfg object is an instance of the DictConfig class. You can think of it as a dictionnary , when dic['key'] is accessible as the( dict.key)
# cfg is loaded from the yaml file at path ../conf/train_model.yaml
def launch(cfg: DictConfig):
    train_dataset, test_dataset = load_dataset(cfg)

    # Create data loaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    # Set an experiment name, which must be unique and case-sensitive.
    experiment = mlflow.set_experiment(cfg.name)

    # Get Experiment Details
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.name, tracking_uri="file:./mlruns", log_model=True
    )
    mlf_logger.log_hyperparams(cfg)
    x_shape, y_shape = train_dataset[0][0].size(dim=0), train_dataset[0][1].size(dim=0)
    model = LinearPositive(x_shape, y_shape, lr=cfg.lr, alpha_l1=cfg.alpha_l1)
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        devices="auto",
        strategy="ddp",
        logger=mlf_logger,
    )
    # Train the model
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=test_loader
    )
    # mlflow.pytorch.save_model(model,"path")


if __name__ == "__main__":
    launch()
