import argparse
import torch
import yaml
from torch.utils.data import DataLoader
from dataloader import get_datasets
from network import Network
from train import Trainer
import os.path as osp


def main(args):
    with open(args.config_path, mode="r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    device = torch.device(config["setting"]["device"]) if torch.cuda.is_available() else "cpu"

    # dataloader
    base_path = args.dataset_path

    # tr_dataset, va_dataset, test_dataset = get_datasets(args.dataset, base_path, config[args.dataset])
    tr_dataset, va_dataset = get_datasets(args.dataset, base_path, config[args.dataset])

    tr_loader = DataLoader(tr_dataset, **config["dataloader"]["Train"])
    va_loader = DataLoader(va_dataset, **config["dataloader"]["Validation"])
    test_loader = None
    # test_loader = DataLoader(test_dataset, **config["dataloader"]["Test"])

    # model define
    model = Network(args.dataset, config, device)
    model = model.to(device)
    # model_NLU = torch.nn.Linear(10, 1)  # model_NLU = NLU().to(device)

    # optimizer
    if config["optimizer"]["name"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["optimizer"]["lr"], weight_decay=config["optimizer"]["weight_decay"]
        )
    elif config["optimizer"]["name"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config["optimizer"]["lr"], momentum=config["optimizer"]["momentum"])

    # scheduler
    if config["scheduler"]["name"] == "plateau":
        factor = config["scheduler"]["factor"]
        patience = config["scheduler"]["patience"]
        min_lr = config["scheduler"]["min_lr"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr
        )



    log_path = args.log_path + args.dataset + \
                    "_optimizer_" + config["optimizer"]["name"] + \
                    "_lr_" + str(config["optimizer"]["lr"]) + \
                    "_scheduler_" + config["scheduler"]["name"] 
                    
    trainer = Trainer(device, tr_loader, va_loader, test_loader, model, optimizer, scheduler, config, log_path)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="{dataset name}", choices=["{dataset name1}", "{dataset name2"], help="Dataset to train")
    parser.add_argument("--use_cuda", action="store_true", help="run model using GPU")
    parser.add_argument("--train", action="store_true", help="run SLU training")
    parser.add_argument("--config_path", type=str, default="config/config.yaml", help="path to config file")
    parser.add_argument("--dataset_path", type=str, default="{dataset path}")
    parser.add_argument("--log_path", type=str, default="runs/test", help="tensorboard log path")

    args = parser.parse_args()

    main(args)
