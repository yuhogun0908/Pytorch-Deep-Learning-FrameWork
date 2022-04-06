import os
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Trainer(object):
    def __init__(self, device, tr_loader, va_loader, test_loader, model, optimizer, scheduler, config, log_path):

        # self.cuda_flag = cuda_flag
        self.device = device

        self.tr_loader = tr_loader
        self.va_loader = va_loader
        self.test_loader = test_loader
        # self.model_allosaurus = model_allosaurus
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.writter = SummaryWriter(log_path)

        # If you want to restart the checkpoint model to train
        load_flag, model_load_path = config["train"]["model_load_path"]
        self._reset(load_flag, model_load_path)

        # save the model during training
        self.model_save_folder = config["train"]["mdoel_save_path"]
        Path(self.model_save_folder).mkdir(exist_ok=True, parents=True)

        self.check_flag, self.check_freq = config["train"]["check_point"]

        self.tr_avg_loss = torch.Tensor(config["train"]["epochs"])
        self.val_avg_loss = torch.Tensor(config["train"]["epochs"])

        # gradient clipping
        self.clipping = config["train"]["clipping"]
        self.max_norm = config["train"]["max_norm"]

    def _reset(self, load_flag, model_load_path):

        if load_flag:
            print("Loading checkpoint model %s" % model_load_path)
            package = torch.load(self.config["train"]["model_load"][1], map_location=self.device)
            self.model.load_state_dict(package["model_state_dict"])
            self.optimizer.load_state_dict(package["optimizer"])
            self.start_epoch = int(package.get("epoch", 1))
            self.tr_avg_loss[: self.start_epoch] = package["tr_avg_loss"][: self.start_epoch]
            self.val_avg_loss[: self.start_epoch] = package["val_avg_loss"][: self.start_epoch]
        else:
            self.start_epoch = 0

        self.prev_val_loss = float("inf")
        self.best_val_loss_epoch = float("inf")

    def train(self):

        for epoch in tqdm(range(self.start_epoch, self.config["train"]["epochs"])):
            print("Training Start ...")
            start = time.time()
            self.model.train()
            tr_avg_loss_epoch, tr_avg_acc_epoch = self._run_one_epoch(epoch, training=True, test=False)

            print("-" * 85)
            print(
                "Train Summary | End of Epoch {0} | Time {1:.2f}s | Train Loss {2:.3f} | Train ACC {3:.6f}".format(
                    epoch + 1, time.time() - start, tr_avg_loss_epoch, tr_avg_acc_epoch
                )
            )
            print("_" * 85)
            self.writter.add_scalar("data/Train_Loss", tr_avg_loss_epoch, epoch)
            self.writter.add_scalar("data/Train_ACC", tr_avg_acc_epoch, epoch)

            if self.check_flag:
                if epoch % self.check_freq == 0:
                    file_path = os.path.join(self.model_save_folder, "epoch%d.pth.tar" % (epoch + 1))
                    state_dict = {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "tr_avg_loss": self.tr_avg_loss,
                        "val_avg_loss": self.val_avg_loss,
                    }
                    torch.save(state_dict, file_path)
                    print("Saving checkpoint model to %s" % file_path)

            print("validation...")
            self.model.eval()
            val_avg_loss_epoch, val_avg_acc_epoch = self._run_one_epoch(epoch, training=False, test=False)
            print("-" * 85)
            print(
                "Valid Summary | End of Epoch {0} | Time {1:.2f}s |"
                "Valid Loss {2:.3f} | Valid ACC {3:.6f}".format(
                    epoch + 1, time.time() - start, val_avg_loss_epoch, val_avg_acc_epoch
                )
            )
            print("_" * 85)
            self.writter.add_scalar("data/Validation_Loss", val_avg_loss_epoch, epoch)
            self.writter.add_scalar("data/Validation_ACC", val_avg_acc_epoch, epoch)

            # print("Test...")
            # self.model.eval()
            # test_avg_loss_epoch, test_avg_acc_epoch = self._run_one_epoch(epoch, training=False, test=True)
            # print("-" * 85)
            # print(
            #     "Test Summary | End of Epoch {0} | Time {1:.2f}s |"
            #     "Test Loss {2:.3f} | Test ACC {3:.6f}".format(
            #         epoch + 1, time.time() - start, test_avg_loss_epoch, test_avg_acc_epoch
            #     )
            # )
            # print("_" * 85)
            # self.writter.add_scalar("data/Test_Loss", test_avg_loss_epoch, epoch)
            # self.writter.add_scalar("data/Test_ACC", test_avg_acc_epoch, epoch)

            # save best model
            self.tr_avg_loss[epoch] = tr_avg_loss_epoch
            self.val_avg_loss[epoch] = val_avg_loss_epoch
            if val_avg_loss_epoch < self.best_val_loss_epoch:
                self.best_val_loss_epoch = val_avg_loss_epoch
                file_path = os.path.join(self.model_save_folder, self.model_path)
                state_dict = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "tr_avg_loss": self.tr_avg_loss,
                    "val_avg_loss": self.val_avg_loss,
                }
                torch.save(state_dict, file_path)
                print("Find better validated model, saving to %s" % file_path)

            # scheduler
            self.scheduler.step(val_avg_loss_epoch)
            self.model.allo_embed_saved = True  # TODO: get allo embedding

        self.writter.close()

    def _run_one_epoch(self, epoch, training=True, test=False):
        
        start = time.time()
        total_loss = 0
        total_acc = 0
        loss = 0
        if training:
            data_loader = self.tr_loader
        elif test:
            data_loader = self.test_loader
        else:
            data_loader = self.va_loader

        # data_loader = self.tr_loader if training else self.va_loader
        for idx, (data) in enumerate(data_loader):
            """
            Input :

            Output :
            """
            loss, acc = self.model(data)

            # when beamforming is performed after training.
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                # gradient threshold to clip
                # if self.clip_norm:
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()
            total_acc += acc

            if idx % self.config["train"]["print_freq"] == 0:
                print(
                    "Epoch {0} | Iter {1} | Average Loss {2:.3f} |"
                    " Current Loss {3:6f} | Average Acc {4:3f} | Current Acc {5:3f} | {6:.1f} ms/batch".format(
                        epoch + 1,
                        idx + 1,
                        total_loss / (idx + 1),
                        loss.item(),
                        total_acc / (idx + 1),
                        acc,
                        1000 * (time.time() - start) / (idx + 1),
                    ),
                    flush=True,
                )

        return total_loss / (idx + 1), total_acc / (idx + 1)
