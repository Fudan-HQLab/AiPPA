import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F


class Training:
    def __init__(self, model, train_dataloader,
                 val_dataloader,
                 prefix, device, lr,
                 model_path, log_dir):

        self.device = device
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.prefix = prefix
        self.saved_model_path = model_path
        self.lr = lr
        # optimizer
        self.opt = optim.Adam(params=model.parameters(),
                              lr=self.lr)
        # loss function
        self.loss_fn = nn.MSELoss().to(self.device)
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self, epochs):
        total_steps = 0
        min_loss = 10.0
        if not os.path.exists(self.saved_model_path):
            os.mkdir(self.saved_model_path)
        for epoch in range(epochs):
            self.model.train()
            # print(f"-------------The epoch {epoch} begins-------------")
            for idx, batch in enumerate(self.train_dataloader):
                y = batch.y.to(self.device)
                batch = batch.to(self.device)
                pred = self.model(data=batch)
                loss = self.loss_fn(pred.squeeze(1), y)
                # begin opt
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                # every batch(step) will be recorded
                total_steps += 1
                # log
                self.writer.add_scalar("LossPerBatch", loss.item(), global_step=total_steps)

            # save model every epoch
            model_name = f'{self.prefix}_epoch_{str(epoch).zfill(4)}.pth'
            torch.save(self.model.state_dict(),
                       os.path.join(self.saved_model_path, model_name))

            train_loss = self._validation(self.train_dataloader)
            # validation
            val_loss = self._validation(self.val_dataloader)
            print(f"--The epoch {epoch} loss: {train_loss}--")

            if val_loss < min_loss:
                min_loss = val_loss
                print(f'Lower loss {min_loss:.3f} of validation on `epoch {epoch}`')

            self.writer.add_scalars(main_tag='Loss',
                                    tag_scalar_dict={'train_loss': train_loss,
                                                     'val_loss': val_loss},
                                    global_step=epoch)
        self.writer.close()

    def _validation(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            # total_loss = []
            y_ = torch.tensor([])
            pred_ = torch.tensor([])
            for batch in dataloader:
                y = batch.y.to(self.device)
                batch = batch.to(self.device)
                pred = self.model(data=batch)
                y_ = torch.cat((y_, torch.flatten(y.to('cpu'))), dim=-1)
                pred_ = torch.cat((pred_, torch.flatten(pred.to('cpu'))), dim=-1)
            whole_loss = F.mse_loss(pred_, y_)
        return whole_loss
