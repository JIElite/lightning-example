import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


class LitModel(pl.LightningModule):
    def __init__(self, criterion):
        super().__init__()
        self.l1 = torch.nn.Linear(28*28, 10)
        self.criterion = criterion

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = self.criterion(prediction, y)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)


# dataloader
transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# dataset = torchvision.datasets.CIFAR10(os.getcwd(), train=True, transform=transform, download=True)
dataset = torchvision.datasets.MNIST(os.getcwd(), train=True, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8)
# model = getattr(models, 'resnet18')()
pl_model = LitModel(criterion=F.cross_entropy)
trainer = pl.Trainer(gpus="0", precision=16, max_epochs=20)
trainer.fit(pl_model, train_loader)

