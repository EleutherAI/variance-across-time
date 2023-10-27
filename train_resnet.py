import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from torchvision import models
from torchvision import transforms, datasets
import pytorch_lightning as pl


class ResNet18(pl.LightningModule):

    def __init__(self, num_classes):
        super().__init__()

        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        model.maxpool = nn.Identity()
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, num_classes)
        )
        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.05, momentum=0.99, weight_decay=5e-4
        )
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            0.1,
            epochs=20,
            steps_per_epoch=782,
            three_phase=True,
        )

        return {"optimizer": optimizer, "LRScheduler" : scheduler}
    
    def training_step(self, train_batch, batch_idx):
        self.model.train()
        x, y = train_batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        # self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        self.model.eval()
        x,y = val_batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        # self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss



transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

resnet18 = ResNet18(num_classes=10)
trainer = pl.Trainer(max_epochs=20)
trainer.fit(model=resnet18, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
