
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

import utils


device = "cuda" if torch.cuda.is_available() else "cpu"

def train_batch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f} [{current}/{size}]")

    return loss


def train_mnist(net, config, train_dataloader, valid_dataloader, epochs=20, checkpoint='nn.pt', retrain=True):
    print("Using {} device".format(device))
    net.to(device)

    MODELS_PATH = 'models'
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    MODEL_PATH = os.path.join(MODELS_PATH, checkpoint)

    if os.path.exists(MODEL_PATH) and not retrain:
        model_state, optimizer_state = torch.load(MODEL_PATH)
        net.load_state_dict(model_state)
        optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9)
        optimizer.load_state_dict(optimizer_state)
    else:
        optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9)

    loss_fn = nn.CrossEntropyLoss()
    max_accuracy = -1
    accuracies = np.zeros((epochs,))
    train_losses = np.zeros((epochs,))
    print("----------------Trainining with params------------------")
    print(config)
    print()
    for t in range(epochs):
        print(f"Epoch {t}\n-----------------------")
        train_loss = train_batch(train_dataloader, net, loss_fn, optimizer)
        accuracy, loss = utils.test_pytorch_model(valid_dataloader, net, loss_fn, device)

        accuracies[t] = accuracy
        train_losses[t] = train_loss
        if accuracy > max_accuracy:
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': loss,
                'config': config
            }, MODEL_PATH)
            print(f'Saved best weights at epoch {t} with accuracy {100*accuracy:>5f}%')
            max_accuracy = accuracy

    print("Done!")
    return max_accuracy, accuracies, train_losses