
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


from batchnorm_cnn import BatchnormCNN
from cnn import CNNetwork
import nn_train
import utils


device = "gpu" if torch.cuda.is_available else "cpu"

def main():
    tune_checkpoint = 'norm_cnn_tune.pt'
    checkpoint = 'norm_cnn.pt'
    Network = BatchnormCNN
    model = Network()
    print(model)

    lrs = [1e-4, 1e-3, 1e-2]
    batch_sizes = [16, 32, 64, 128]

    best_acc = -1
    best_config = {}
    for batch_size in batch_sizes:
        train_dataloader, valid_dataloader = utils.mnist_train_valid_split(batch_size)
        for lr in lrs:
            config = {"lr": lr, "batch_size": batch_size}
            model = Network()
            acc, accuracies, train_losses = nn_train.train_mnist(model, config, train_dataloader, valid_dataloader, epochs=20, checkpoint=tune_checkpoint)
            if acc > best_acc:
                best_acc = acc
                best_config = config

    print(f"Done optimizing hyperparams, best config {best_config}")
    model = Network()
    train_dataloader, test_dataloader = utils.mnist_data(best_config['batch_size'])
    acc, accuracies, train_losses = nn_train.train_mnist(model, best_config, train_dataloader, test_dataloader, checkpoint=checkpoint)

if __name__ == '__main__':
    main()