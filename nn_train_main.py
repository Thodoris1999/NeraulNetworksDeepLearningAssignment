
import nn_train
import utils

import torch
import matplotlib.pyplot as plt

import fcn
import cnn
import batchnorm_cnn
import resnet

device = "gpu" if torch.cuda.is_available else "cpu"

def main():
    checkpoint = "bn_cnn_main.pt"
    net = batchnorm_cnn.BatchnormCNN()
    config = {'batch_size': 32, "lr": 1e-2}

    train_dataloader, test_dataloader = utils.mnist_data(config['batch_size'])

    acc, accuracies, train_losses = nn_train.train_mnist(net, config, train_dataloader, test_dataloader, checkpoint=checkpoint)

    plt.plot(accuracies)
    plt.show()

if __name__ == '__main__':
    main()