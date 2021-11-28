
import torch
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import os
import matplotlib.pyplot as plt

from cnn import CNNetwork
from resnet import ResCNN
import utils
import viz_utils

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    checkpoint_name = "rescnn_main.pt"
    MODELS_PATH = 'models'
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    MODEL_PATH = os.path.join(MODELS_PATH, checkpoint_name)

    net = ResCNN()
    checkpoint = torch.load(MODEL_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)
    config = checkpoint['config']
    print(f"train config: {config}")

    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
    )

    test_dataloader = DataLoader(test_data, batch_size=config['batch_size'])

    accuracy, precisions, recalls, cm, cm_examples, dur = utils.eval_pytorch_model(test_dataloader, net)
    print(f"Infer time: {dur}s")
    print(f"Accuracy: {100*accuracy}%")
    print(recalls)
    print(precisions)
    viz_utils.bar_plot('NN recalls', list(range(10)), recalls)
    viz_utils.bar_plot('NN precisions', list(range(10)), precisions)
    viz_utils.viz_cm_examples(cm_examples)
    plt.show()


if __name__ == '__main__':
    main()