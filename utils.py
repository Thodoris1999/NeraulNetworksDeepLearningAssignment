
import gzip
import os
import time
from urllib.request import urlretrieve
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def eval_sklearn_clf(clf, X, y):
    size = len(y)
    start = time.time()
    pred = clf.predict(X)
    dur = time.time()-start
    
    cm = confusion_matrix(y, pred)
    precisions = np.diag(cm) / np.sum(cm, axis=1)
    recalls = np.diag(cm) / np.sum(cm, axis=0)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    ncorr = sum([1 for y_pred, label in zip(pred, y) if y_pred == label])
    return accuracy, precisions, recalls, cm, dur


def eval_pytorch_model(dataloader, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    model.eval()
    ys = np.zeros((size,))
    preds = np.zeros((size,))

    start = time.time()
    cm_examples = np.zeros((10,10,28,28)) # examples of (mis)classification for each combination
    with torch.no_grad():
        for batch, (X,y) in enumerate(dataloader):
            ys[batch_size*batch:batch_size*(batch+1)] = y
            X, y = X.to(device), y.to(device)

            logits = model(X)
            batch_preds = logits.argmax(1).to("cpu")
            preds[batch_size*batch:batch_size*(batch+1)] = batch_preds
            for y,pred,x in zip(y,batch_preds,X):
                x,y = x.to("cpu"),y.to("cpu")
                cm_examples[int(y)][int(pred)] = x
    dur = time.time()-start

    cm = confusion_matrix(ys, preds)
    precisions = np.diag(cm) / np.sum(cm, axis=1)
    recalls = np.diag(cm) / np.sum(cm, axis=0)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    return accuracy, precisions, recalls, cm, cm_examples, dur


def test_pytorch_model(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X,y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = correct / size
    print(f"loss: {test_loss:>7f}, accuracy: {100*accuracy:>5f}%")
    return accuracy, test_loss

def mnist_train_valid_split(batch_size):
    training_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor()
    )

    train_len = int(len(training_data)*0.85)
    train_data, valid_data = torch.utils.data.random_split(training_data, [train_len, len(training_data)-train_len])

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size)
    return train_dataloader, valid_dataloader


def mnist_data(batch_size):
    training_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader


#credit: https://mattpetersen.github.io/load-mnist-with-numpy
"""Load from /home/USER/data/mnist or elsewhere; download if missing."""
def mnist(path=None):
    r"""Return (train_images, train_labels, test_images, test_labels).

    Args:
        path (str): Directory containing MNIST. Default is
            /home/USER/data/mnist or C:\Users\USER\data\mnist.
            Create if nonexistant. Download any missing files.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values.
            Columns of labels are a onehot encoding of the correct class.
    """
    url = 'http://yann.lecun.com/exdb/mnist/'
    files = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']

    if path is None:
        # Set path to ./data/MNIST/raw (common to pytorch default dataloader path)
        path = os.path.join('data', 'MNIST', 'raw')

    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Download any missing files
    for file in files:
        if file not in os.listdir(path):
            urlretrieve(url + file, os.path.join(path, file))
            print("Downloaded %s to %s" % (file, path))

    def _images(path):
        """Return images loaded locally."""
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        return pixels.reshape(-1, 784).astype('float32') / 255

    def _labels(path):
        """Return labels loaded locally."""
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            integer_labels = np.frombuffer(f.read(), 'B', offset=8)

        return integer_labels

    train_images = _images(os.path.join(path, files[0]))
    train_labels = _labels(os.path.join(path, files[1]))
    test_images = _images(os.path.join(path, files[2]))
    test_labels = _labels(os.path.join(path, files[3]))

    return train_images, train_labels, test_images, test_labels