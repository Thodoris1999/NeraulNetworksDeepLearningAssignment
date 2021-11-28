
import numpy as np
import matplotlib.pyplot as plt

def bar_plot(title, labels, data):
    ind = np.arange(len(labels))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(ind, data)
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)


def viz_cm_examples(cm_examples):
    fig = plt.figure()
    fig.supxlabel('predicted classes')
    fig.supylabel('actual classes')
    for y in range(10):
        for pred in range(10):
            img = cm_examples[y][pred]
            i = y*10+pred
            ax = fig.add_subplot(10,10,i+1)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            img = 255*img.reshape(28,28)
            ax.imshow(img)