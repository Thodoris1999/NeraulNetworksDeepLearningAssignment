
from numpy.core.fromnumeric import size
import utils
from sklearn import neighbors
from matplotlib import pyplot as plt
import numpy as np
import time

import viz_utils

def train_knn(X, y, k):
    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(X, y)
    return clf


def train_nearest_centroid(X, y):
    clf = neighbors.NearestCentroid()
    clf.fit(X, y)
    viz_centroids(clf.centroids_)
    return clf


def knn_classifier_test(train_images, train_labels, test_images, test_labels, k):
    train_start = time.time()
    clf = train_knn(train_images, train_labels, k)
    trdur = time.time() - train_start
    print(f"{k}-NN classifier trained in {trdur}s")

    accuracy, precisions, recalls, cm, infdur = utils.eval_sklearn_clf(clf, test_images, test_labels)
    print(f"{k}-NN classifier accuracy: {100*accuracy}%")
    print(f"{k}-NN classifier infer time: {infdur}s")
    print(f"{k}-NN classifier per-class precisions")
    print(precisions)
    print(f"{k}-NN classifier per-class recalls")
    print(recalls)
    return accuracy, precisions, recalls, trdur, infdur


def nearest_centroid_classifier_test(train_images, train_labels, test_images, test_labels):
    train_start = time.time()
    clf = train_nearest_centroid(train_images, train_labels)
    trdur = time.time() - train_start
    print(f"Nearest centroid classifier trained in {trdur}s")

    accuracy , precisions, recalls, cm, infdur= utils.eval_sklearn_clf(clf, test_images, test_labels)
    print(f"Nearest centroid classifier accuracy: {100*accuracy}%")
    print(f"Nearest centroid classifier infer time: {infdur}s")
    print(f"Nearest centroid classifier per-class precisions")
    print(precisions)
    print(f"Nearest centroid classifier per-class recalls")
    print(recalls)

    return accuracy, precisions, recalls, trdur, infdur


#helps visualize and compare statistics per class for multiple classifiers
def multibar_plot(title, labels, data1, data2, data3, data4):
    ind = np.arange(10)
    width = 0.2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, data1, width, color='royalblue')
    rects2 = ax.bar(ind+width, data2, width, color='seagreen')
    rects3 = ax.bar(ind+2*width, data3, width, color='darkorange')
    rects4 = ax.bar(ind+3*width, data4, width, color='firebrick')

    # add some
    ax.set_title(title)
    ax.set_xticks(ind + width)
    ax.set_xticklabels( [str(idx) for idx in ind] )

    ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), labels )


def viz_centroids(centroids):
    fig = plt.figure()
    for i, centroid in enumerate(centroids):
        ax = fig.add_subplot(2,5,i+1)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        img = 255*centroid.reshape(28,28)
        ax.imshow(img)


def main():
    train_images, train_labels, test_images, test_labels = utils.mnist()

    onenn_acc, onenn_pr, onenn_re, onenn_trdur, onenn_infdur = knn_classifier_test(train_images, train_labels, test_images, test_labels, 1)
    print("---------------------------------------")
    twonn_acc, twonn_pr, twonn_re, twonn_trdur, twonn_infdur = knn_classifier_test(train_images, train_labels, test_images, test_labels, 2)
    print("---------------------------------------")
    threenn_acc, threenn_pr, threenn_re, threenn_trdur, threenn_infdur = knn_classifier_test(train_images, train_labels, test_images, test_labels, 3)
    print("---------------------------------------")
    nc_acc, nc_pr, nc_re, nc_trdur, nc_infdur = nearest_centroid_classifier_test(train_images, train_labels, test_images, test_labels)

    labels = ('1-NN', '2-NN', '3-NN', 'Nearest centroid')
    multibar_plot('Precision per class', labels, onenn_pr, twonn_pr, threenn_pr, nc_pr)
    multibar_plot('Recall per class', labels, onenn_re, twonn_re, threenn_re, nc_re)
    viz_utils.bar_plot('Total accuracy', labels, [onenn_acc, twonn_acc, threenn_acc, nc_acc])
    viz_utils.bar_plot('Train time', labels, [onenn_trdur, twonn_trdur, threenn_trdur, nc_trdur])
    viz_utils.bar_plot('Infer time', labels, [onenn_infdur, twonn_infdur, threenn_infdur, nc_infdur])
    plt.show()

if __name__ == '__main__':
    main()
