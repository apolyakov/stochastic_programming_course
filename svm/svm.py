from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def draw_models(models, titles, objects):
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = objects[:, 0], objects[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=data[:, 2], cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()


def classify(objects, classes):
    models = []
    titles = []
    metrics = ['balanced_accuracy', 'average_precision']

    svc_parameters = {'kernel': ('linear', 'poly', 'rbf'),
                      'C': np.arange(1., 10., 0.1),
                      'gamma': ['auto', 'scale']}
    knn_parameters = {'n_neighbors': range(1, 10),
                      'algorithm': ['auto', 'brute', 'kd_tree', 'ball_tree']}

    for metric in metrics:
        svc = GridSearchCV(SVC(), svc_parameters, cv=5, scoring=metric).fit(objects, classes)
        knn = GridSearchCV(KNeighborsClassifier(), knn_parameters, cv=5,
                           scoring=metric).fit(objects, classes)
        models.append(svc.best_estimator_)
        titles.append('SVC algorithm with the %s metric = %f' % (metric, svc.best_score_))
        models.append(knn.best_estimator_)
        titles.append('KNN algorithm with the %s metric = %f' % (metric, knn.best_score_))

    draw_models(models, titles, objects)


if __name__ == '__main__':
    data = np.loadtxt('chips.txt', delimiter=',')
    objects, classes = data[:, :2], data[:, 2]
    classify(objects, classes)
