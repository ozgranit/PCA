import matplotlib.pyplot as plt
import sklearn.decomposition
from sklearn.datasets import fetch_lfw_people
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def plot_vector_as_image(image, h, w, title):
    """
    utility function to plot a vector as image.
    Args:
    image - vector of pixels
    h, w - dimensions of original pi
    """
    plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
    plt.title(title, size=12)
    # plt.show()
    #plt.savefig(title)
    #plt.clf()


def get_pictures_by_name(name):
    """
    Given a name returns all the pictures of the person with this specific name.
    """
    lfw_people = load_data()
    selected_images = []
    n_samples, h, w = lfw_people.images.shape
    target_label = list(lfw_people.target_names).index(name)
    for image, target in zip(lfw_people.images, lfw_people.target):
        if target == target_label:
            image_vector = image.reshape((h * w, 1))
            selected_images.append(image_vector)
    return selected_images, h, w


def load_data():
    # Don't change the resize factor!!!
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return lfw_people


def plot_pair(U1, U2, h, w, title):
    plt.subplot(1, 2, 1)
    plt.imshow(U1.reshape((h, w)), cmap=plt.cm.gray)
    plt.subplot(1, 2, 2)
    plt.imshow(U2.reshape((h, w)), cmap=plt.cm.gray)
    plt.tight_layout()
    # plt.show()
    #plt.savefig(title)
    #plt.clf()


def prepare_data():
    lfw_people = load_data()
    lst_names = lfw_people.target_names
    selected_images = []
    y_labels = []
    n_samples, h, w = lfw_people.images.shape
    all_targets = list()
    for name in lst_names:
        target_label = list(lfw_people.target_names).index(name)
        all_targets.append(target_label)
    for image, target in zip(lfw_people.images, lfw_people.target):
        if target in all_targets:
            image_vector = image.reshape((h * w, 1))
            selected_images.append(image_vector)
            y_labels.append(list(lfw_people.target_names)[target])
    return selected_images, y_labels, h, w


def make_mean_zero(X):
    mean = np.zeros((1, len(X[0])))
    for i in range(len(X)):
        mean = mean + X[i]
    mean = mean / len(X)
    for i in range(len(X)):
        X[i] = X[i] - mean


def get_eigen(A):
    eigenValues, eigenVectors = np.linalg.eig(A)
    eigenVectors = np.matrix.transpose(eigenVectors)
    ev_list = zip(eigenValues, eigenVectors)
    ev_list = sorted(ev_list, key=lambda tup: tup[0], reverse=True)
    eigenValues, eigenVectors = zip(*ev_list)
    return (eigenValues, eigenVectors)


def PCA(X, k):
    """
    Compute PCA on the given matrix.

    Args:
        X - Matrix of dimensions (n,d). Where n is the number of sample points and d is the dimension of each sample.
        For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimension of the matrix
        would be (10,100).
        k - number of eigenvectors to return

    Returns:
      U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest
      k eigenvectors of the covariance matrix.
      S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
    """
    U = None
    S = None
    A = np.matmul(np.matrix.transpose(X), X)
    eigenValues, eigenVectors = get_eigen(A)
    # Values & Vectors sorted large to small
    U = np.stack([eigenVectors[i] for i in range(k)], axis=0)
    S = np.array([eigenValues[i] for i in range(k)]).T
    return U, S


def main():
    all_images, y_labels, h, w = prepare_data()
    X = np.array(all_images)[:, :, 0]
    make_mean_zero(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.25, random_state=0)
    k_values = [1, 5, 10, 30, 50, 100, 150, 300, len(X[0])]
    lst_accuracy = list()
    for k in k_values:
        U, S = PCA(X_train, k)
        X_train_trans = np.matmul(X_train, np.matrix.transpose(U))
        X_test_trans = np.matmul(X_test, np.matrix.transpose(U))
        """
        parameters = {'C': [1, 10, 100, 1000], 'gamma': [0.00001, 0.0001, 0.001, 0.01, 1]}
        model = SVC()
        grid = GridSearchCV(estimator=model, param_grid=parameters)
        clf = grid.fit(X_train_trans, y_train)
        # summarize the results of the grid search
        print(grid.best_score_)
        print(grid.best_estimator_)
        """
        svc = SVC(C=1000, gamma=10 ** -7)
        clf = svc.fit(X_train_trans, y_train)
        lst_accuracy.append(clf.score(X_test_trans, y_test))
    plt.plot(k_values, lst_accuracy, color='m', marker='o')
    plt.xlabel('K values')
    plt.ylabel('Accuracy')
    #plt.savefig('Accuracy as a func of K')
    #plt.clf()
    plt.show()


if __name__ == '__main__':
    main()
