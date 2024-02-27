from sklearn import datasets
from sklearn import svm

import matplotlib.pyplot as plt

iris = datasets.load_iris()
digits = datasets.load_digits()

img = digits.images[0].reshape(8, 8)
clf = svm.SVC(gamma=0.001, C=100.)

plt.imshow(img, cmap='gray')
plt.title(f'Etiqueta: {digits.target[0]}')
plt.show()

