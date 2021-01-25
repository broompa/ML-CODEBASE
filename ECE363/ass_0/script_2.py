import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
#--------------------------------

digits = load_digits()
X = digits.data
print("X Shape:"+str(X.shape))
Y = digits.target
print("Y Shape:"+str(Y.shape))
Y_unique,counts = np.unique(Y,return_counts=True)
X_images = 	np.reshape(X,(X.shape[0],8,8))
I = digits.images

# plt.gray()
# plt.matshow(I[4])
# plt.show()
#

plt.bar(Y_unique,counts/np.sum(counts))

plt.show()