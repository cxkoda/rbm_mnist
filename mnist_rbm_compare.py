from rbmLib.rbm import *
import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":
	dt = np.dtype('>u4, >u4, >u4, >u4, (10000,784)u1')
	mnist = np.fromfile('t10k-images.idx3-ubyte', dtype=dt)['f4'][0]
	imgs = np.zeros((10000, 784), dtype=np.dtype('b'))
	imgs[mnist > 127] = 1


	fig, ax = plt.subplots(2, 2)

	idx = np.random.randint(0, imgs.shape[0])
	ax[1, 0].imshow(mnist[idx].reshape((28,28)))

	rbm = RBM(filename="mnist_200.rbm")

	visibles = imgs[idx]
	hidden = rbm.sampleHidden(visibles)
	visibles = rbm.sampleVisible(hidden)

	ax[0, 1].imshow(visibles.reshape((28, 28)))
	visibles = rbm.probVisibleGivenHidden(1, hidden)

	ax[1, 1].imshow(visibles.reshape((28,28)))

	ax[0, 0].imshow(imgs[idx].reshape((28, 28)))


	plt.show()
