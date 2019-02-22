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
	ax[0, 0].imshow(mnist[idx].reshape((28,28)))
	ax[0, 0].axis('off')
	ax[0, 0].set_title('Original Mnist')


	ax[0, 1].imshow(imgs[idx].reshape((28, 28)))
	ax[0, 1].axis('off')
	ax[0, 1].set_title('Binarized Mnist')

	rbm = RBM(filename="trainedRBMs/mnist_100.rbm")

	visibles = imgs[idx]
	hidden = rbm.sampleHidden(visibles)
	visibles = rbm.sampleVisible(hidden)

	ax[1, 0].imshow(visibles.reshape((28, 28)))
	ax[1, 0].axis('off')
	ax[1, 0].set_title('RBM Reconstructed')

	visibles = rbm.probVisibleGivenHidden(1, hidden)
	ax[1, 1].imshow(visibles.reshape((28,28)))
	ax[1, 1].axis('off')
	ax[1, 1].set_title('RBM Visible Probability')

	#plt.savefig('comparison.pdf')
	plt.show()
