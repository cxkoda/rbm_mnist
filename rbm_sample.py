from rbmLib.rbm import *
import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":
	rbm = RBM(filename="trainedRBMs/mnist_100.rbm")
	sampler = gibbsSampler(rbm)

	nSamplesX = 5
	nSamplesY = 3
	nSamples = nSamplesX * nSamplesY
	hidden = np.random.random_integers(0, 1, size=(rbm.nHidden, nSamples))

	for _ in range(400):
		visibles = rbm.sampleVisible(hidden)
		hidden = rbm.sampleHidden(visibles)

	visibles = rbm.probVisibleGivenHidden(1, hidden)

	f, axs = plt.subplots(nSamplesY, nSamplesX)
	plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

	if nSamples > 1:
		axs2 = []
		for sublist in axs:
			for item in sublist:
				axs2.append(item)
		axs = axs2
	else:
		axs = [axs]

	for i, visible in enumerate(visibles.T):
		axs[i].imshow(visible.reshape((28,28)))
		axs[i].axis('off')

	#plt.savefig('sample.pdf')
	plt.show()