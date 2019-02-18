from rbmLib.rbm import *
import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":
	rbm = RBM(filename="mnist_300.rbm", randomSeed=1234)
	sampler = gibbsSampler(rbm)

	nSamples = 10

	hidden = np.random.random_integers(0, 1, size=(rbm.nHidden, nSamples))

	# visibles, _ = sampler.sample(nMarkovChains=nSamples, nMarkovIter=1000)
	for _ in range(1000):
		visibles = rbm.sampleVisible(hidden)
		hidden = rbm.sampleHidden(visibles)

	visibles = rbm.probVisibleGivenHidden(1, hidden)


	for i, visible in enumerate(visibles.T):
		plt.figure()
		plt.imshow(visible.reshape((28,28)))


	plt.show()
	exit(1)
	for i, visible in enumerate(visibles.T):
		print(i)
		plt.axis('off')
		plt.imshow(visible.reshape((28,28)))
		plt.savefig(f"digits/digit{i:04}.png", bbox_inches='tight')

	plt.show()
