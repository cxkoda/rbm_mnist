from rbmLib.rbm import *
from rbmLib.training import *
import numpy as np



if __name__ == "__main__":
	dt = np.dtype('>u4, >u4, >u4, >u4, (10000,784)u1')
	mnist = np.fromfile('t10k-images.idx3-ubyte', dtype=dt)['f4'][0]
	imgs = np.zeros((10000, 784), dtype=np.dtype('b'))
	imgs[mnist > 127] = 1

	rbm = RBM(784, 100)

	try:
		rbm.load("trainedRBMs/mnist_100.rbm")
	except:
		pass

	trainer = RBMTrainerPCD()

	trainer.train(rbm, imgs,
				learningRate=1, learningRateDecay=-1e-1,
				nMarkovChains=100, nMarkovIter=1,
				epochs=50, convergenceThreshold=1e-8, miniBatchSize=100, autosave=1)



