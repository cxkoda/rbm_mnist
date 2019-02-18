from rbmLib.rbm import *
from rbmLib.training import *
import numpy as np



if __name__ == "__main__":
	dt = np.dtype('>u4, >u4, >u4, >u4, (10000,784)u1')
	mnist = np.fromfile('t10k-images.idx3-ubyte', dtype=dt)['f4'][0].T
	imgs = np.zeros((784, 10000), dtype=np.dtype('b'))
	imgs[mnist > 127] = 1
	imgs = imgs.T

	rbm = RBM(784, 100)

	try:
		rbm.load("mnist.rbm")
	except:
		pass

	trainer = RBMTrainerPCD()

	trainer.train(rbm, imgs,
				learningRate=0.1, learningRateDecay=-1e-1,
				nMarkovChains=100, nMarkovIter=5,
				epochs=1000, convergenceThreshold=1e-8, miniBatchSize=100)


