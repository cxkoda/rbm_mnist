from rbmLib.rbm import *
from rbmLib.training import *



if __name__ == "__main__":
	rbm = RBM(4, 2)

	# try:
	# 	rbm.load("simpleHessian.rbm")
	# except:
	# 	pass

	testData = np.array([[0,0,0,0], [1,1,0,0], [0,0,1,1], [1,1,1,1], [1,1,1,1]])
	for _ in range(11):
		testData = np.vstack((testData, testData))
	np.random.shuffle(testData)

	print(len(testData))

	trainer = RBMTrainerPCDHessian()
	trainer.train(rbm, testData, learningRate=1, learningRateDecay=-1e-1, nMarkovChains=400, nMarkovIter=5,
				  epochs=1000, miniBatchSize=None, convergenceThreshold=1e-3, autosave=False)

	rbm.save("simpleHessian.rbm")
	print(rbm)

	sampler = gibbsSampler(rbm)

	nMarkovTest = 10000
	visibles, _ = sampler.sample(nMarkovChains=nMarkovTest, nMarkovIter=1000)

	unique_elements, counts_elements = np.unique(visibles.T, axis=0, return_counts=True)

	for element, count in zip(unique_elements, counts_elements):
		print(element, count/nMarkovTest)

