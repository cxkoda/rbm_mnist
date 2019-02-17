from rbmLib.rbm import *
from rbmLib.training import *



if __name__ == "__main__":
	rbm = RBM(4, 2)

	try:
		rbm.load("simple.rbm")
	except:
		pass

	testData = np.array([[0,0,0,0], [1,1,0,0], [0,0,1,1], [1,1,1,1], [1,1,1,1]])

	trainer = RBMTrainerPCB()
	trainer.train(rbm, testData, learningRate=1, nMarkovChains=100, nMarkovIter=100, maxTrainingIter=1000)

	rbm.save("simple.rbm")


	sampler = gibbsSampler(rbm)

	visibles, _ = sampler.sample(nMarkovChains=10000, nMarkovIter=1000)

	unique_elements, counts_elements = np.unique(visibles.T, axis=0, return_counts=True)

	for element, count in zip(unique_elements, counts_elements):
		print(element, count)

