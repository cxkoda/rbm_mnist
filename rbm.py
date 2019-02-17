import numpy as np
import pickle

def sigmoid(x):
	return 1. / (1. + np.exp(-x))

class RBM:
	def __init__(self, nVisible, nHidden, randomSeed=None):
		self.nVisible = nVisible
		self.nHidden = nHidden
		self.nParams = nVisible * nHidden + nVisible + nHidden

		self.rng = np.random.RandomState(randomSeed)
		self.weights = np.asarray(self.rng.uniform(low=-0.01, high=0.01, size=self.nParams))

		# create some views on the stored weights
		iStart = 0
		iEnd = nVisible * nHidden
		self.W = self.weights[iStart: iEnd].reshape(nVisible, nHidden)

		iStart = iEnd
		iEnd += nVisible
		self.B = self.weights[iStart: iEnd]

		iStart = iEnd
		iEnd += nHidden
		self.C = self.weights[iStart: iEnd]

	def load(self, filename):
		f = open(filename, 'rb')
		tmp_dict = pickle.load(f)
		f.close()
		self.__dict__.update(tmp_dict)

	def save(self, filename):
		f = open(filename, 'wb')
		pickle.dump(self.__dict__, f, 2)
		f.close()

	def __repr__(self):
		return f'-------------\nW:\n{self.W}\n-------------\nB:\n{self.B}\n-------------\nC:\n{self.C}\n-------------'

	def logProb(self, visible, hidden):
		return (self.W.dot(hidden) + self.B).dot(visible) + self.C.dot(hidden)

	def probVisibleGivenHidden(self, visible, hidden):
		return sigmoid((2. * visible - 1.) * (np.dot(self.W, hidden).T + self.B).T)

	def probHiddenGivenVisible(self, visible, hidden):
		return sigmoid((2. * hidden - 1.) * (np.dot(self.W.T, visible).T + self.C).T)

	def sampleHidden(self, visible):
		probsOne = self.probHiddenGivenVisible(visible, 1)
		uniformSamples = self.rng.uniform(low=0, high=1, size=probsOne.shape)
		probSamples = np.array(uniformSamples < probsOne, dtype=np.int)
		return probSamples

	def sampleVisible(self, hidden):
		probsOne = self.probVisibleGivenHidden(1, hidden)
		uniformSamples = self.rng.uniform(low=0, high=1, size=probsOne.shape)
		probSamples = np.array(uniformSamples < probsOne, dtype=np.int)
		return probSamples

	def logProb_dW(self, visible, hidden):
		return np.einsum('ik,jk->ijk', visible, hidden)

	def logProb_dB(self, visible, hidden):
		return visible

	def logProb_dC(self, visible, hidden):
		return hidden

	def logProb_dTheta(self, visible, hidden):
		dW = self.logProb_dW(visible, hidden)
		dW = dW.reshape(-1, dW.shape[-1])

		return np.vstack((
			dW,
			self.logProb_dB(visible, hidden),
			self.logProb_dC(visible, hidden)
		))

	def get_params(self):
		return self.weights

	def update(self, deltaTheta):
		self.weights += deltaTheta

class gibbsSampler:
	def __init__(self, rbm, nMarkovChains, nMarkovIter):
		self.rbm = rbm
		self.nMarkovChains = nMarkovChains
		self.nMarkovIter = nMarkovIter

		self.markovVisible = np.zeros((rbm.nVisible, self.nMarkovChains), dtype=np.int)
		self.markovHidden = np.zeros((rbm.nHidden, self.nMarkovChains), dtype=np.int)

	def reset(self):
		self.markovVisible = np.random.randint(0, 2, size=self.markovVisible.shape)

	def sample(self, reset=False):
		if reset:
			self.reset()
		for _ in range(self.nMarkovIter):
			self.markovHidden = rbm.sampleHidden(self.markovVisible)
			self.markovVisible = rbm.sampleVisible(self.markovHidden)

		return self.markovVisible, self.markovHidden

class RBMTrainerPCB:
	def __init__(self, nMarkovChains):
		self.nMarkovChains = nMarkovChains

	def prepare(self, rbm, visibleData):
		self.markovVisible = np.zeros((rbm.nVisible, self.nMarkovChains), dtype=np.int)
		self.markovHidden = np.zeros((rbm.nHidden, self.nMarkovChains), dtype=np.int)

		self.markovVisible = np.random.randint(0, 2, size=self.markovVisible.shape)
		self.markovHidden = np.random.randint(0, 2, size=self.markovHidden.shape)

	def gibbs(self, rbm, nMarkovIter):
		self.markovVisible = np.random.randint(0,2, size=self.markovVisible.shape)
		for _ in range(nMarkovIter):
			self.markovHidden = rbm.sampleHidden(self.markovVisible)
			self.markovVisible = rbm.sampleVisible(self.markovHidden)

	def get_deltaTheta(self, rbm, visibleData, nMarkovIter):
		_visibleData = visibleData.T
		hiddenDataSample = rbm.sampleHidden(_visibleData)

		positivePhase = np.mean(rbm.logProb_dTheta(_visibleData, hiddenDataSample), axis=1)

		self.gibbs(rbm, nMarkovIter)
		negativePhase = np.mean(rbm.logProb_dTheta(self.markovVisible, self.markovHidden), axis=1)

		deltaTheta = positivePhase - negativePhase
		return deltaTheta

	def train(self, rbm, visibleData, learningRate, nMarkovIter, maxTrainingIter=10000, convergenceThreshold=1e-3):
		assert(isinstance(rbm, RBM))

		self.prepare(rbm, visibleData)

		for iTrainingIter in range(maxTrainingIter):
			try:
				deltaTheta = self.get_deltaTheta(rbm, visibleData, nMarkovIter)
			except Exception as e:
				print('Training Aborted: exception in get_delta:', e)
				break

			if np.isnan(deltaTheta).any():
				print('Training Aborted: nans detected')
				break


			rbm.update(learningRate * deltaTheta)

			relUpdate = np.linalg.norm(deltaTheta / rbm.get_params()) / rbm.nParams
			print(f'Training Iteration: {iTrainingIter} -> {relUpdate}')

			if relUpdate < convergenceThreshold:
				print('Training Converged')
				break

		print(rbm)


class RBMTrainerPCBBroyden(RBMTrainerPCB):
	def __init__(self, nMarkovChains):
		RBMTrainerPCB.__init__(self, nMarkovChains)
		self.deltaTheta_old = 0
		self.logProb_dTheta_old = 0

	def prepare(self, rbm, visibleData):
		RBMTrainerPCB.prepare(self, rbm, visibleData)
		self.inverseHessian = np.diagflat(np.ones(rbm.nParams))

		# First step is explicit in the broyden scheme
		_visibleData = visibleData.T
		hiddenDataSample = rbm.sampleHidden(_visibleData)

		positivePhase = np.mean(rbm.logProb_dTheta(_visibleData, hiddenDataSample), axis=-1)

		self.gibbs(rbm, 100)
		negativeVectors = rbm.logProb_dTheta(self.markovVisible, self.markovHidden)
		negativePhase = np.mean(negativeVectors, axis=-1)

		logProb_dTheta = positivePhase - negativePhase

		self.logProb_dTheta_old = logProb_dTheta

		hessian = np.mean(np.einsum('ik,jk->ijk', negativeVectors, negativeVectors), axis=-1) - np.outer(negativePhase, negativePhase)
		self.inverseHessian = np.linalg.inv(hessian)

		deltaTheta = self.inverseHessian @ logProb_dTheta
		rbm.update(deltaTheta)
		self.deltaTheta_old = deltaTheta


	def get_deltaTheta(self, rbm, visibleData, nMarkovIter):
		_visibleData = visibleData.T
		hiddenDataSample = rbm.sampleHidden(_visibleData)

		positivePhase = np.mean(rbm.logProb_dTheta(_visibleData, hiddenDataSample), axis=-1)

		self.gibbs(rbm, nMarkovIter)
		negativeVectors = rbm.logProb_dTheta(self.markovVisible, self.markovHidden)
		negativePhase = np.mean(negativeVectors, axis=-1)

		logProb_dTheta = positivePhase - negativePhase

		# # Broyden's Inverse update
		# temp1 = self.inverseHessian @ logProb_dTheta
		# temp2 = self.deltaTheta_old @ (self.deltaTheta_old + temp1)
		# self.inverseHessian -= np.outer(temp1, self.deltaTheta_old) @ self.inverseHessian / temp2
		# deltaTheta = - self.inverseHessian @ logProb_dTheta

		hessian = np.mean(np.einsum('ik,jk->ijk', negativeVectors, negativeVectors), axis=-1) - np.outer(negativePhase, negativePhase)
		deltaTheta = - np.linalg.solve(hessian.T @ hessian, hessian.T @ logProb_dTheta)

		self.deltaTheta_old = deltaTheta
		self.logProb_dTheta_old = logProb_dTheta

		return deltaTheta


if __name__ == "__main__":
	rbm = RBM(4, 2)

	# try:
	# 	rbm.load("simple.rbm")
	# except:
	# 	pass

	testData = np.array([[0,0,0,0], [1,1,0,0], [0,0,1,1], [1,1,1,1], [1,1,1,1]])

	trainer = RBMTrainerPCBBroyden(10000)
	trainer.train(rbm, testData, 1, 100, maxTrainingIter=100)

	rbm.save("simpleB.rbm")


	sampler = gibbsSampler(rbm, 10000, 1000)

	visibles, _ = sampler.sample()

	unique_elements, counts_elements = np.unique(visibles.T, axis=0, return_counts=True)

	for element, count in zip(unique_elements, counts_elements):
		print(element, count)
	#
	# for hidden in np.array([[0, 0], [1, 0], [0, 1]]):
	# 	for visible in testData:
	# 		print(visible, hidden, np.prod(rbm.probGivenHidden(visible, hidden)))




