import numpy as np
import pickle

def sigmoid(x):
	return 1. / (1. + np.exp(-x))

class RBM:
	def __init__(self, nVisible, nHidden):
		self.nVisible = nVisible
		self.nHidden = nHidden

		self.W = np.ones((self.nVisible, self.nHidden))
		self.B = np.ones(self.nVisible)
		self.C = np.ones(self.nHidden)

		# self.W = np.random.random(self.W.shape)
		# self.B = np.random.random(self.B.shape)
		# self.C = np.random.random(self.C.shape)

		self.nParams = self.W.size + self.B.size + self.C.size

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

	def probGivenHidden(self, visible, hidden):
		return sigmoid((2. * visible - 1.) * (np.dot(self.W, hidden).T + self.B).T)

	def probGivenVisible(self, visible, hidden):
		return sigmoid((2. * hidden - 1.) * (np.dot(self.W.T, visible).T + self.C).T)

	def sampleHidden(self, visible):
		probsOne = self.probGivenVisible(visible, 1)
		uniformSamples = np.random.rand(*probsOne.shape)
		probSamples = np.array(uniformSamples < probsOne, dtype=np.int)
		return probSamples

	def sampleVisible(self, hidden):
		probsOne = self.probGivenHidden(1, hidden)
		uniformSamples = np.random.rand(*probsOne.shape)
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
		return np.concatenate((
			np.ravel(self.W),
			self.B,
			self.C
		))

	def update(self, deltaTheta):
		iStart = 0
		deltaW = deltaTheta[iStart:iStart + self.W.size].reshape(self.W.shape)
		iStart += self.W.size
		deltaB = deltaTheta[iStart:iStart + self.B.size]
		iStart += self.B.size
		deltaC = deltaTheta[iStart:iStart + self.C.size]

		self.W += deltaW
		self.B += deltaB
		self.C += deltaC

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

	def prepare(self, rbm):
		self.markovVisible = np.zeros((rbm.nVisible, self.nMarkovChains), dtype=np.int)
		self.markovHidden = np.zeros((rbm.nHidden, self.nMarkovChains), dtype=np.int)

	def gibbs(self, rbm, nMarkovIter):
		# self.markovVisible = np.random.randint(0,1, size=self.markovVisible.shape)

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

		self.prepare(rbm)

		for iTrainingIter in range(maxTrainingIter):
			deltaTheta = self.get_deltaTheta(rbm, visibleData, nMarkovIter)
			rbm.update(learningRate * deltaTheta)

			relUpdate = np.linalg.norm(deltaTheta / rbm.get_params()) / rbm.nParams
			print(f'Training Iteration: {iTrainingIter} -> {relUpdate}')

			if relUpdate < convergenceThreshold:
				print('Training Converged')
				print(rbm)
				break


if __name__ == "__main__":
	rbm = RBM(4, 2)

	try:
		rbm.load("simple.rbm")
	except:
		pass

	testData = np.array([[0,0,0,0], [1,1,0,0], [0,0,1,1], [1,1,1,1], [1,1,1,1]])

	trainer = RBMTrainerPCB(40)
	trainer.train(rbm, testData, 0.01, 20)

	rbm.save("simple.rbm")
	print(rbm)

	for hidden in np.array([[0, 0], [1, 0], [0, 1]]):
		for visible in testData:
			print(visible, hidden, np.prod(rbm.probGivenHidden(visible, hidden)))




