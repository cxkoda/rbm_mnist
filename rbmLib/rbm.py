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
	def __init__(self, rbm, randomSeed=None):
		self.rbm = rbm
		self.rng = np.random.RandomState(randomSeed)
		self.nMarkovChains = 0

	def resetMarkovChains(self, nMarkovChains):
		self.nMarkovChains = nMarkovChains
		self.markovVisible = self.rng.random_integers(0, 1, size=(self.rbm.nVisible, self.nMarkovChains))
		self.markovHidden = self.rng.random_integers(0, 1, size=(self.rbm.nHidden, self.nMarkovChains))


	def sample(self, nMarkovChains=100, nMarkovIter=100):
		if self.nMarkovChains != nMarkovChains:
			self.resetMarkovChains(nMarkovChains)

		for _ in range(nMarkovIter):
			self.markovHidden = self.rbm.sampleHidden(self.markovVisible)
			self.markovVisible = self.rbm.sampleVisible(self.markovHidden)

		return self.markovVisible, self.markovHidden



