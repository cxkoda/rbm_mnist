import numpy as np
import pickle
import os

def sigmoid(x):
	return 1. / (1. + np.exp(-x))

class RBM:
	def __init__(self, nVisible=0, nHidden=0, randomSeed=None, filename=None):
		self.rng = np.random.RandomState(randomSeed)

		if filename is not None:
			self.__load_unsave(filename)
			return

		self.nVisible = nVisible
		self.nHidden = nHidden
		self.nParams = nVisible * nHidden + nVisible + nHidden

		self.weights = np.asarray(self.rng.uniform(low=-0.01, high=0.01, size=self.nParams))

		self.nTrainedEpochs = 0

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

	def __load(self, filename):
		self.filename = filename
		if not os.path.isfile(filename):
			raise FileExistsError('RBM load path does not exist')
		print(f'Loading RBM from {filename}')
		f = open(filename, 'rb')
		tmpDict = pickle.load(f)
		f.close()
		try:
			del tmpDict['filename']
		except:
			pass
		return tmpDict

	def load(self, filename):
		tmpDict = self.__load(filename)
		loadedNVisible = tmpDict['nVisible']
		loadedNHidden = tmpDict['nHidden']
		if self.nVisible != loadedNVisible or self.nHidden != loadedNHidden:
			raise RuntimeError('Node number mismatch. Loaded RBM has %i visible and %i hidden nodes' %
								(loadedNVisible, loadedNHidden))
		self.__dict__.update(tmpDict)

	def __load_unsave(self, filename):
		tmpDict = self.__load(filename)
		self.__dict__.update(tmpDict)

	def save(self, filename=None):
		if filename is None:
			if not hasattr(self, 'filename'):
				raise RuntimeError('No save path for the RBM was specified.')
			filename = self.filename
		else:
			self.filename = filename

		print(f'Writing RBM to {filename}')
		f = open(filename, 'wb')
		tmpDict = self.__dict__.copy()
		del tmpDict['filename']
		del tmpDict['rng']
		pickle.dump(tmpDict, f, 2)
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



