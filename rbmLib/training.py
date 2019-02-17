from rbmLib.rbm import *
import numpy as np


class RBMTrainerPCB:
	def __init__(self, nMarkovChains, randomSeed = None):
		self.nMarkovChains = nMarkovChains
		self.rng = np.random.RandomState(randomSeed)

	def prepare(self, rbm, visibleData):
		self.markovVisible = np.zeros((rbm.nVisible, self.nMarkovChains), dtype=np.int)
		self.markovHidden = np.zeros((rbm.nHidden, self.nMarkovChains), dtype=np.int)

		self.markovVisible = self.rng.random_integers(0, 1, size=self.markovVisible.shape)
		self.markovHidden = self.rng.random_integers(0, 1, size=self.markovHidden.shape)

	def gibbs(self, rbm, nMarkovIter):
		self.markovVisible = self.rng.random_integers(0, 1, size=self.markovVisible.shape)
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

