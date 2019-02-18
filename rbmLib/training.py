from rbmLib.rbm import *
import numpy as np
import datetime


class RBMTrainerPCD:
	def __init__(self,  randomSeed=None):
		self.randomSeed = randomSeed
		self.rng = np.random.RandomState(randomSeed)

	def prepare(self, rbm, visibleData, miniBatchSize):
		self.gibbs = gibbsSampler(rbm, randomSeed=self.randomSeed)

	def get_deltaTheta(self, rbm, visibleData, nMarkovChains, nMarkovIter):
		hiddenDataSample = rbm.sampleHidden(visibleData)

		positivePhase = np.mean(rbm.logProb_dTheta(visibleData, hiddenDataSample), axis=1)

		sampledVisible, sampledHidden = self.gibbs.sample(nMarkovChains, nMarkovIter)
		sampledHidden = rbm.probHiddenGivenVisible(sampledVisible, 1)
		negativePhase = np.mean(rbm.logProb_dTheta(sampledVisible, sampledHidden), axis=1)

		deltaTheta = positivePhase - negativePhase
		return deltaTheta

	def train(self, rbm, visibleData,
				learningRate=0.1, learningRateDecay=0,
				nMarkovChains=100, nMarkovIter=10,
				epochs = 100, miniBatchSize=1000, convergenceThreshold=1e-3,
				autosave = 10
			):
		assert(isinstance(rbm, RBM))
		self.prepare(rbm, visibleData, miniBatchSize)

		if miniBatchSize is None:
			miniBatchSize = len(visibleData)

		nConvergesScores = 3
		convergenceScores = [1] * nConvergesScores

		nMiniBatches = int(len(visibleData) / miniBatchSize)
		iEpoch = 0
		trainingStartTime = datetime.datetime.now()
		while iEpoch < epochs:
			iEpoch += 1
			epochStartTime = datetime.datetime.now()
			iScore = iEpoch % nConvergesScores
			convergenceScores[iScore] = 0

			currentLearningRate = learningRate * np.power(iEpoch, learningRateDecay)
			print(f'Epoch: {iEpoch}')
			print(f'\tCurrent LearningRate:               {currentLearningRate}')

			self.rng.shuffle(visibleData)
			for iMiniBatch in range(nMiniBatches):
				miniBatch = visibleData[iMiniBatch * miniBatchSize : (iMiniBatch+1) * miniBatchSize].T
				deltaTheta = self.get_deltaTheta(rbm, miniBatch, nMarkovChains, nMarkovIter)

				if np.isnan(deltaTheta).any():
					raise RuntimeError('Nans detected during training')

				rbm.update(currentLearningRate * deltaTheta)

				relUpdate = np.linalg.norm(deltaTheta / rbm.get_params()) / rbm.nParams
				convergenceScores[iScore] += relUpdate

			epochEndTime = datetime.datetime.now()
			print(f'\tTime elapsed for epoch:             {epochEndTime - epochStartTime}')
			print(f'\tTotal training time:                {datetime.datetime.now() - trainingStartTime}')

			convergenceScores[iScore] /= nMiniBatches
			print(f'\tAveraged relative MiniBatch-Update: {convergenceScores[iScore]}')

			rbm.nTrainedEpochs += 1
			print(f'\tTotal epochs trained:               {rbm.nTrainedEpochs}')
			print(f'')

			if autosave > 0:
				if iEpoch % autosave == 0:
					rbm.save()

			if (np.array(convergenceScores) < convergenceThreshold).all():
				print('Training Converged')
				break

		if autosave > 0:
			rbm.save()

class RBMTrainerPCDHessian(RBMTrainerPCD):
	def __init__(self, randomSeed=None):
		RBMTrainerPCD.__init__(self, randomSeed)

	def prepare(self, rbm, visibleData, miniBatchSize):
		visibleData = RBMTrainerPCD.prepare(self, rbm, visibleData, miniBatchSize)
		self.hessian = np.diagflat(np.ones(rbm.nParams))
		self.deltaTheta_old = 0
		self.logProb_dTheta_old = 0

		return visibleData


	def get_deltaTheta(self, rbm, visibleData, nMarkovChains, nMarkovIter):
		hiddenDataSample = rbm.sampleHidden(visibleData)

		positivePhase = np.mean(rbm.logProb_dTheta(visibleData, hiddenDataSample), axis=-1)

		sampledVisible, sampledHidden = self.gibbs.sample(nMarkovChains, nMarkovIter)
		sampledHidden = rbm.probHiddenGivenVisible(sampledVisible, 1)
		negativeVectors = rbm.logProb_dTheta(sampledVisible, sampledHidden)
		negativePhase = np.mean(negativeVectors, axis=-1)

		logProb_dTheta = positivePhase - negativePhase

		hessian = - np.mean(np.einsum('ik,jk->ijk', negativeVectors, negativeVectors), axis=-1) + np.outer(negativePhase, negativePhase)
		update = - hessian + np.identity(hessian.shape[0])
		negHessian = - hessian

		deltaTheta = logProb_dTheta
		Ainv = 1. / (np.diagonal(negHessian) + 1e-20 * np.linalg.norm(deltaTheta))
		B = (np.triu(negHessian) + np.tril(negHessian))

		for _ in range(20):
			Ainv @ (logProb_dTheta - B @ deltaTheta)



		self.deltaTheta_old = deltaTheta
		self.logProb_dTheta_old = logProb_dTheta

		return deltaTheta