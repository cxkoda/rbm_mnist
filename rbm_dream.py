from rbmLib.rbm import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation



if __name__ == "__main__":
	rbm = RBM(filename="trainedRBMs/mnist_100.rbm")
	fig = plt.figure()

	hidden = np.random.random_integers(0, 1, size=rbm.nHidden)
	visible = rbm.sampleVisible(hidden)
	hidden = rbm.sampleHidden(visible)
	visiblesReal = rbm.probVisibleGivenHidden(1, hidden).reshape((28,28))

	im = plt.imshow(visiblesReal, animated=True)
	plt.axis('off')

	def updatefig(*args):
		global visible, hidden, visiblesReal
		visible = rbm.sampleVisible(hidden)
		hidden = rbm.sampleHidden(visible)
		# flipSelection = np.random.uniform(0, 1, size=hidden.shape) < 0.05
		# hidden[flipSelection] = 1 - hidden[flipSelection]
		visiblesReal = rbm.probVisibleGivenHidden(1, hidden).reshape((28,28))

		im.set_array(visiblesReal)
		return im,


	ani = animation.FuncAnimation(fig, updatefig, interval=5, blit=True)
	plt.show()