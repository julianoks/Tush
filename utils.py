import random
from functools import reduce

class random_sampler(object):
	def __init__(self, probs):
		self.probs = self.normalize_probs(probs)

	def normalize_probs(self, pdict):
		for v in pdict.values(): assert v>=0, "Probabilities need be nonnegative"
		total = sum([v for v in pdict.values()])
		assert total > 0, "Total probability must be positive"
		partial, probs = 0, []
		for k,v in pdict.items():
			partial += v / total
			probs.append([k, partial])
		probs[-1][-1] = 1.1 # just bigger than 1
		return probs

	def pick_event(self):
		x = random.random()
		for k,v in self.probs:
			if v>x: return k

prod = lambda vec: reduce(lambda x1,x2: x1*x2, vec)
