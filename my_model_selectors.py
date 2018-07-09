import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
	'''
	base class for model selection (strategy design pattern)
	'''

	def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
				 n_constant=3,
				 min_n_components=2, max_n_components=10,
				 random_state=14, verbose=False):
		self.words = all_word_sequences
		self.hwords = all_word_Xlengths
		self.sequences = all_word_sequences[this_word]
		self.X, self.lengths = all_word_Xlengths[this_word]
		self.this_word = this_word
		self.n_constant = n_constant
		self.min_n_components = min_n_components
		self.max_n_components = max_n_components
		self.random_state = random_state
		self.verbose = verbose

	def select(self):
		raise NotImplementedError

	def base_model(self, num_states):
		# with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		# warnings.filterwarnings("ignore", category=RuntimeWarning)
		try:
			hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
									random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
			if self.verbose:
				print("model created for {} with {} states".format(self.this_word, num_states))
			return hmm_model
		except:
			if self.verbose:
				print("failure on {} with {} states".format(self.this_word, num_states))
			return None


class SelectorConstant(ModelSelector):
	""" select the model with value self.n_constant

	"""

	def select(self):
		""" select based on n_constant value65

		:return: GaussianHMM object
		"""
		best_num_components = self.n_constant
		return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
	""" select the model with the lowest Bayesian Information Criterion(BIC) score

	http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
	Bayesian information criteria: BIC = -2 * logL + p * logN
	"""

	def select(self):
		""" select the best model for self.this_word based on
		BIC score for n between self.min_n_components and self.max_n_components

		:return: GaussianHMM object
		"""
		# TODO implement model selection based on BIC scores
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		maxBIC = 0.0
		totalLogL = 0.0
		split_method = KFold(n_splits = 3, shuffle = False, random_state = None)
		best_num_components = self.n_constant
		try:
			for n_comp in range(self.min_n_components, self.max_n_components+1):
					tr_model =  self.base_model(n_comp)
					logL = tr_model.score(self.X, self.lengths)
					parameters = n_comp * n_comp + 2 * n_comp * len(self.X[0]) - 1 # Number of Transition probabilities + Emission Probabilities
					bic = (-2 * logL) + (parameters * np.log(len(self.X)))
					if bic > maxBIC:
						best_num_components = n_comp
						maxBIC= bic
		except:
			pass
		selected = self. base_model(best_num_components)   
		return selected


class SelectorDIC(ModelSelector):
	''' select best model based on Discriminative Information Criterion

	Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
	Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
	http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
	https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
	DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
	'''

	def select(self):
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		maxDIC = 0.0
		totalLogL = 0.0
		split_method = KFold(n_splits = 3, shuffle = False, random_state = None)
		best_num_components = self.n_constant
		try:
			for n_comp in range(self.min_n_components, self.max_n_components+1):
				tr_model =  self.base_model(n_comp)
				logL = tr_model.score(self.X, self.lengths)
				totalAntiLogL = self.calculate_antilikelihood(n_comp)
				dic = logL - (totalAntiLogL/(self.max_n_components - self.min_n_components))
				if dic > maxDIC:
					best_num_components = n_comp
					maxDIC= dic
		except:
			pass
		selected = self. base_model(best_num_components)   
		return selected

	def calculate_antilikelihood( self, n):
		totalLogL = 0.0
		tr_model =  self.base_model(n)
		for word,(X,lengths) in self.hwords.items():
			if(word != self.this_word):
				logL = tr_model.score(X, lengths)
				totalLogL = logL+totalLogL
		return totalLogL




class SelectorCV(ModelSelector):
	''' select best model based on average log Likelihood of cross-validation folds

	'''

	def select(self):
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		maxLogL = 0.0
		totalLogL = 0.0
		split_method = KFold(n_splits = 3, shuffle = False, random_state = None)
		best_num_components = self.n_constant
		for n_comp in range(self.min_n_components, self.max_n_components+1):
			try:
				for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
					Xn, LengthsN = combine_sequences(cv_test_idx,self.sequences)
					self.X, self.lengths = combine_sequences(cv_train_idx,self.sequences)
					tr_model =  self.base_model(n_comp)
					logL = tr_model.score(Xn, LengthsN)
					totalLogL = totalLogL + logL
					avgLogL = logL/totalLogL
					if avgLogL > maxLogL:
						best_num_components = n_comp
						maxLogL = logL
			except:
				pass


		selected = self. base_model(best_num_components)
		return selected
