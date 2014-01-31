import pickle
import numpy as np    

from sklearn import cross_validation

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA

class Classifier():

	USE_CACHED_VERSION = True

	def __init__(self, data_source):
		self.data_source = data_source
			
		self.color_names = []

		if self.USE_CACHED_VERSION:
			classifier_from_cache = self.load_classifier_from_cache()
			if classifier_from_cache:
				self.clf, self.X, self.Y, self.color_names = classifier_from_cache
		else: 
			self.X = []
			self.Y = []

			self.generate( self.data_source.db )
			self.cross_validation()
			self.save_classifier_to_cache()

	def load_classifier_from_cache(self):
		try:
			return pickle.load(open("data/cached_classifier.bin", "rb"))
		except:
			return False
	
	def save_classifier_to_cache(self):
		try:
			archived_classifier = [self.clf, self.X, self.Y, self.color_names]
			return pickle.dump( archived_classifier, open( "data/cached_classifier.bin", "wb" ) )
		except:
			return False
		
	def generate(self, dict_of_histograms):
		color_class_id = 0

		for key in dict_of_histograms.keys():			
			self.color_names.append( key )

			hists = dict_of_histograms[key]

			for hist in hists:
				
				if not np.all( np.isfinite( hist ) ):
					pass
				else:
					self.X.append( hist )
					self.Y.append( [color_class_id] )
				
			color_class_id = color_class_id + 1

		self.X = np.asarray( self.X , dtype=np.float64 )
		self.Y = np.asarray( self.Y , dtype=np.float64 )

		self.Y = np.ravel( self.Y )

	def cross_validation(self):
		X_train, X_test, y_train, y_test = cross_validation.train_test_split( self.X,self.Y, test_size=0.3, random_state=0 )

		print("")
		print( "X_train " )
		print( X_train.shape )
		print( "X_test" )
		print( X_test.shape )

		print( "y_train" )
		print( y_train.shape )
		print( "y_test" )
		print( y_test.shape )
		print("")

		print( "LogisticRegression: 1e5" )
		self.clf = LogisticRegression(C=1e5).fit(X_train, y_train)
		print( self.clf.score(X_test, y_test) )

		"""
		print( "LDA: " )
		clf = LDA().fit(X_train, y_train)
		print( clf.score(X_test, y_test) )

		print( "SVC: " )
		clf = SVC().fit(X_train, y_train)
		print( clf.score(X_test, y_test) )
		"""

	def colorname_by_index( self, index ):
		return self.color_names[ index ]

	def predict_from_filename(self, filename):
		return self.predict_from_histogram( self.data_source.histogram_from_filename( filename ) )

	def predict_from_histogram( self, histogram ):
		clf_predict = self.clf.predict_proba( histogram )

		unsorted_predictions = []

		for p in clf_predict:
			for i in range(len(p)):
				if p[i] > 0.0:
					unsorted_predictions.append( [color_names[i], p[i] ] )

		sorted_predictions = sorted( unsorted_predictions, key=itemgetter(1), reverse=True)

		for res in sorted_predictions: # add [:3]: to show to three
			color, percentage = res
			print( color + "\t" + str(int(100*percentage)).zfill(2) + "%" )
		print()

		return clf_predict
		