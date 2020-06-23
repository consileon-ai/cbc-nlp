"""
consileon.data.word2vec_tools
=============================

Tools and classes for integrating "consileon.data.tokens" und for
easing the training of word2vec/doc2vec models
"""

import consileon.data.tokens as tkns

import logging
import time

import multiprocessing
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from os.path import basename

from time import localtime
from time import strftime
import tqdm
import numpy as np
from numpy import dot
from numpy.linalg import norm as l2

def normalize(v) :
	"""
	Normalize vectors which don't have length 0
	"""
	l = l2(v)
	if l == 0 :
		l = 1.0
	return v / l

logger = logging.getLogger('consileon.data.word2vec_tools')

class TaggedDocuments(tkns.IteratorModifier) :
	def __call__(self, iterator) :
		if not iterator.is_tagged :
			raise Exception(
				"ToTaggedDocument: 'iterator' is not tagged - 'ToTaggedDocument' expects a tagged iterator."
			)
		def generator() :
			for x in iterator :
				yield TaggedDocument(words=x[0], tags=x[1])
		return tkns.Iterator(generator, is_tagged=False)

class GensimBaseWrap(tkns.IteratorConsumer) :
	def __init__(self,
		model_filename=None,
		append_time_str=True
	) :
		if model_filename is None :
			raise Exception("'model_filename' must not be 'None'")
		self.model_filename = model_filename
		if append_time_str :
			self.model_filename = self.model_filename + "_" + strftime("%Y%m%d_%H%M%S")
		self.model_logger = self.create_model_logger()
	def create_model_logger(self) :
		model_logger_fn = self.model_filename + ".log"
		handler = logging.FileHandler(model_logger_fn)
		handler.setFormatter(logging.Formatter('%(asctime)s %(name)s- %(levelname)s - %(message)s'))
		model_logger = logging.getLogger(basename(self.model_filename))
		model_logger.setLevel(logging.INFO)
		model_logger.addHandler(handler)
		return model_logger
	def get_iterator(self, iterator) :
		return iterator
	def __call__(self, iterator) :
		try :
			model = self.model
			model_logger = self.model_logger
			model_logger.info("starting")
			logger.info("starting")
			start = time.time()
			model_logger.info("build vocab")
			logger.info("build vocab")
			#
			i = self.get_iterator(iterator)
			model.build_vocab(i)
			#
			time_used = time.time() - start
			log = "built vocab, time used = %i s" % (time_used)
			model_logger.info(log)
			logger.info(log)
			#
			model.train(i, total_examples=model.corpus_count, **self.train_args)
			#
			log = "saving model to file %s" % (self.model_filename)
			model_logger.info(log)
			logger.info(log)
			log = "saving model to file %s" % (self.model_filename)
			model_logger.info(log)
			logger.info(log)
			model.save(self.model_filename)
			model_logger.info("finished.")
			logger.info("finished.")
			self.model = model
			return self
		except Exception as e :
			logger.error("error creating model %s" % (self.model_filename), exc_info=True)

class Word2VecWrap(GensimBaseWrap) :
	def __init__(self,
		model_filename="w2v_model",
		append_time_str=True,
		replace=True,
		**kwargs
	) :
		train_keys = [
			'epochs', 'start_alpha', 'end_alpha', 'queue_factor', 'report_delay',
			'compute_loss', 'callbacks'
		]
		self.train_args = dict((k, kwargs[k]) for k in train_keys if k in kwargs)
		model_keys = [
			'size', 'window', 'min_count', 'sg', 'hs', 'negative',
			'ns_exponent', 'cbox_mean', 'alpha', 'min_alpha', 'seed', 'max_vocab_size',
			'max_final_vocab', 'sample', 'hashfxn', 'iter', 'trim_rule'
		]
		self.model_args = dict((k, kwargs[k]) for k in model_keys if k in kwargs)
		self.replace = replace
		super(Word2VecWrap, self).__init__(model_filename=model_filename, append_time_str=append_time_str)
		self.model = Word2Vec(**self.model_args)
		if self.train_args.get('epochs') is None :
			self.train_args['epochs'] = self.model.epochs
		self.model_logger.info("input model args %s" % (self.model_args))
		self.model_logger.info("input train args %s" % (self.train_args))
		self.model_logger.info("model string %s" % (self.model))

class Doc2VecWrap(GensimBaseWrap) :
	def __init__(self,
		model_filename="d2v_model",
		append_time_str=True,
		**kwargs
	) :
		train_keys = [
			'epochs', 'start_alpha', 'end_alpha', 'word_count', 'queue_factor', 'report_delay',
			'callbacks'
		]
		self.train_args = dict((k, kwargs[k]) for k in train_keys if k in kwargs)
		model_keys = [
			'dm', 'vector_size', 'window', 'alpha', 'min_alpha', 'seed', 'min_count', 'max_vocab_size',
			'sample', 'workers', 'epochs', 'hs', 'negative', 'ns_exponent', 'dm_mean', 'dm_concat',
			'dm_tag_count', 'dbow_words', 'trim_rule'
		]
		self.model_args = dict((k, kwargs[k]) for k in model_keys if k in kwargs)
		super(Doc2VecWrap, self).__init__(model_filename=model_filename, append_time_str=append_time_str)
		self.model_logger.info("input train args %s" % (self.train_args))
		self.model = Doc2Vec(**self.model_args)
		self.model_logger.info("model string %s" % (self.model))
	def get_iterator(self, iterator) :
		return TaggedDocuments() ** iterator

def as_list(a) :
	result = None
	if isinstance(a, str) :
		result = [a]
	elif isinstance(a, list) :
		result = a
	elif isinstance(a, tuple) :
		result = list(a)
	if result is None :
		raise Exception("input must be 'str' or 'list' (of 'str') or 'tuple' of ('str')")
	return result

def vector_cos(v, w) :
	l = l2(v) * l2(w)
	if l == 0 :
		l = 1.0
	return dot(v, w) / l

def str_to_wv(model, a) :
	wv = model.wv
	if a in wv :
		result = normalize(wv[a])
	else :
		result = np.zeros(model.vector_size)
	return result

def as_vector(model, a) :
	wv = model.wv
	l = as_list(a)
	if len(l) > 0 :
		result = sum([str_to_wv(model, w) for w in l])
	else :
		result = np.zeros(model.vector_size)
	return result

def cos(model, a, b) :
	return vector_cos(as_vector(model, a), as_vector(model, b))
