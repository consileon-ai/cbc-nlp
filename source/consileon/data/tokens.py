"""
consileon.data.tokens
=====================

Frameword and tools for stream based processing of textual data.

The python concept of an "iterator" is heavily used.
This is a python class implementing the "iterator protocol", which consists of
__iter__() and __next__(). An iterator is something like an iterable
which can be reset by calling __iter__(). Therefore the elements of a loop can
be passed through many times.

Tokens implements two basic ideas which simplify working with iterators:

1)
	Modifiers on iterator items "also work on iterators". Those modifiers can be composed
	using the operator "*".

	Example:

	::

		>>> import consileon.data.tokens as tkns
		>>> m = tkns.Lower() * tkns.LemmaTokenizeText()
		>>> m("Bälle sind meißt rund.")
		['ball', 'sein', 'meißt', 'rund', '.']

2) 
	Modifiers on iterators can be composed ("pipelined") using the operator "**".

Iterators may be used as *document input* for word2vec training.
"""
import spacy
import nltk
import re
import xml.etree.ElementTree as ET
from nltk.tokenize import TreebankWordTokenizer

import logging
import numbers
from collections import Counter

import random
import string
import ast

logger = logging.getLogger('consileon.data.tokens')

nltk.download('stopwords')

STANDARD_STOPWORD=nltk.corpus.stopwords.words('german')
STANDARD_FILTER_SYMBOLS=['|', '*', '``', "''", '“', '„', '–', '-', '"', ')', '(', "'", ".", ",", '`', ":", "?", ";", "‘", "{", "}", "#", "&", "!", "]", "[", "%", "−", "..."]

RE_NUMBER=re.compile(r"^\d+[\.,eE]?\d*?$")
RE_SINGLE_LETTER=re.compile(r"^\w$")
RE_REPLACE_SPACE_CHARS = re.compile(r'[\|\"\/\(\)—]')
RE_REMOVE_CHARS = re.compile(r"[\\'\\-]")

RESTR_STD_PARAGRAPH_DEL = "\s*\n\s*\n\s*"
"""
Standard paragraph delimiter: At least two newlines which may include and be surrounded by
further arbitrary space
"""

RE_WHITESPACE = re.compile(r"\s+")

LEMMATIZE_MAX_SIZE = 10000

LEMMATIZE_MAX_CHUNK_SIZE = 100000

STANDARD_SEPARATOR = ":-):-|:-("

VOWELS = "aeiouäöüyAEIOUÄÖÜY"
CONSONANTS = "bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXY"

def splitInChunks(text, maxChunkLength=LEMMATIZE_MAX_CHUNK_SIZE) :
	"""
	Split a text in chunks of given maximum length splitting only at white space
	(leaving words intact)
	"""	
	l = len(text)
	i = 0
	j = l
	chunks = []
	while i < l :
		if j - i > maxChunkLength :
			j = text[0:i+maxChunkLength].rfind(" ")
		if j > i :
			chunks.append(text[i:j])
			i = j
			j = l
		else :
			break
	return chunks

class Iterator():
	"""
	Base class for reusable iterators
	"""
	def __init__(
		self,
		generator_function,
		is_tagged=False
	) :
		"""
		Args:
			:generator_function (function): A function returning a python iterable of items
	
		Kwargs:
			:is_tagged (boolean, optional, default=True): The resuling iterator is producing "tagged" items
		"""
		self.generator_function = generator_function
		self.generator = self.generator_function()
		self.is_tagged=is_tagged
	def __iter__(self) :
		self.generator = None
		self.generator = self.generator_function()
		return self
	def __next__(self) :
		try :
			result = next(self.generator)
			return result
		except :
			raise StopIteration

class SelfIterable :
	"""
	An iterable having itself as generator function.
	"""
	def __call__(self) :
		yield ["dummy"]
	def __init__(self, is_tagged=False) :
		self.is_tagged=is_tagged	
		self.generator = self.__call__()
	def __iter__(self):
		self.generator = None
		self.generator = self.__call__()
		return self
	def __next__(self) :
		result = next(self.generator)
		if result is None :
			raise StopIteration
		else :
			return result

class IteratorModifier :
	"""
	Base class for all classes which modify an iterator and which are not working
	'element wise' on items. E.g. "Merge", "SplitText".
	"""
	def __pow__(self, iterator) :
		return self.__call__(iterator)
	def __call__(self, iterator) :
		def generator() :
			for x in iterator :
				if (x is not None) :
					yield x
		return Iterator(generator, iterator.is_tagged)

class IteratorConsumer :
	"""
	Base class for operations which somehow consume an iterator like writing
	its items to a file or applying some KI method on its items.
	"""
	def __pow__(self, iterator) :
		return self.__call__(iterator)
	def __call__(self, iterator) :
		iterator.__iter__()
		print(str(next(iterator)))
		return self
	

class ItemModifier :
	'''

	Base class for operators on arbitrary objects which may serve as items
	of iterators.

	The class is the base class of many "predefined" modifiers, but also has
	but can also be used directly. It is initialized with a function operating
	on items:

	::

		import consileon.data.tokens as tkns
		dublicate = tkns.ItemModifier(f=lambda l : l * 2)
		m = dublicate * tkns.TokenizeText()
		m("Der Ball ist rund.")

		['Der', 'Ball', 'ist', 'rund', '.', 'Der', 'Ball', 'ist', 'rund', '.']

	'''

	def __create_call_function(f, is_tagged) :
		if is_tagged :
			def cf(x) :
				f_x0 = f(x[0])
				if f_x0 is not None :
					return (f_x0, x[1])
				else :
					return None
			result = cf
		else :
			result = f
		return result	
	def __init__(self, f = lambda x:x) :
		"""
		Initialize object with function operating on item
			
		:f (function): the function operating on the items
		"""
		self.f = f
	def __call__(self, x) :
		return self.f(x)
	def __mul__(self, other) :
		return MulItemModifier(self, other)
	def __pow__(self, other) :
		return self.applyToIterator(other)
	def applyToIterator(self, iterator) :
		cf = ItemModifier.__create_call_function(self.f, iterator.is_tagged)
		def generator() :
			for x in iterator :
				if (x is not None) :
					x = cf(x)
				if (x is not None) :
					yield x
		return Iterator(generator, iterator.is_tagged)
	
class MulItemModifier(ItemModifier) :
	"""
	Class used interally to *compose* to ItemModifiers
	"""
	def __init__(self, left, right) :
		"""
			Args:

				:left (ItemModifier): left side of composition

				:right (ItemModifier): right side of composition
		"""
		self.left = left
		self.right = right
		f = lambda x : self.left.f( self.right.f(x) )
		super(MulItemModifier, self).__init__(f = f)
	def __mul__(self, other) :
		return MulItemModifier(self, other)
	def __pow__(self, other) :
		return self.applyToIterator(other)

class Lower(ItemModifier) :
	"""
	Expects a list of strings as item.
	Transforms each member of the list to lower case.
	"""
	def __init__(self) :
		"""
		(No arguments allowed.)
		"""
		super(Lower, self).__init__(
			f = lambda tokens : [t.lower() for t in tokens]
		)

class Re(ItemModifier) :
	def __init__(self, reg_ex) :
		my_re = re.compile(reg_ex)
		super(Re, self).__init__(
			f = lambda t : list(filter(lambda w : my_re.match(w) is None, t))
		)

class ReSub(ItemModifier) :
	def __init__(self, reg_ex_list, replace) :
		my_re_list = tuple([re.compile(reg_ex) for reg_ex in reg_ex_list])
		def f(w) :
			x = w
			for my_re in my_re_list :
				x = my_re.sub(replace, x)
			return x
		super(ReSub, self).__init__(f=f)

class ReSplit(ItemModifier) :
	def __init__(self, reg_ex) :
		my_re = re.compile(reg_ex)
		super(ReSplit, self).__init__(
			f = lambda t : my_re.split(t)
		)

class Append(ItemModifier) :
	def __init__(self, append="_DE") :
		self.append=append
		super(Append, self).__init__(
			f = lambda tokens : [t+self.append for t in tokens]
		)

class LowerAppend(ItemModifier) :
	def __init__(self, append="_DE") :
		self.append=append
		f = lambda : [t.lower()+self.append for t in tokens]
		super(LowerAppend, self).__init__( f = f )

class IsNLText(ItemModifier) :
	"""
	Check whether a string consist of Natural Language (NL) Text using
	heuristics on distribution of characters.

	Kwargs:
		:lb_vows_by_letters (float, default=0.25): lower bound on "vowels by letters"
		:ub_vows_by_letters (float, default=0.53): upper bound on "vowels by letters"
		:lb_letters_by_chars (float, default=0.67): lower bound on "letters by all chars"
		:ub_spaces_by_chars (float, default=0.2): lower bound on "spaces by all chars"
		:ub_digits_by_chars (float, default=0.2): lower bound on "digits by all chars"
	"""
	def __init__(self,
		lb_vows_by_letters=0.25,
		ub_vows_by_letters=0.53,
		lb_letters_by_chars=0.67,
		ub_spaces_by_chars=0.2,
		ub_digits_by_chars=0.2
	) :
		self.lb_vows_by_letters = lb_vows_by_letters
		self.ub_vows_by_letters = ub_vows_by_letters
		self.lb_letters_by_chars = lb_letters_by_chars
		self.ub_spaces_by_chars = ub_spaces_by_chars
		self.ub_digits_by_chars = ub_digits_by_chars
		def f(w) :
			dist = Counter(w)
			chars = len(w)
			vows = sum([dist.get(c) for c in VOWELS if c in dist])
			letters = sum([dist.get(c) for c in string.ascii_letters if c in dist])
			spaces = dist.get(" ", 0)
			digits = sum([dist.get(c) for c in string.digits if c in dist])
			result = None
			if \
				chars > 0 and \
				letters > 0 and \
				self.ub_vows_by_letters > vows/letters > self.lb_vows_by_letters and \
				letters/chars > self.lb_letters_by_chars and \
				self.ub_spaces_by_chars > spaces/chars and \
				self.ub_digits_by_chars > digits/chars \
			:
				result = w
			return result
		super(IsNLText, self).__init__(f = f)

class LemmatizeModifier(ItemModifier) :
	def __init__(self,
		lemmatizer = spacy.load('de'),
		chunksize=LEMMATIZE_MAX_SIZE
	) :
		self.lemmatizer = lemmatizer
		self.chunksize=chunksize
		f = lambda tokens : [
			t.lemma_ for i in range(0, len(tokens) ,self.chunksize) \
				for t in self.lemmatizer(
					" ".join(tokens[i : i + self.chunksize])
				) 
		]
		super(LemmatizeModifier, self).__init__(f = f)

class Remove(ItemModifier) :
	def __init__(self,
		stopwords=STANDARD_STOPWORD,
		filter_function=lambda w : RE_NUMBER.match(w) == None and RE_SINGLE_LETTER.match(w) == None,
		filter_symbols=STANDARD_FILTER_SYMBOLS
	):
		self.stopwords = stopwords
		self.filter_function = filter_function
		self.filter_symbols = filter_symbols
		self.createFF()
		super(Remove, self).__init__(
			f = lambda tokens : list(filter(self.ff, tokens))
		)
	def createFF(self) :
		self.ff = lambda w : self.filter_function(w) and w.lower() not in self.stopwords+self.filter_symbols
	def addStopwords(self, aList) :
		self.stopwords = self.stopwords + aList
		return self
	def addFilterSymbols(self, aList) :
		self.filter_symbols = self.filter_symbols + aList
		return self

def tokenize_text(text, re_replace_space_chars=RE_REPLACE_SPACE_CHARS) :
	t = text
	if re_replace_space_chars is not None :
		t = re_replace_space_chars.sub(" ", text)	
	return TreebankWordTokenizer().tokenize(t)

class TokenizeText(ItemModifier) :
	def __init__(self,
		re_replace_space_chars=RE_REPLACE_SPACE_CHARS
	) :
		self.re_replace_space_chars = re_replace_space_chars
		def f(text) :
			t = text
			if self.re_replace_space_chars is not None :
				t = self.re_replace_space_chars.sub(" ", text)
			return TreebankWordTokenizer().tokenize(t)
		super(TokenizeText, self).__init__(f = f)

class LemmaTokenizeText(ItemModifier) :
	def __init__(self,
		lemmatizer=spacy.load('de'),
		maxChunkLength=LEMMATIZE_MAX_CHUNK_SIZE,
		re_replace_space_chars=RE_REPLACE_SPACE_CHARS,
		re_remove_chars=RE_REMOVE_CHARS
	) :
		self.lemmatizer = lemmatizer
		self.maxChunkLength = maxChunkLength
		self.re_replace_space_chars = re_replace_space_chars
		self.re_remove_chars = re_remove_chars
		def f(text) :
			if self.re_replace_space_chars is not None :
				txt = self.re_replace_space_chars.sub(" ", text)	
			if self.re_remove_chars is not None :
				txt = self.re_remove_chars.sub("", txt)	
			txt = RE_WHITESPACE.sub(" ", txt)
			chunks = splitInChunks(txt, self.maxChunkLength)
			return [ t.lemma_.strip() for chunk in chunks for t in self.lemmatizer(chunk) ]
		super(LemmaTokenizeText, self).__init__(f = f)

class SplitText(IteratorModifier) :
	def __init__(self,
		reTextSeparator = "\s*\n\s*\n\s*",
		minTextLength = 0,
		do_trim_text=True
	) :
		self.re_text_separator = re.compile(reTextSeparator)
		self.minTextLength = minTextLength
		self.do_trim_text=do_trim_text
	def __call__(self, other) :
		if other.is_tagged :
			def generator() :
				n = 0
				for text in other :
					p_c = 0
					paragraphs = self.re_text_separator.split(text[0].strip())
					for p in paragraphs :
						if self.do_trim_text :
							p = p.strip()
						if len(p) >= self.minTextLength :
							yield (p, text[1] + [tuple(text[1] + [p_c])])
						n += 1
						p_c += 1
		else :
			def generator() :
				n = 0
				for text in other :
					paragraphs = self.re_text_separator.split(text.strip())
					for p in paragraphs :
						if self.do_trim_text :
							p = p.strip()
						if len(p) >= self.minTextLength :
							yield p
						n += 1
		return Iterator(generator, is_tagged = other.is_tagged)

class LineSourceIterator(SelfIterable) :
	def __init__(self,
		input_file,
		logFreq = 1000,
		outputFreq = 1,
		tag_separator = STANDARD_SEPARATOR
	) :
		self.input_file = input_file
		self.logFreq = logFreq
		self.outputFreq = outputFreq
		self.tag_separator = tag_separator
		super(LineSourceIterator, self).__init__()
		self.handleFirstLine()
	def handleFirstLine(self) :
		file = open(self.input_file, 'r')
		first_line = None
		try :
			file = open(self.input_file, 'r')
			for first_line in file :
				break
		except :
			logger.error("could not read from file %s" % (self.input_file), sys.exc_info()[0])
			raise
		if first_line is None :
			raise Exception("input file %s is empty" % (self.input_file))
		line_l = first_line.split(self.tag_separator)
		if len(line_l) == 1 :
			self.is_tagged = False
			get_line = lambda l : l
		else :
			self.is_tagged = True
			def get_line(l) :
				ll = l.split(self.tag_separator)
				return ( ll[0], ast.literal_eval(ll[1]) )
		self.get_line = get_line
	def __call__(self) :
		file = open(self.input_file, 'r')
		try:
			c_in = 0
			c_out = 0	   
			for line in file :
				if (c_in % self.outputFreq == 0) :
					if c_out % self.logFreq == 0 :
						logger.debug("%s read=%i, ouput=%i\n" %(self.input_file, c_in, c_out))
					yield self.get_line(line)
					c_out += 1
				c_in += 1
			logger.debug("ready: %s read=%i, ouput=%i\n" %(self.input_file, c_in, c_out))
		except Exception as e:
			logger.error("could not read %s" % (self.input_file), exc_info=True)

class LineSourceTokenizer(LineSourceIterator) :
	def __init__(self,
		input_file,
		tokenizer = TokenizeText(),
		**kwargs
	) :
		self.tokenizer = tokenizer
		super(LineSourceTokenizer, self).__init__(input_file, **kwargs)
		self.handleFirstLine()
		if (self.is_tagged) :
			def get_line(l) :
				ll = l.split(self.tag_separator)
				return ( tokenizer(ll[0]), ast.literal_eval(ll[1]) )
		else :
			get_line = lambda l : tokenizer(l)
		self.get_line = get_line

class XmlSourceIterator(SelfIterable) :
	def __init__(self,
		sourceFiles,
		contentTag = "content",
		minTextLength = 10,
		logFreq = 1000,
		is_tagged = False,
		tag_rule = 0 # 0 = "filename", 1 = "number"
	) :
		self.sourceFiles = sourceFiles
		self.contentTag = contentTag
		self.minTextLength = minTextLength
		self.logFreq = logFreq
		self.tag_rule = tag_rule
		self.num_source_files = len(self.sourceFiles)
		logger.info("num_source_files : %i" % (self.num_source_files))
		super(XmlSourceIterator, self).__init__(is_tagged=is_tagged)
	def __call__(self) :
		if self.is_tagged :
			if self.tag_rule == 0 :
				do_tag = lambda content, f, n_f : (content, [f])
			elif self.tag_rule == 1 :
				do_tag = lambda content, f, n_f : (content, [n_f])
		else :
			do_tag = lambda content, f, n_f : content
		n = 0
		n_f = 0
		for f in self.sourceFiles :
			try :
				xml = ET.parse(f).getroot()
				content = " ".join([ c.text for c in xml.findall("./" + self.contentTag) if not c.text is None ]).strip()
				xml.clear()
				if len(content) >= self.minTextLength :
					if n % self.logFreq == 0 :
						logger.info("xml out=%i, read=%i/%i, (%s)" %(n, n_f, self.num_source_files, f))
					yield do_tag(content, f, n_f)
					n += 1	
				n_f += 1
			except Exception as e:
				logger.error("could not parse %s" % (f), exc_info=True)
					
class Merge(IteratorModifier) :
	def __init__(self, append_number_to_tag=False) :
		self.append_number_to_tag = append_number_to_tag
	def genInfoFromInput(self, other):
		iters = []
		weights = []
		if (isinstance(other, tuple) or isinstance(other, list)) and len(other) >= 2 :
			for o in other :
				iter, weight = self.genInfoFromListItem(o)
				iters.append(iter)
				weights.append(weight)		
		else :
			raise Exception("Right side of Merge has to be 'tuple' or 'list' of length >= 2")
		return iters, weights
	def genInfoFromListItem(self, item) :
		weight = 1.0
		iterator = None
		if isinstance(item, tuple) :
			iterator = item[0]
			if len(item) > 1 :
				weight = item[1]
			if not ( isinstance(weight, numbers.Number) and weight > 0 ):
				raise Exception("MergeTwo : Object %s is no positive number" % str(weight))
		else :
			iterator = item
		return iterator, weight
	def __call__(self, other) :
		iters, weights = self.genInfoFromInput(other)
		steps = [1.0 / w for w in weights]
		tags = [i.is_tagged for i in iters]
		if all(tags) != any(tags) :
			raise(
				Exception(
					"For 'Merge' : Either all or none of Iterators on the right side must be tagged !"
				)
			) 
		is_tagged = all(tags)
		if is_tagged and self.append_number_to_tag :
			def mask_output(x, i) :
				x[1].append(i)
				return x
		else :
			mask_output = lambda x, i : x
		def generator() :
			counts = { i:s for i, s in enumerate(steps) }
			for i in iters :
				i.__iter__()
			items = [True] * len(iters)
			while any(items) :
				next_index = min(counts, key=counts.get)
				if items[next_index] :
					try:
						items[next_index] = next(iters[next_index])
						yield mask_output(items[next_index], next_index)
					except StopIteration :
						logger.debug("iter %i finished" % (next_index))
						items[next_index] = None
				counts[next_index] += steps[next_index]
		return Iterator(generator, is_tagged=is_tagged)
			
class Subset(IteratorModifier) :
	def __init__(self, outputFrom=1, outputUntil=-1, distance=1) :
		self.outputFrom=outputFrom
		self.outputUntil=outputUntil
		self.distance=distance
		super(Subset, self).__init__()
	def __call__(self, iterator) :
		def generator() :
			n = 0
			for t in iterator :
				if self.outputUntil > 0 and n+1 > self.outputUntil :
					break
				if \
					n % self.distance == 0 \
					and \
					n+1 >= self.outputFrom \
				:
					yield t
				n += 1
		return Iterator(generator, is_tagged=iterator.is_tagged)

class MinMaxTokens(IteratorModifier) :
	def __init__(self, minTokens=1, maxTokens=-1) :
		self.minTokens = minTokens
		self.maxTokens = maxTokens
	def __call__(self, iterator) :
		result = None
		if self.maxTokens >= 0 :
			if iterator.is_tagged :
				def generator() :
					for t in iterator :
						if self.minTokens <= len(t[0]) <= self.maxTokens :
							yield t
			else :	
				def generator() :
					for t in iterator :
						if self.minTokens <= len(t) <= self.maxTokens :
							yield t
		else :
			if iterator.is_tagged :
				def generator() :
					for t in iterator :
						if len(t[0]) >= self.minTokens :
							yield t
			else :
				def generator() :
					for t in iterator :
						if len(t) >= self.minTokens :
							yield t
		return Iterator(generator, is_tagged=iterator.is_tagged)

class Repeat(IteratorModifier) :
	def __init__(self, total_repeats=1, total_items=None) :
		self.total_repeats = total_repeats
		self.total_items = total_items
	def __call__(self, iterator) :
		if self.total_items is not None :
			def generator() :
				n = 0
				r = 0
				while n < self.total_items :
					for i in iterator :
						if n >= self.total_items :
							break
						n += 1
						yield i
					r += 1
					logger.debug("Repeat - repeats=%i, items=%i" % (r, n))
			result = Iterator(generator, is_tagged=iterator.is_tagged)
		elif self.total_repeats is not None :	
			def generator() :
				n = 0
				r = 0
				while r < self.total_repeats :
					for i in iterator :
						n += 1
						yield i
					r += 1
				logger.debug("Repeat - repeats=%i, items=%i" % (r, n))
			result = Iterator(generator, is_tagged=iterator.is_tagged)
		else :
			raise Exception("Either 'total_items' or 'total_repeats' has to be not None")
		return result

class TokensToFile(IteratorConsumer) :
	def __init__(self, filename, output_tag=True, tag_separator=STANDARD_SEPARATOR) :
		self.filename = filename
		self.output_tag = output_tag
		self.tag_separator = tag_separator	
	def __call__(self, iterator) :
		if iterator.is_tagged :
			if self.output_tag :
				to_str = lambda t : " ".join(t[0]) + self.tag_separator + str(t[1])
			else :
				to_str = lambda t : " ".join(t[0])
		else :
			to_str = lambda t : " ".join(t)
		n = 0
		try :
			file = open(self.filename, "w")
			for t in iterator :
				n += 1
				file.write(to_str(t))
				file.write("\n")
			file.close()
		except Exception as e :
			s = "could not write to file %s" % (s)
			logger.error(s, exc_info=True)
		self.number = n
		return self

class CountTokens(IteratorConsumer) :
	def __init__(self, word_counter=Counter(), tagged_counter=Counter()) :
		self.word_counter = word_counter
		self.tagged_counter = tagged_counter
	def __call__(self, iterator) :
		if iterator.is_tagged :
			def count(tokens) :
				self.word_counter.update(tokens[0])
				self.tagged_counter([(w, ";".join([str(t) for t in tokens[1]])) for w in tokens[0] ])
		else :
			def count(tokens) :
				self.word_counter.update(tokens)
		for tokens in iterator :
			count(tokens)
		return self

class ListIterator(SelfIterable) :
	def __init__(self, input_list, is_tagged=False) :
		self.input_list = input_list
		super(ListIterator, self).__init__(is_tagged=is_tagged)
	def __call__(self) :
		if self.is_tagged :
			for d in enumerate(self.input_list) :
				yield (d[1], [d[0]])
		else :
			for d in self.input_list :
				yield d

class RandomStrings(ListIterator) :
	def __init__(self, number_of_docs=10, length_of_words=5, number_of_words=15, is_tagged=False) :
		def gen_word() :
			allLetters = string.ascii_lowercase + string.ascii_uppercase
			lcLetters = string.ascii_lowercase
			return random.choice(allLetters) + \
				''.join(random.choice(lcLetters) for i in range(length_of_words - 1))
		def gen_doc() :
			d = ' '.join(gen_word() for i in range(number_of_words))
			return d[0].upper() + d[1:] + "."
		my_list = [gen_doc() for i in range(number_of_docs)]
		super(RandomStrings, self).__init__(my_list, is_tagged=is_tagged)

class Untag(IteratorModifier) :
	def __call__(self, iterator) :
		if not iterator.is_tagged :
			raise Exception(
				"Untag: 'iterator' is not tagged - 'Untag' expects a tagged iterator and untags it."
			)
		def generator() :
			for x in iterator :
				yield x[0]
		return Iterator(generator, is_tagged=False)
