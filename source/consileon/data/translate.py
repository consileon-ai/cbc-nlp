"""
consileon.data.translate
========================

Translate single words in texts literally using dictionaries.
Used e.g. for generating bilingual word2vec models.
"""
import re
import random

import consileon.data.tokens as tkns

RE_LEFT_SIDE = re.compile(r"^\s*(?:\[[~ '\w\.]+\])?(?:\([ '\w]+\))?\s*([ \'\-\w]+\.?)(?: +\([ \d\w\.\:]+\))?\s+(?:\/.*\/)$")
RE_RIGHT_SIDE = re.compile(r"\s*(?:\([ \'\w]+\))?([ \-\(\)\w\.\/\'\:]+[\.!\?]?)(?:\.\.\.)?(?: +\([ \d\w\.\:]+\))?(?: +\[[ \d\w\.]+\])?(?: +<(?:[^<]*[fnm;]+[^>]*)>)?(?: \([\.\d]+\))?$")
RE_IN_BRACKETS = re.compile(r"\(\w+\)|\[\w+\]")
RE_REMOVE_LEADING_ENGLISH_WORDS = re.compile(r"^(a|to) ")
RE_REMOVE_LEADING_GERMAN_WORDS = re.compile(r"^(ein|eine) ")

def getFreedictOrgPairs(dict_filename,
	re_remove_leading_left_words=RE_REMOVE_LEADING_ENGLISH_WORDS,
	re_remove_leading_right_words=RE_REMOVE_LEADING_GERMAN_WORDS
) :
	file = open(dict_filename, 'r')
	line = file.readline()
	while line :
		m = RE_LEFT_SIDE.match(line)
		if m is not None :
			orig_l = line.strip()
			l = re_remove_leading_left_words.sub("", m.group(1))
			line = file.readline()
			orig_r = line.strip()
			r = []
			for w in line.split(',') :
				m = RE_RIGHT_SIDE.match(w.strip())
				if m is not None :
					r.append(re_remove_leading_right_words.sub("", RE_IN_BRACKETS.sub("", m.group(1)).strip()))
			yield l, r, orig_l, orig_r
		line = file.readline()
	file.close()

def getLanguageDicts(anIterable) :
	leftToRight = {}
	rightToLeft = {}
	for l, r, l_orig, r_orig in anIterable :
		if l in leftToRight :
			leftToRight[l].update(r)
		else :
			leftToRight[l] = set(r)
		for anR in r :
			if anR in rightToLeft :
				rightToLeft[anR].add(l)
			else :
				rightToLeft[anR] = set([l])
	return {l:list(r) for l,r in leftToRight.items() if len(r) > 0}, {l:list(r) for l,r in rightToLeft.items() if len(r) > 0}

def getFreedictOrgDicts(dict_filename,
	re_remove_leading_left_words=RE_REMOVE_LEADING_ENGLISH_WORDS,
	re_remove_leading_right_words=RE_REMOVE_LEADING_GERMAN_WORDS
) :
	return getLanguageDicts(
		getFreedictOrgPairs(dict_filename,
			re_remove_leading_left_words=re_remove_leading_left_words,
			re_remove_leading_right_words=re_remove_leading_right_words
		)
	)

def reduce_to_noun_if_exists(aDict) :
	for l, r in aDict.items() :
		r = [ w for w in r if w[0].isupper() ]
		if len(r) > 0 :
			aDict[l] = r	

class PartiallyTranslate(tkns.ItemModifier) :
	def __init__(self,
		dictionary,
		translateFreq = 5,
		origTokenModifier = tkns.Append("_ORIG"),
		tokenizeTranslated = tkns.Append("_TRANS") * tkns.TokenizeText()
	) :
		self.dictionary = dictionary
		self.translateFreq = translateFreq
		self.origTokenModifier = origTokenModifier
		self.tokenizeTranslated = tokenizeTranslated
	def __call__(self, tokens) :
		s = random.randint(0, 20)
		i = 0
		l = len(tokens)
		new_tokens = []
		while i < l :
			j = i + self.translateFreq - 1
			while j < l :
				if tokens[j] in self.dictionary :
					break
				j += 1
			new_tokens += self.origTokenModifier(tokens[i:j])
			if j < l :
				t = self.dictionary[tokens[j]]
				new_tokens += self.tokenizeTranslated(t[s%len(t)])
				j += 1
				s += 1
			i = j
		return new_tokens
