"""
consileon.data.wikitools
========================

Construct a corpus from a Wikipedia (or other MediaWiki-based) database dump.
Create text streams compatible with "consileon.data.tokens" which can be
feed (e.g.) into generator for word2vec models.

The parts extracting the (very big) wiki source files are taken from the
gensim code (but had to be modified).
"""


import consileon.data.tokens as tkns
import bz2
import logging
import re
from xml.etree.cElementTree import iterparse # LXML isn't faster, so let's go with the built-in solution
import multiprocessing

from gensim import utils

# cannot import whole gensim.corpora, because that imports wikicorpus...
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus

logger = logging.getLogger('de.consileon.data.wikitools')

# ignore articles shorter than ARTICLE_MIN_WORDS characters (after full preprocessing)
ARTICLE_MIN_WORDS = 50


RE_P0 = re.compile('<!--.*?-->', re.DOTALL | re.UNICODE) # comments
RE_P1 = re.compile('<ref([> ].*?)(</ref>|/>)', re.DOTALL | re.UNICODE) # footnotes
RE_P2 = re.compile("(\n\[\[[a-z][a-z][\w-]*:[^:\]]+\]\])+$", re.UNICODE) # links to languages
RE_P3 = re.compile("{{([^}{]*)}}", re.DOTALL | re.UNICODE) # template
RE_P4 = re.compile("{{([^}]*)}}", re.DOTALL | re.UNICODE) # template
RE_P5 = re.compile('\[(\w+):\/\/(.*?)(( (.*?))|())\]', re.UNICODE) # remove URL, keep description
RE_P6 = re.compile("\[([^][]*)\|([^][]*)\]", re.DOTALL | re.UNICODE) # simplify links, keep description
RE_P7 = re.compile('\n\[\[[iI]mage(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE) # keep description of images
RE_P8 = re.compile('\n\[\[[fF]ile(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE) # keep description of files
RE_P9 = re.compile('<nowiki([> ].*?)(</nowiki>|/>)', re.DOTALL | re.UNICODE) # outside links
RE_P10 = re.compile('<math([> ].*?)(</math>|/>)', re.DOTALL | re.UNICODE) # math content
RE_P11 = re.compile('<(.*?)>', re.DOTALL | re.UNICODE) # all other tags
RE_P12 = re.compile('\n(({\|)|(\|-)|(\|}))(.*?)(?=\n)', re.UNICODE) # table formatting
RE_P13 = re.compile('\n(\||\!)(.*?\|)*([^|]*?)', re.UNICODE) # table cell formatting
RE_P14 = re.compile('\[\[Category:[^][]*\]\]', re.UNICODE) # categories
# Remove File and Image template
RE_P15 = re.compile('\[\[([fF]ile:|[iI]mage)[^]]*(\]\])', re.UNICODE)


def filter_wiki(raw):
	"""
	Filter out wiki mark-up from `raw`, leaving only text. `raw` is either unicode
	or utf-8 encoded string.
	"""
	# parsing of the wiki markup is not perfect, but sufficient for our purposes
	# contributions to improving this code are welcome :)
	text = utils.to_unicode(raw, 'utf8', errors='ignore')
	text = utils.decode_htmlentities(text) # '&amp;nbsp;' --> '\xa0'
	return remove_markup(text)


def remove_markup(text):
	text = re.sub(RE_P2, "", text) # remove the last list (=languages)
	# the wiki markup is recursive (markup inside markup etc)
	# instead of writing a recursive grammar, here we deal with that by removing
	# markup in a loop, starting with inner-most expressions and working outwards,
	# for as long as something changes.
	text = remove_template(text)
	text = remove_file(text)
	iters = 0
	while True:
		old, iters = text, iters + 1
		text = re.sub(RE_P0, "", text) # remove comments
		text = re.sub(RE_P1, '', text) # remove footnotes
		text = re.sub(RE_P9, "", text) # remove outside links
		text = re.sub(RE_P10, "", text) # remove math content
		text = re.sub(RE_P11, "", text) # remove all remaining tags
		text = re.sub(RE_P14, '', text) # remove categories
		text = re.sub(RE_P5, '\\3', text) # remove urls, keep description
		text = re.sub(RE_P6, '\\2', text) # simplify links, keep description only
		# remove table markup
		text = text.replace('||', '\n|') # each table cell on a separate line
		text = re.sub(RE_P12, '\n', text) # remove formatting lines
		text = re.sub(RE_P13, '\n\\3', text) # leave only cell content
		# remove empty mark-up
		text = text.replace('[]', '')
		if old == text or iters > 2: # stop if nothing changed between two iterations or after a fixed number of iterations
			break

	# the following is needed to make the tokenizer see '[[socialist]]s' as a single word 'socialists'
	# TODO is this really desirable?
	text = text.replace('[', '').replace(']', '') # promote all remaining markup to plain text
	return text


def remove_template(s):
	"""Remove template wikimedia markup.
	Return a copy of `s` with all the wikimedia markup template removed. See
	http://meta.wikimedia.org/wiki/Help:Template for wikimedia templates
	details.
	Note: Since template can be nested, it is difficult remove them using
	regular expresssions.
	"""

	# Find the start and end position of each template by finding the opening
	# '{{' and closing '}}'
	n_open, n_close = 0, 0
	starts, ends = [], []
	in_template = False
	prev_c = None
	for i, c in enumerate(iter(s)):
		if not in_template:
			if c == '{' and c == prev_c:
				starts.append(i - 1)
				in_template = True
				n_open = 1
		if in_template:
			if c == '{':
				n_open += 1
			elif c == '}':
				n_close += 1
			if n_open == n_close:
				ends.append(i)
				in_template = False
				n_open, n_close = 0, 0
		prev_c = c

	# Remove all the templates
	s = ''.join([s[end + 1:start] for start, end in
				 zip(starts + [None], [-1] + ends)])

	return s


def remove_file(s):
	"""Remove the 'File:' and 'Image:' markup, keeping the file caption.
	Return a copy of `s` with all the 'File:' and 'Image:' markup replaced by
	their corresponding captions. See http://www.mediawiki.org/wiki/Help:Images
	for the markup details.
	"""
	# The regex RE_P15 match a File: or Image: markup
	for match in re.finditer(RE_P15, s):
		m = match.group(0)
		caption = m[:-2].split('|')[-1]
		s = s.replace(m, caption, 1)
	return s


def tokenize(content):
	"""
	Tokenize a piece of text from wikipedia. The input string `content` is assumed
	to be mark-up free (see `filter_wiki()`).
	Return list of tokens as utf8 bytestrings. Ignore words shorted than 2 or longer
	that 15 characters (not bytes!).
	"""
	# TODO maybe ignore tokens with non-latin characters? (no chinese, arabic, russian etc.)
	return [token.encode('utf8') for token in utils.tokenize(content, lower=True, errors='ignore')
			if 2 <= len(token) <= 15 and not token.startswith('_')]


def get_namespace(tag):
	"""Returns the namespace of tag."""
	m = re.match("^{(.*?)}", tag)
	namespace = m.group(1) if m else ""
	if not namespace.startswith("http://www.mediawiki.org/xml/export-"):
		raise ValueError("%s not recognized as MediaWiki dump namespace"
						 % namespace)
	return namespace
_get_namespace = get_namespace

def extract_pages(f, filter_namespaces=False):
	"""
	Extract pages from MediaWiki database dump.
	Returns
	-------
	pages : iterable over (str, str)
		Generates (title, content) pairs.
	"""
	elems = (elem for _, elem in iterparse(f, events=("end",)))

	# We can't rely on the namespace for database dumps, since it's changed
	# it every time a small modification to the format is made. So, determine
	# those from the first element we find, which will be part of the metadata,
	# and construct element paths.
	elem = next(elems)
	namespace = get_namespace(elem.tag)
	ns_mapping = {"ns": namespace}
	page_tag = "{%(ns)s}page" % ns_mapping
	text_path = "./{%(ns)s}revision/{%(ns)s}text" % ns_mapping
	title_path = "./{%(ns)s}title" % ns_mapping
	ns_path = "./{%(ns)s}ns" % ns_mapping
	pageid_path = "./{%(ns)s}id" % ns_mapping

	for elem in elems:
		if elem.tag == page_tag:
			title = elem.find(title_path).text
			text = elem.find(text_path).text

			ns = elem.find(ns_path).text
			if filter_namespaces and ns not in filter_namespaces:
				text = None

			pageid = elem.find(pageid_path).text
			yield title, text or "", pageid	 # empty page will yield None

			# Prune the element tree, as per
			# http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
			# except that we don't need to prune backlinks from the parent
			# because we don't use LXML.
			# We do this only for <page>s, since we need to inspect the
			# ./revision/text element. The pages comprise the bulk of the
			# file, so in practice we prune away enough.
			elem.clear()
_extract_pages = extract_pages  # for backward compatibility

def process_article(args):
	"""
	Parse a wikipedia article, returning its content as a list of tokens
	(utf8-encoded strings).
	"""
	text, title, pageid = args
	text = filter_wiki(text)
	return text, title, pageid


class WikiCorpus(TextCorpus):
	"""
	Treat a wikipedia articles dump (\*articles.xml.bz2) as a (read-only) corpus.
	The documents are extracted on-the-fly, so that the whole (massive) dump
	can stay compressed on disk.
	>>> wiki = WikiCorpus('enwiki-20100622-pages-articles.xml.bz2') # create word->word_id mapping, takes almost 8h
	>>> wiki.saveAsText('wiki_en_vocab200k') # another 8h, creates a file in MatrixMarket format plus file with id->word
	"""
	def __init__(self, fname, processes=None, dictionary=None, filter_namespaces=('0',)):
		"""
		Initialize the corpus. Unless a dictionary is provided, this scans the
		corpus once, to determine its vocabulary.
		If `pattern` package is installed, use fancier shallow parsing to get
		token lemmas. Otherwise, use simple regexp tokenization. You can override
		this automatic logic by forcing the `lemmatize` parameter explicitly.
		"""
		self.fname = fname
		self.filter_namespaces = filter_namespaces
		self.metadata = False
		if processes is None:
			processes = max(1, multiprocessing.cpu_count() - 1)
		self.processes = processes
		if dictionary is None:
			self.dictionary = Dictionary([[]])
		else:
			self.dictionary = dictionary

	def get_texts(self):
		"""
		Iterate over the dump, returning text version of each article as a list
		of tokens.
		Only articles of sufficient length are returned (short articles & redirects
		etc are ignored).
		Note that this iterates over the **texts**; if you want vectors, just use
		the standard corpus interface instead of this function::
		>>> for vec in wiki_corpus:
		>>>	 print(vec)
		"""
		articles, articles_all = 0, 0
		texts = ((text, title, pageid) for title, text, pageid in extract_pages(bz2.BZ2File(self.fname), self.filter_namespaces))
		pool = multiprocessing.Pool(self.processes)
		# process the corpus in smaller chunks of docs, because multiprocessing.Pool
		# is dumb and would load the entire input into RAM at once...
		ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
		for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
			for text, title, pageid in pool.imap(process_article, group): # chunksize=10):
				articles_all += 1
				# article redirects and short stubs are pruned here
				articles += 1
				yield (text, title, pageid)
		pool.terminate()

		# logger.info("finished iterating over Wikipedia corpus of %i documents (all : %i)" % (articles, articles_all))
		self.length = articles # cache corpus length
# endclass WikiCorpus

RE_START_ARTICLE = re.compile("^===== START")
RE_END_ARTICLE = re.compile("^===== END")
RE_ARTICLE_BOUND = re.compile("^=====")
RE_START_PARAGRAPH = re.compile("^=====")
RE_IGNORE_LINE = re.compile("^(\*| |;|\:|==|#|Kategorie\:)|^\s*[\d\w]{1,5}$")
REPLACE_SPACE_CHARS = re.compile(">|\-|\/|,")
RE_REPLACE_DOUBLE_CR = re.compile("[ \t]*\n[ \t]*\n\s*")
MIN_TOKENS = 15
MIN_LENGTH_LINE = 20

def check_tokens(tokens) :
	return len(tokens) >= MIN_TOKENS

def ignore_line(line):
	return len(line) < MIN_LENGTH_LINE or RE_IGNORE_LINE.match(line)

class WikiFromTextIterator(tkns.SelfIterable) :
	def __init__(self,
		inputFile,
		logFreq = 1000,
		outputFreq=1,
		minTextLength=20,
		reStartArticle=RE_START_ARTICLE,
		reEndArticle=RE_END_ARTICLE,
		reIgnoreLine=RE_IGNORE_LINE
	) :
		self.inputFile = inputFile
		self.logFreq = logFreq
		self.outputFreq = outputFreq
		self.minTextLength = minTextLength
		self.reStartArticle = reStartArticle
		self.reEndArticle = reEndArticle
		self.reIgnoreLine = reIgnoreLine
		super(WikiFromTextIterator, self).__init__()
	def __call__(self) :
		try:
			f = open(self.inputFile)
			n_lines = 0
			n_parsed_articles = 0
			n_output_articles = 0
			article_text = ""
			article_start_line = ""
			lines_in_article = 0
			for line in f :
				n_lines += 1
				is_start_article = self.reStartArticle.match(line) is not None
				is_end_article = self.reEndArticle.match(line) is not None
				is_ignore_line = self.reIgnoreLine.match(line) is not None
				if not is_ignore_line and not is_start_article :
					article_text = article_text + line
				if is_end_article :
					article_text = article_text.strip()
					if lines_in_article > 0 :
						n_parsed_articles += 1
						article_text = RE_REPLACE_DOUBLE_CR.sub("\n\n", article_text)
						if len(article_text) >= self.minTextLength :
							if n_output_articles % self.logFreq == 0 :
								logger.debug(
									"lines read= %i, parsed articles= %i, output articles= %i, article start line:\n%s\n" %
									(n_lines, n_parsed_articles, n_output_articles + 1, article_start_line)
								)
							if n_output_articles % self.outputFreq == 0 :
								yield article_text
							n_output_articles += 1
					lines_in_article = 0
				else :
					lines_in_article += 1
				if is_start_article :
					lines_in_article = 0
					article_start_line = line.strip()
					if is_ignore_line :
						article_text = ""
					else :
						article_text = line
			if lines_in_article > 0 :
				article_text = article_text.strip()
				n_parsed_articles += 1
				if len(article_text) >= self.minTextLength :
					yield article_text
			logger.debug("ready input file %s: lines read= %i, parsed article= %i, output articles= %i" % (self.inputFile, n_lines, n_parsed_articles, n_output_articles))
		finally:
			f.close()

def wikiToSimpleText(inputFilename, outputFilename) :
	wikiCorpus = WikiCorpus(inputFilename)
	file = open(outputFilename, 'w')
	for (text, title, pageid) in wikiCorpus.get_texts():
		file.write("===== START %s : \n" % (pageid, title))
		file.write(text)
		file.write("\n===== END %s\n" % (pageid))
	file.close
