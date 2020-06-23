"""
consileon.data.grabbers
=======================

Read and store content from rss feeds
"""
from bs4 import BeautifulSoup as bs

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import ElementTree

from urllib.request import Request, urlopen

import hashlib, os
from os import listdir
from os.path import isfile, join

import time

from tika import parser as tkParser
import re

import random
import logging

logger = logging.getLogger('consileon.data.grabbers')

class RssGrabber :
	"""
	Class responsible for handling a list of rss feeds which are semantically related and which are handled
	the same was
	"""	
	def __init__(self, urls=[], folder='.', extractor=None, timeWaitSeconds=3600, doSave=False, timeWaitBetweenItems=0.01):
		self.urls = urls
		self.folder = folder
		self.extractors = {}
		self.extractors['default'] = extractor
		self.knownItems = {}
		self.timeWaitSeconds = timeWaitSeconds
		self.doSave = doSave
		self.userAgent = 'Mozilla/5.0'
		self.timeWaitBetweenItems=timeWaitBetweenItems
		try:  
			os.mkdir(self.folder)
		except OSError:  
			print ("Creation of the directory %s failed" % self.folder)
		else:  
			print ("Successfully created the directory %s " % self.folder)
		if self.doSave :
			self.knownItems = { f:"dummy" for f in listdir(self.folder) if isfile(join(self.folder, f)) }

	def getItemFromFile(self, aFileName) :
		"""
		Read a single rss item from a file

		Args:
			:aFileName str: full name (including path) of file containing the item
		"""
		result = ET.parse(self.folder + "/" + aFileName).getroot()
		return result

	def saveItemToFile(self, anItem, aFileName, itemUrl="-") :
		try :
			file = open(self.folder + "/" + aFileName, 'w')
			file.write(ET.tostring(anItem, encoding='utf-8', method='xml').decode(encoding='utf-8'))
			file.close()
		except :
			print("could not write item \n%s\nto '%s' in folder '%s'" % (itemUrl, aFileName, self.folder))
		
	def grabOnce(self) :
		for i in self.getAllItems() :
			l = RssGrabber.getLinkFromItem(i)
			fileName = RssGrabber.getMd5Hash(l) + ".xml"
			if not fileName in self.knownItems :
				for aLanguage in self.extractors :
					RssGrabber.addContentToItem(
						i,
						self.extractors[aLanguage],
						aLanguage,
						timeWait=self.timeWaitBetweenItems
					)
				self.knownItems[fileName] = i
				if self.doSave :
					self.saveItemToFile(i, fileName, itemUrl=l)

	def grab(self) :
		while(True) :
			self.grabOnce()
			time.sleep(self.timeWaitSeconds)
	
	def getAllItems(self) :
		rssDocs = [ RssGrabber.appendChannelInfoToItems(RssGrabber.parseXmlFromUrl(url)) for url in self.urls ]
		result = [ i for doc in rssDocs if doc != None for i in doc.findall('./channel/item') ]
		return result
	
	def getAllItemLinks(self) : 
		return [ RssGrabber.getLinkFromItem(i) for i in self.getAllItems() ]
	
	def addContentToItem(anItem, anExtractor, language='default', timeWait=0) :
		linkUrl = RssGrabber.getLinkFromItem(anItem)
		if linkUrl != None :
			time.sleep(timeWait)
			html = RssGrabber.readDocFromUrl(linkUrl)
			if html != None :
				oldContent = anItem.find('content[@language="' + language + '"]')
				if oldContent is not None :
					anItem.remove(oldContent)	
				aText = anExtractor(html)
				content = Element("content")
				if language != None :
					content.set('language', language)
				content.text = aText
				anItem.append(content)
		return
	
	def getLinkFromItem(anItem) :
		link = anItem.find('link')
		linkUrl = None
		if link != None:
			linkUrl = link.text
		return linkUrl
		
	def getMd5Hash(s) :
		m = hashlib.md5()
		result = None
		try :
			m.update(s.encode('utf-8', errors='backslashreplace'))
			result = m.hexdigest()
		except :
			print("could not encode " + s)	   
		return result
	
	def appendChannelInfoToItems(aDoc) :
		if aDoc != None :
			theChannel = aDoc.findall('./channel')[0]
			channelElem = Element("channel")
			for tagName in ['title', 'description', 'link', 'language'] :
				tag = theChannel.find(tagName)
				if tag != None :
					channelElem.append(tag)
			for i in aDoc.findall('./channel/item') :
				i.append(channelElem)
		return aDoc
	
	def readDocFromUrl(anUrl) :
		result = None
		try :
			req = Request(anUrl, headers={'User-Agent': 'Mozilla/5.0'})
			result = urlopen(req).read()
		except :
			print("could not read from %s" % anUrl)
		return result
	
	def parseXmlFromUrl(anUrl) :
		result = None
		if anUrl != None :
			try :
				result = ET.fromstring(RssGrabber.readDocFromUrl(anUrl))
			except :
				print ("could not parse xml from url '%s'" % anUrl)
		return result

	def reloadContent(self, aFileName) :
		result = None
		theItem = self.getItemFromFile(aFileName)
		if theItem is not None :
			for aLanguage in self.extractors :
				RssGrabber.addContentToItem(theItem, self.extractors[aLanguage], aLanguage)
		return theItem

	def regenerateContent(self, aFileList) :
		for f in aFileList :
			item = self.reloadContent(f)
			self.saveItemToFile(item, f)
			

def getTextFromPdfBuffer(someBytes) :
	result = None
	if (someBytes is not None and someBytes.startswith(b'%PDF')) :
		text = tkParser.from_buffer(someBytes)['content']
		result = re.sub('(\w)- *\n([a-zäüö])', r'\1\2', text).strip()
	return result

def createRssFileList(dataBaseDir, channels, do_random_shuffle=True) :
	"""
	Create a list of xml files which are stored within the structure of the "rss grabber".

	Args:
		:dataBaseDir (str): the directory which has the "channel directories" as its subfolders.

		:channels (list of str): the list of subfolders of the "dataBaseDir" which contain the xml files
			which are considered as content items

	"""
	files = [
		join(join(dataBaseDir, d), f) for d in channels for f in listdir(join(dataBaseDir, d)) if f.endswith(".xml")
	]
	if do_random_shuffle :
		random.shuffle(files)
	return files

def load_xml_doc_from_file(filename) :
	"""
	Read xml object from a file

	Args:
		:filename (str): the full filename of a file containing an xml file

	Returns:
		The xml object of type xml.etree.ElementTree.Element

	"""
	return ET.parse(filename).getroot()

def get_texts_from_item_file(filename) :
	"""
	Read the content of the tags "title", "description", "content" from an xml file (typically in the "item" format)

	Args:
		:filename (str): the full filename of a file containing an xml file

	Returns:
		:(title, description, content) (str, str, str): the respective tags of the xml file

	"""
	xml = load_xml_doc_from_file(filename)
	try :
		content = " ".join([ c.text for c in xml.findall("./" + "content") if not c.text is None ]).strip()
	except :
		content = ""
	try :
		title = " ".join([ c.text for c in xml.findall("./" + "title") if not c.text is None ]).strip()
	except :
		title = ""
	try :
		description = " ".join([ c.text for c in xml.findall("./" + "description") if not c.text is None ]).strip()
	except :
		description = ""
	return (title, description, content)
