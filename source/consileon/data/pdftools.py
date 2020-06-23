"""
consileon.data.pdftools
=======================

Helper functions for transforming pdf documentents into plain text and wrap it in xml
documents according to the rss item format.
"""
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import ElementTree

import hashlib
import re
import logging

from tika import parser as tkParser

logger = logging.getLogger('consileon.data.pdftools')
RE_REPLACE_UNWANTED_CHARS_IN_FN=re.compile(r"[ \(\)]")


STANDARD_PROP_MAP = {
	"title":"title",
	"pubDate":"Last-Modified",
	"author":"Author",
	"description":"description",
	"resourceName":"resourceName"
}

def replace_hyphenation(text) :
	return re.sub('(\w)- *\n+([a-zäüö])', r'\1\2', text).strip()

def get_pdf_from_file(filename) :
	try :
		result = tkParser.from_file(filename)
	except Exception as e :
		s = "could not read from %s" % (filename)
		logger.error("could not read %s" % (self.input_file), exc_info=True)
		raise
	return result

def remap_metadata(pdf, prop_map=STANDARD_PROP_MAP, input_map={}) :
	metadata = pdf["metadata"]
	for k in prop_map.keys() :
		if prop_map[k] in metadata.keys() :
			input_map[k] = metadata[prop_map[k]]
	return input_map

def create_xml(element_map, root_tag="item") :
	xml = Element(root_tag)
	for e in element_map.keys() :
		child = Element(e)
		child.text = str(element_map[e]).strip()
		xml.append(child)
	return xml

def xml_to_string(xml) :
	return ET.tostring(xml, encoding='utf-8', method='xml').decode(encoding='utf-8')

def xml_to_file(xml, filename, add_link_tag=True, link_tag_name="link") :
	try :
		if add_link_tag and link_tag_name is not None :
			link_elem = Element(link_tag_name)
			link_elem.text = filename.strip()
			xml.append(link_elem)
		file = open(filename, 'w')
		file.write(xml_to_string(xml))
	except Exception as e :
		logger.error("could write to %s" % (filename), exc_info=True)
		raise

def pdf_to_simple_xml(
	pdf,
	prop_map=STANDARD_PROP_MAP,
	content_tag_name="content",
	root_tag="item",
	cleanse_content_function=replace_hyphenation
) :
	elems = {}
	if prop_map :
		elems = remap_metadata(pdf, prop_map)
	if content_tag_name :
		content = pdf["content"]
		if cleanse_content_function :
			content = cleanse_content_function(content)
		elems[content_tag_name] = content
	return create_xml(elems, root_tag=root_tag)

def get_xml_filename(pdf_filename, re_cleanse_fn=RE_REPLACE_UNWANTED_CHARS_IN_FN) :
	if re.match(".*\.(pdf|PDF)$", pdf_filename) :
		result = re.sub("(pdf|PDF)$", "xml", pdf_filename)
		if re_cleanse_fn is not None :
			result = RE_REPLACE_UNWANTED_CHARS_IN_FN.sub("_", result)
	else :
		s = "'%s' is no pdf" %(pdf_filename)
		logging.error(s)
		raise Exception(s)
	return result
