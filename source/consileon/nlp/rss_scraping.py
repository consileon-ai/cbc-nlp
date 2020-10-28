"""
consileon.nlp.rss_scraping
=======================

Read and store content from rss feeds
"""

import hashlib
import logging
import os
import random
import re
import time
import xml.etree.ElementTree as Et
from os import listdir
from os.path import isfile, join
from urllib.request import Request, urlopen
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import ParseError

from tika import parser as tk_parser

logger = logging.getLogger('consileon.nlp.rss_scraping')


class RssScraper:
    """
    Class responsible for handling a list of rss feeds which are semantically related and which are handled
    the same was
    """

    def __init__(self,
                 urls=(),
                 folder='.',
                 extractor=None,
                 time_wait_seconds=3600,
                 do_save=False,
                 time_wait_between_items=0.01):
        self.urls = list(urls)
        self.folder = folder
        self.extractors = {'default': extractor}
        self.knownItems = {}
        self.timeWaitSeconds = time_wait_seconds
        self.doSave = do_save
        self.userAgent = 'Mozilla/5.0'
        self.timeWaitBetweenItems = time_wait_between_items
        try:
            os.mkdir(self.folder)
        except OSError:
            print("Creation of the directory %s failed" % self.folder)
        else:
            print("Successfully created the directory %s " % self.folder)
        if self.doSave:
            self.knownItems = {f: "dummy" for f in listdir(self.folder) if isfile(join(self.folder, f))}

    def get_item_from_file(self, a_file_name):
        """
        Read a single rss item from a file

        Args:
            :a_file_name str: full name (including path) of file containing the item
        """
        result = Et.parse(self.folder + "/" + a_file_name).getroot()
        return result

    def save_item_to_file(self, an_item, a_file_name, item_url="-"):
        try:
            file = open(self.folder + "/" + a_file_name, 'w')
            file.write(Et.tostring(an_item, encoding='utf-8', method='xml').decode(encoding='utf-8'))
            file.close()
        except IOError:
            logger.exception("could not write item \n%s\nto '%s' in folder '%s'" % (item_url, a_file_name, self.folder))
            print("could not write item \n%s\nto '%s' in folder '%s'" % (item_url, a_file_name, self.folder))

    def pull_once(self):
        for i in self.get_all_items():
            l_ = RssScraper.get_link_from_item(i)
            file_name = RssScraper.get_md5_hash(l_) + ".xml"
            if file_name not in self.knownItems:
                for aLanguage in self.extractors:
                    RssScraper.add_content_to_item(
                        i,
                        self.extractors[aLanguage],
                        aLanguage,
                        time_wait=self.timeWaitBetweenItems
                    )
                self.knownItems[file_name] = i
                if self.doSave:
                    self.save_item_to_file(i, file_name, item_url=l_)

    def poll(self):
        while True:
            self.pull_once()
            time.sleep(self.timeWaitSeconds)

    def get_all_items(self):
        rss_docs = [RssScraper.append_channel_info_to_items(RssScraper.parse_xml_from_url(url)) for url in self.urls]
        result = [i for doc in rss_docs if doc is not None for i in doc.findall('./channel/item')]
        return result

    def get_all_item_links(self):
        return [RssScraper.get_link_from_item(i) for i in self.get_all_items()]

    @staticmethod
    def add_content_to_item(an_item, an_extractor, language='default', time_wait=0.0):
        link_url = RssScraper.get_link_from_item(an_item)
        if link_url is not None:
            time.sleep(time_wait)
            html = RssScraper.read_doc_from_url(link_url)
            if html is not None:
                old_content = an_item.find('content[@language="' + language + '"]')
                if old_content is not None:
                    an_item.remove(old_content)
                a_text = an_extractor(html)
                content = Element("content")
                if language is not None:
                    content.set('language', language)
                content.text = a_text
                an_item.append(content)
        return

    @staticmethod
    def get_link_from_item(an_item):
        link = an_item.find('link')
        link_url = None
        if link is not None:
            link_url = link.text
        return link_url

    @staticmethod
    def get_md5_hash(s):
        m = hashlib.md5()
        result = None
        try:
            m.update(s.encode('utf-8', errors='backslashreplace'))
            result = m.hexdigest()
        except UnicodeError:
            logger.exception("could not encode " + s)
            print("could not encode " + s)
        return result

    @staticmethod
    def append_channel_info_to_items(a_doc):
        if a_doc is not None:
            the_channel = a_doc.findall('./channel')[0]
            channel_elem = Element("channel")
            for tagName in ['title', 'description', 'link', 'language']:
                tag = the_channel.find(tagName)
                if tag is not None:
                    channel_elem.append(tag)
            for i in a_doc.findall('./channel/item'):
                i.append(channel_elem)
        return a_doc

    @staticmethod
    def read_doc_from_url(an_url):
        result = None
        try:
            req = Request(an_url, headers={'User-Agent': 'Mozilla/5.0'})
            result = urlopen(req).read()
        except IOError:
            logger.exception("could not read from %s" % an_url)
            print("could not read from %s" % an_url)
        return result

    @staticmethod
    def parse_xml_from_url(an_url):
        result = None
        if an_url is not None:
            result = Et.fromstring(RssScraper.read_doc_from_url(an_url))
        return result

    def reload_content(self, a_file_name):
        the_item = self.get_item_from_file(a_file_name)
        if the_item is not None:
            for aLanguage in self.extractors:
                RssScraper.add_content_to_item(the_item, self.extractors[aLanguage], aLanguage)
        return the_item

    def regenerate_content(self, a_file_list):
        for f in a_file_list:
            item = self.reload_content(f)
            self.save_item_to_file(item, f)


def get_text_from_pdf_buffer(some_bytes):
    result = None
    if some_bytes is not None and some_bytes.startswith(b'%PDF'):
        text = tk_parser.from_buffer(some_bytes)['content']
        result = re.sub(r'(\w)- *\n([a-zäüö])', r'\1\2', text).strip()
    return result


def create_rss_file_list(database_dir, channels, do_random_shuffle=True):
    """
    Create a list of xml files which are stored within the structure of the "rss grabber".

    Args:
        :database_dir (str): the directory which has the "channel directories" as its subfolders.

        :channels (list of str): the list of subfolders of the "database_dir" which contain the xml files
            which are considered as content items

    """
    files = [
        join(join(database_dir, d), f) for d in channels for f in listdir(join(database_dir, d)) if f.endswith(".xml")
    ]
    if do_random_shuffle:
        random.shuffle(files)
    return files


def load_xml_doc_from_file(filename):
    """
    Read xml object from a file

    Args:
        :filename (str): the full filename of a file containing an xml file

    Returns:
        The xml object of type xml.etree.ElementTree.Element

    """
    return Et.parse(filename).getroot()


def get_texts_from_item_file(filename):
    """
    Read the content of the tags "title", "description", "content" from an xml file (typically in the "item" format)

    Args:
        :filename (str): the full filename of a file containing an xml file

    Returns:
        :(title, description, content) (str, str, str): the respective tags of the xml file

    """
    xml = load_xml_doc_from_file(filename)
    try:
        content = " ".join([c.text for c in xml.findall("./" + "content") if c.text is not None]).strip()
    except (ParseError, TypeError):
        content = ""
    try:
        title = " ".join([c.text for c in xml.findall("./" + "title") if c.text is not None]).strip()
    except (ParseError, TypeError):
        title = ""
    try:
        description = " ".join([c.text for c in xml.findall("./" + "description") if c.text is not None]).strip()
    except (ParseError, TypeError):
        description = ""
    return title, description, content
