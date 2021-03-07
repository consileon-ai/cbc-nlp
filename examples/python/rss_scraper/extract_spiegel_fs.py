import consileon.nlp.content as content
from bs4 import BeautifulSoup as bs
from consileon.nlp.rss_scraping import RssScraper

h = content.FileSystemContentHandler("../../../temp")

spiegelRssUrls = [
        "http://www.spiegel.de/schlagzeilen/tops/index.rss",
        "http://www.spiegel.de/politik/index.rss",
        "http://www.spiegel.de/wirtschaft/index.rss",
        "http://www.spiegel.de/panorama/index.rss",
        "http://www.spiegel.de/sport/index.rss",
        "http://www.spiegel.de/kultur/index.rss",
        "http://www.spiegel.de/netzwelt/index.rss",
        "http://www.spiegel.de/wissenschaft/index.rss",
        "http://www.spiegel.de/gesundheit/index.rss",
        "http://www.spiegel.de/karriere/index.rss",
        "http://www.spiegel.de/reise/index.rss",
        "http://www.spiegel.de/auto/index.rss"
]

def spiegel_extractor(anHtml):
    soup = bs(anHtml, "lxml")
    for c in soup.find_all('div'):
        if c.has_attr('class'):
            if "article-copyright" in c['class']:
                c.clear()
    text = "\n\n".join([p.get_text() for p in soup.find_all(['p', 'h', 'h1', 'h2', 'h3', 'title'])])

    return (text.strip())


s = RssScraper(urls=spiegelRssUrls, prefix="spiegel", extractor=spiegel_extractor, time_wait_seconds=1 * 3600, do_save=True, content_handler=h)

s.pull_once()