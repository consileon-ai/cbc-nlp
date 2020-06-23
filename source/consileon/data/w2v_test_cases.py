"""
consileon.data.w2v_test_cases
=============================

Testcases for the framework "w2v_test_framework" which have been
predefined for ease of use.
"""
import consileon.data.tokens as tkns

import nltk
import spacy

from consileon.data.w2v_const import SYNONYM, SIMILAR, NOT_SIMILAR, WORD_CALC, WORD_CALC_NEG, POS, NEG, IS, MOD


STANDARD_DE = {
    MOD : tkns.Lower() * tkns.Remove() * tkns.LemmaTokenizeText(),
    SYNONYM : [
        ('Ehemann', 'Gatte'),
        ('Auto', 'Wagen'),
        ('Monarch', 'König'),
        ('Verteidigungsministerium', 'Ministerium der Verteidigung'),
        ('Nutztierhaltung', 'Viehwirtschaft'),
	('Straßenverkehr', 'Verkehr auf der Straße'),
	('gehen', 'laufen'),
	('sehen', 'wahrnehmen'),
	('antizipieren', 'vorhersehen'),
	('schnell', 'flink'),
	('Nutzpflanze', 'Feldfrucht'),
	('Grundgesetz', 'deutsche Verfassung'),
	('Athlet', 'Leistungssportler'),
	('Fußballer', 'Fußballspieler'),
	('SPD', 'Sozialdemokraten'),
	('Queen', 'Königin'),
	('stark', 'kraftvoll')
    ] ,
    SIMILAR : [
        ('Vereinigte Staaten', 'USA'),
        ('Kanzlerin', 'Merkel'),
        ('Kanzlerin', 'Angela Merkel'),
        ('US Präsident', 'Donald Trump'),
        ('Kaiser', 'König'),
        ('Landwirtschaft', 'Ackerbau'),
        ('Landwirtschaft', 'Viehwirtschaft'),
        ('Altbundeskanzler', 'Sebastian Kurz'),
        ('Bundeskanzler von Österreich', 'Sebastian Kurz'),
	('deutsche Hauptstadt', 'Berlin'),
	('Napoleon', 'Kaiser von Frankreich'),
	('Wilhelm', 'deutscher Kaiser'),
	('Wilhelm II', 'deutscher Kaiser'),
	('Wilhelm II', 'letzer deutscher Kaiser'),
	('Trockenheit', 'Klimawandel'),
	('Trockenheit', 'Folge des Klimawandel'),
        ('Haus', 'Fahrstuhl'),
	('Hund', 'Katze'),
	('Verteidigungsministerin', 'AKK'),
	('Verteidigungsministerin', 'Annegret Kramp-Karrenbauer'),
	('Verteidigungsministerin', 'Ursula von der Leyen'),
	('Idi Amin', 'Gewaltherrscher'),
	('gehen', 'fliegen'),
	('gehen', 'fahren'),
	('gehen', 'schwimmen'),
	('gehen', 'lesen'),
	('blau', 'grün'),
	('grün', 'olivgrün'),
	('blau', 'türkis'),
	('blau', 'hellblau'),
	('braun', 'nationalsozialistisch'),
	('Unternehmensberatung', 'Roland Berger'),
	('Strategieberatung', 'Roland Berger'),
	('Strategie', 'Handlungsempfehlung'),
	('christlich', 'katholisch'),
	('zwei', 'drei'),
	('drei', 'vier'),
	('zwei', 'vier'),
	('zwei', 'fünf'),
	('zwei', 'sechs'),
	('zwei', 'sieben'),
	('vier','iv'),
	('drei', 'zehn'),
	('drei', 'zwanzig'),
	('drei', 'dreißig'),
	('drei', 'vierzig'),
	('drei', 'hundert')
    ],
    NOT_SIMILAR : [
	('Haus', 'Philosophie'),
	('Bundeskanzlerin', 'Donald Trump'),
        ('Landwirtschaft', 'Wolkenkratzer'),
	('Almabtrieb', 'Folge des Klimawandel'),
	('Konflikt im nahen Osten', 'Folge des Klimawandel'),
	('braun', 'kommunistisch'),
	('gehen', 'Maus')
    ],
    WORD_CALC : [
        { POS : 'König Frau', NEG : 'Mann', IS : 'Königin'},
        { POS : 'Paris Deutschland', NEG : 'Frankreich', IS : 'Berlin'},
        { POS : 'Washington Deutschland', NEG : 'USA', IS : 'Berlin'},
        { POS : 'Macron Deutschland', NEG : 'Frankreich', IS : 'Merkel'},
        { POS : 'Bundeskanzlerin Frankreich', NEG : 'Deutschland', IS : 'Präsident'},
        { POS : 'Bundesland Frankreich', NEG : 'Deutschland', IS : 'Departement'},
	{ POS : 'deutscher Außenminister', IS : 'Maas' },
	{ POS : 'deutsche Verteidigungsministerin', IS : 'Kramp-Karrenbauer' },
	{ POS : 'deutsche Verteidigungsministerin', IS : 'Leyen' },
	{ POS : 'französische Verteidigungsministerin', IS : 'Parly' },
	{ POS : 'US-Präsident', IS : 'Trump' },
	{ POS : 'US-Präsident', IS : 'Obama' },
	{ POS : 'früherer US-Präsident', IS : 'Obama' }
    ],
    WORD_CALC_NEG : [
	{ POS : 'US-Präsident', IS : 'Obama' }
    ]
}
"""
Standard testcases for German language.
"""

de = tkns.Append("_DE") * tkns.Lower() * tkns.Remove() * tkns.LemmaTokenizeText()
en = tkns.Append("_EN") * \
            tkns.Lower() * \
                tkns.Remove(stopwords=nltk.corpus.stopwords.words('english') + ['-pron-']) * \
                    tkns.LemmaTokenizeText(lemmatizer=spacy.load('en'))

STANDARD_DE_EN = {
    SYNONYM : [
        ( de('Ehemann'), en('husband') ),
        ( de('Auto'), en('car') ),
        ( de('König'), en('King') ),
        ( de('Königin'), en('Queen') )
    ] ,
    SIMILAR : [
        ( en('president'), en('Trump') ),
        ( de('Vereinigte Staaten'), en('USA') ),
        ( de('Kanzlerin'), en('Merkel') ),
        ( de('Kanzlerin'), en('Angela Merkel') ),
        ( en('President'), de('Donald Trump') ),
        ( de('Kaiser'), en('king') ),
        ( de('Landwirtschaft'), de('Ackerbau') ),
        ( de('Bundeskanzler von Österreich'), en('Sebastian Kurz') )
    ],
    WORD_CALC : [
        { POS : de('König Frau'), NEG : de('Mann'), IS : de('Königin') },
        { POS : en('king woman'), NEG : de('man'), IS : de('queen') }
    ]
}

STANDARD_EN = {
	MOD : tkns.Lower() * \
        tkns.Remove(stopwords=tkns.nltk.corpus.stopwords.words('english') + ['-pron-']) * \
        tkns.LemmaTokenizeText(lemmatizer=tkns.spacy.load("en")) * \
        tkns.ReSub(["[^\s]*>[^\s]+", "html5[^\s]*", "px[^\s]*", ">+read more", "[^\s]+\:\/\/[^\s]+"], " "),
	SYNONYM : [
		('ask', 'demand'),
		('beautiful', 'pretty'),
		('begin', 'start'),
		('big', 'huge'),
		('difference', 'disagreement'),
		('false', 'fake'),
		('fast', 'quick'),
		('king', 'monarch'),
		('PPACA', 'Patient Protection and Affordable Care Act')
	],
	SIMILAR : [
		('administration', 'government'),
		('president', 'Trump'),
		('president', 'Obama'),
		('president', 'Putin'),
		('president', 'Erdogan'),
		('former president', 'Trump'),
		('former president', 'Obama'),
		('chancellor', 'Merkel'),
		('chancellor', 'Schröder'),
		('former chancellor', 'Merkel'),
		('former chancellor', 'Schröder'),
		('fake news', 'Trump'),
		('great britain', 'brexit'),
		('queen', 'elizabeth'),
		('go', 'run'),
		('farming', 'agriculture'),
		('car', 'vehicle'),
		('polution', 'smoke'),
		('Obamacare', 'health insurance'),
		('Obamacare', 'Patient Protection and Affordable Care Act'),
		('Obamacare', 'Increasing Capital Access for Rural Small Businesses')
	],
	NOT_SIMILAR : [
		('car', 'tree'),
		('human', 'dark'),
		('go', 'think'),
		('Obamacare', 'Stratigic Arms Reduction Talks')
	],
	WORD_CALC : [
		{ POS : 'woman king', NEG : 'man', IS : 'queen' }
	]
}
