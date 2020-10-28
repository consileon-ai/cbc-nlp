"""
consileon.nlp.translate
========================

Translate single words in texts literally using dictionaries.
Used e.g. for generating bilingual word2vec models.
"""
import re
import random

import consileon.nlp.tokens as tkns

RE_LEFT_SIDE = re.compile(
    r"^\s*(?:\[[~ '\w.]+\])?(?:\([ '\w]+\))?\s*([ \'\-\w]+\.?)(?: +\([ \d\w.:]+\))?\s+(?:/.*/)$")
RE_RIGHT_SIDE = re.compile(
    r"\s*(?:\([ \'\w]+\))?([ \-()\w./\':]+[.!?]?)(?:\.\.\.)?(?: +\([ \d\w.:]+\))?(?: +\[[ \d\w.]+\])?"
    r"(?: +<(?:[^<]*[fnm;]+[^>]*)>)?(?: \([.\d]+\))?$")
RE_IN_BRACKETS = re.compile(r"\(\w+\)|\[\w+\]")
RE_REMOVE_LEADING_ENGLISH_WORDS = re.compile(r"^(a|to) ")
RE_REMOVE_LEADING_GERMAN_WORDS = re.compile(r"^(ein|eine) ")


def get_freedict_org_pairs(dict_filename,
                           re_remove_leading_left_words=RE_REMOVE_LEADING_ENGLISH_WORDS,
                           re_remove_leading_right_words=RE_REMOVE_LEADING_GERMAN_WORDS
                           ):
    file = open(dict_filename, 'r')
    line = file.readline()
    while line:
        m = RE_LEFT_SIDE.match(line)
        if m is not None:
            orig_l = line.strip()
            l_ = re_remove_leading_left_words.sub("", m.group(1))
            line = file.readline()
            orig_r = line.strip()
            r = []
            for w in line.split(','):
                m = RE_RIGHT_SIDE.match(w.strip())
                if m is not None:
                    r.append(re_remove_leading_right_words.sub("", RE_IN_BRACKETS.sub("", m.group(1)).strip()))
                yield l_, r, orig_l, orig_r
        line = file.readline()
    file.close()


def get_language_dicts(an_iterable):
    left_to_right = {}
    right_to_left = {}
    for l_, r, l_orig, r_orig in an_iterable:
        if l_ in left_to_right:
            left_to_right[l_].update(r)
        else:
            left_to_right[l_] = set(r)
        for anR in r:
            if anR in right_to_left:
                right_to_left[anR].add(l_)
            else:
                right_to_left[anR] = {l_}
    return {l_: list(r) for l_, r in left_to_right.items() if len(r) > 0},\
           {l_: list(r) for l_, r in right_to_left.items() if len(r) > 0}


def get_freedict_org_dicts(dict_filename,
                           re_remove_leading_left_words=RE_REMOVE_LEADING_ENGLISH_WORDS,
                           re_remove_leading_right_words=RE_REMOVE_LEADING_GERMAN_WORDS
                           ):
    return get_language_dicts(
        get_freedict_org_pairs(dict_filename,
                               re_remove_leading_left_words=re_remove_leading_left_words,
                               re_remove_leading_right_words=re_remove_leading_right_words
                               )
    )


def reduce_to_noun_if_exists(a_dict):
    for l_, r in a_dict.items():
        r = [w for w in r if w[0].isupper()]
        if len(r) > 0:
            a_dict[l_] = r


class PartiallyTranslate(tkns.ItemModifier):
    def __init__(self,
                 dictionary,
                 translate_freq=5,
                 orig_token_modifier=tkns.Append("_ORIG"),
                 tokenize_translated=tkns.Append("_TRANS") * tkns.TokenizeText()
                 ):
        self.dictionary = dictionary
        self.translateFreq = translate_freq
        self.origTokenModifier = orig_token_modifier
        self.tokenizeTranslated = tokenize_translated

    def __call__(self, tokens):
        s = random.randint(0, 20)
        i = 0
        l_ = len(tokens)
        new_tokens = []
        while i < l_:
            j = i + self.translateFreq - 1
            while j < l_:
                if tokens[j] in self.dictionary:
                    break
                j += 1
            new_tokens += self.origTokenModifier(tokens[i:j])
            if j < l_:
                t = self.dictionary[tokens[j]]
                new_tokens += self.tokenizeTranslated(t[s % len(t)])
                j += 1
                s += 1
            i = j
        return new_tokens
