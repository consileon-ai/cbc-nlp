"""
consileon.nlp.w2v_test_framework
=================================

A framework meant to ease the testing of word2vec models.

Many testcases (e.g. consisting of similar words) are combined in a testset. This
testset is evaluated using a word2vec model. Operations for nlp cleansing which have
been used for building the models can be feed in the model and are automatically applied
so the the testcases can be written in "normal language".

The results (which represent similarities of words or expressions, e.g.) are given in such
a way that the can be either used by "machines" in further processing, or by humans.

Example:

::

	from consileon.nlp.w2v_test_framework import TestW2vModel
	ts = {
		'synonym' : [
			('huge', 'big'),
		],
		'similar' : [
			('go', 'ride')
		],
		'not_similar' : [
			('tree', 'white')
		]
	}

"""

import consileon.nlp.pipeline as tkns
import consileon.nlp.word2vec_tools as w2v
import pandas as pd
from consileon.nlp.w2v_const import SYNONYM, SIMILAR, NOT_SIMILAR, WORD_CALC, WORD_CALC_NEG, POS, NEG, IS, MOD


class TestW2vModel:
    """
    Class to perform tests gensim-Word2Vec which are specified in a test set as
    given in the module "consileon.nlp.w2v_test_sets". A test set is a collection
    of test cases of (currently two) different types. The test cases in the test set
    are grouped, giving the the test set the structure of a python dictionary with
    the group names as keys.

    The (optional) entry with key 'modifier' (= consileon.nlp.w2v_const.MOD) in the
    test set plays a special role. It's a function which is applied to the strings
    or lists in the test cases. It's generally defined as the 'ItemModifier' which
    is used for nlp preparation of the input nlp of the word2vec model which will
    be tested with the test set.

    If not such modifier is found, the identity function is taken.

    If the input test set is selft empty (= None), the empty dictionary is assigned
    as test set. The makes sense for simple usage of the functions "eval_pair" and
    "eval_word_calc".
    """

    def __init__(self,
                 test_set,
                 pairs=[SYNONYM, SIMILAR, NOT_SIMILAR],
                 word_calcs=[WORD_CALC, WORD_CALC_NEG]
                 ):
        self.test_set = test_set
        if self.test_set is None:
            self.test_set = {}
        self.pairs = pairs
        self.word_calcs = word_calcs

    def __call__(self, model):
        result = {}
        for p in self.pairs:
            if p in self.test_set:
                result[p] = self.eval_pair_list(model, self.test_set[p])
        for w in self.word_calcs:
            if w in self.test_set:
                result[w] = self.eval_word_calcs(model, self.test_set[w])
        return result

    def eval_pair_list(self, model, l):
        return [self.eval_pair(model, a, b) for a, b in l]

    def eval_pair(self, model, a, b):

        """
        Evaluates the cosine simularity of the pair (a, b) for the given model,
        Before calculation the "modifier" given in the test set is used.

        If the method is meant to use without any modifier (or the identity modifier,
        resp.) it may be used as following:

        ::

            from consileon.nlp.w2v_test_framework import TestW2vModel
            TestW2vModel(None).eval_pair(model, 'big', 'huge')
            # or
            cos = lambda a, b : TestW2vModel(None).eval_pair(model, a, b)
            cos('big', 'huge')

        """

        mf = self.test_set.get(MOD, tkns.ItemModifier())
        return w2v.cos(model, mf(a), mf(b))

    def eval_word_calc(self, model, expr):
        mf = self.test_set.get(MOD, tkns.ItemModifier())
        default = []
        pos = expr.get(POS)
        if isinstance(pos, str):
            default = ""
        pos = mf(pos)
        expr_v = w2v.as_vector(model, pos) - w2v.as_vector(model, mf(expr.get(NEG, default)))
        is_ = mf(expr.get(IS))
        result_pos, result_cos = 21, 0.0
        if len(is_) > 0:
            rl = model.wv.similar_by_vector(expr_v, topn=20, restrict_vocab=None)
            rl = [i[0] for i in rl if i[0] not in pos]
            if is_[0] in rl:
                result_pos = rl.index(is_[0])
                result_cos = w2v.vector_cos(expr_v, model.wv[is_[0]])
            else:
                result_pos, result_cos = 20, 0.0
        return (result_pos, result_cos)

    def eval_word_calcs(self, model, l):
        return [self.eval_word_calc(model, expr) for expr in l]


def eval_test(model, test):
    test_to_list = lambda d: [(k, v) for k in d if isinstance(d[k], list) for v in d[k]]
    index = test_to_list(test.test_set)
    values = [v for _, v in test_to_list(test(model))]
    return pd.Series(values, index=index)
