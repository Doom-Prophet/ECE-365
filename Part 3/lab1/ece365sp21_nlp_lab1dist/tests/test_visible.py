import unittest, lab1, nltk
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
import json
import os.path

class TestStep(unittest.TestCase):

    @weight(5)
    def test_get_top_10(self):
        freqs = {"one":1, "two":2, "three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8, "nine":9,
        "ten":10, "eleven":11, "twelve":12, "thirteen":13, "fourteen":14, "fifteen":15, "sixteen":16,
        "seventeen":17, "eighteen":18, "nineteen":19, "twenty":20}
        self.assertTrue(lab1.get_top_10(freqs) == ['twenty', 'nineteen', 'eighteen', 'seventeen',
        'sixteen', 'fifteen', 'fourteen', 'thirteen', 'twelve', 'eleven'])

    @weight(5)
    def test_get_bottom_10(self):
        freqs = {"one":1, "two":2, "three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8, "nine":9,
        "ten":10, "eleven":11, "twelve":12, "thirteen":13, "fourteen":14, "fifteen":15, "sixteen":16,
        "seventeen":17, "eighteen":18, "nineteen":19, "twenty":20}
        self.assertTrue(lab1.get_bottom_10(freqs) == ['one', 'two', 'three', 'four',
        'five', 'six', 'seven', 'eight', 'nine', 'ten'])

    @weight(15)
    def test_get_freqs(self):
        puncts = ['.','!','?',',',';',':','[', ']', '{', '}', '(', ')', '\'', '\"']
        nltk.download('reuters')
        raw_corpus = nltk.corpus.reuters.raw()
        freqs = lab1.get_freqs(raw_corpus, puncts)
        self.assertTrue(lab1.get_top_10(freqs) == ['the', 'of', 'to', 'in', 'and', 'said', 'a', 'mln', 's', 'vs'])
        self.assertTrue(lab1.get_bottom_10(freqs) == ['inflict', 'sheen', 'stand-off', 'avowed', 'kilolitres', 'kilowatt/hour', 'janunary/march', 'pineapples', 'hasrul', 'paian'])

    @weight(5)
    def test_get_percentage_singletons(self):
        with open(os.path.dirname(__file__) + '/../freqs.json') as fp:
            freqs = json.load(fp)
        self.assertTrue(lab1.get_percentage_singletons(freqs)>40.4)
        self.assertTrue(lab1.get_percentage_singletons(freqs)<40.6)

    @weight(5)
    def test_get_freqs_lemmatized(self):
        puncts = ['.','!','?',',',';',':','[', ']', '{', '}', '(', ')', '\'', '\"']
        nltk.download('reuters')
        nltk.download('wordnet')
        raw_corpus = nltk.corpus.reuters.raw()
        freqs_lemmatized = lab1.get_freqs_lemmatized(raw_corpus, puncts)
        self.assertTrue(lab1.get_top_10(freqs_lemmatized) == ['the', 'of', 'to', 'in', 'be', 'say', 'and', 'a', 'mln', 's'])
        self.assertTrue(lab1.get_bottom_10(freqs_lemmatized) == ['inflict', 'sheen', 'stand-off', 'avow', 'kilolitres', 'kilowatt/hour', 'janunary/march', 'pineapples', 'hasrul', 'paian'])
        self.assertTrue(lab1.get_percentage_singletons(freqs_lemmatized)>41.9)
        self.assertTrue(lab1.get_percentage_singletons(freqs_lemmatized)<42.2)

    @weight(5)
    def test_get_freqs_stemming(self):
        puncts = ['.','!','?',',',';',':','[', ']', '{', '}', '(', ')', '\'', '\"']
        nltk.download('reuters')
        nltk.download('wordnet')
        raw_corpus = nltk.corpus.reuters.raw()
        freqs_stemming = lab1.get_freqs_stemming(raw_corpus, puncts)
        self.assertTrue(lab1.get_top_10(freqs_stemming) == ['the', 'of', 'to', 'in', 'and', 'said', 'a', 'mln', 'it', 's'])
        self.assertTrue(lab1.get_bottom_10(freqs_stemming) == ['inflict', 'sheen', 'stand-off', 'avow', 'kilolitr', 'kilowatt/hour', 'janunary/march', 'hasrul', 'paian', 'sawn'])
        self.assertTrue(lab1.get_percentage_singletons(freqs_stemming)>41.9)
        self.assertTrue(lab1.get_percentage_singletons(freqs_stemming)<42.2)

    @weight(5)
    def test_vocab_size(self):
        with open(os.path.dirname(__file__) + '/../freqs.json') as fp:
            freqs = json.load(fp)
        with open(os.path.dirname(__file__) + '/../freqs_lemmatized.json') as fp:
            freqs_lemmatized = json.load(fp)
        with open(os.path.dirname(__file__) + '/../freqs_stemming.json') as fp:
            freqs_stemming = json.load(fp)
        self.assertTrue(lab1.size_of_raw_corpus(freqs) == 33206)
        self.assertTrue(lab1.size_of_stemmed_raw_corpus(freqs_stemming) == 25778)
        self.assertTrue(lab1.size_of_lemmatized_raw_corpus(freqs_lemmatized) == 29032)

    @weight(5)
    def test_percentage_of_unseen_vocab(self):
        nltk.download('reuters')
        raw_corpus = nltk.corpus.reuters.raw()
        self.assertTrue(lab1.percentage_of_unseen_vocab(raw_corpus.split()[:100], raw_corpus.split()[-100:], 100) == 0.79)
        self.assertTrue(lab1.percentage_of_unseen_vocab(raw_corpus.split()[:1000], raw_corpus.split()[-1000:], 1000) == 0.464)
        self.assertTrue(lab1.percentage_of_unseen_vocab(raw_corpus.split()[:10000], raw_corpus.split()[-10000:], 10000) == 0.2182)
        self.assertTrue(lab1.percentage_of_unseen_vocab(raw_corpus.split()[:100000], raw_corpus.split()[-100000:], 100000) == 0.10077)
        self.assertTrue(lab1.percentage_of_unseen_vocab(raw_corpus.split()[:500000], raw_corpus.split()[-500000:], 500000) == 0.052344)

    @weight(15)
    def test_frac_80_perc(self):
        with open(os.path.dirname(__file__) + '/../freqs.json') as fp:
            freqs = json.load(fp)
        self.assertTrue(lab1.frac_80_perc(freqs) > 0.033)
        self.assertTrue(lab1.frac_80_perc(freqs) < 0.034)

    @weight(15)
    def test_get_TTRs(self):
        languages = ['Italian-Latin1', 'English-Latin1', 'German_Deutsch-Latin1', 'Finnish_Suomi-Latin1']
        nltk.download('udhr')
        TTRs = lab1.get_TTRs(languages)
        self.assertTrue(TTRs['Italian-Latin1'] == [64, 110, 143, 179, 221, 260, 286, 326, 355, 386, 412, 426, 451])
        self.assertTrue(TTRs['English-Latin1'] == [57, 99, 133, 167, 207, 231, 262, 292, 318, 339, 358, 381, 403])
        self.assertTrue(TTRs['German_Deutsch-Latin1'] == [63, 113, 155, 204, 254, 284, 324, 358, 388, 418, 446, 475, 504])
        self.assertTrue(TTRs['Finnish_Suomi-Latin1'] == [74, 137, 192, 252, 303, 356, 406, 459, 491, 537, 586, 631, 675])
