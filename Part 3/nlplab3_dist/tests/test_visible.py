import unittest, lab3
from gradescope_utils.autograder_utils.decorators import weight
import nltk
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline

class TestStep(unittest.TestCase):

    def setUp(self):
        food = ['barley', 'castor-oil', 'cocoa', 'coconut', 'coconut-oil', 'coffee', 'copra-cake''grain', 'groundnut', 'groundnut-oil', 'potato''soy-meal', 'soy-oil', 'soybean', 'sugar', 'sun-meal', 'sun-oil', 'sunseed', 'tea', 'veg-oil', 'wheat']
        natural_resources = ['alum', 'fuel', 'gas', 'gold', 'iron-steel', 'lead', 'nat-gas', 'palladium', 'propane', 'tin', 'zinc']
        nltk.download('reuters')
        nltk.download('punkt')
        corpus = nltk.corpus.reuters
        self.food_corpus = corpus.raw(categories=food)
        self.natr_corpus = corpus.raw(categories=natural_resources)

    @weight(5)
    def test_d1_1_tk(self):
        nltk.download('punkt')
        food_corpus_tk = lab3.tokenize_corpus(self.food_corpus)
        natr_corpus_tk = lab3.tokenize_corpus(self.natr_corpus)
        self.assertEqual(food_corpus_tk[25][5],'Monday')
        self.assertEqual(natr_corpus_tk[25][5],'are')

    @weight(5)
    def test_d1_2_pad(self):
        nltk.download('punkt')
        food_corpus_tk = lab3.tokenize_corpus(self.food_corpus)
        natr_corpus_tk = lab3.tokenize_corpus(self.natr_corpus)
        food_corpus_tk_pd = lab3.pad_corpus(food_corpus_tk)
        natr_corpus_tk_pd = lab3.pad_corpus(natr_corpus_tk)
        self.assertEqual(food_corpus_tk_pd[35][0], '<s>')
        self.assertEqual(natr_corpus_tk_pd[35][-1], '</s>')
        self.assertEqual(len(food_corpus_tk_pd[45]), 14)
        self.assertEqual(len(natr_corpus_tk_pd[45]), 19)
        self.assertEqual(len(food_corpus_tk_pd[45]) - len(food_corpus_tk[45]), 2)

    @weight(5)
    def test_d1_3_spc(self):
        nltk.download('punkt')
        food_corpus_tk = lab3.tokenize_corpus(self.food_corpus)
        natr_corpus_tk = lab3.tokenize_corpus(self.natr_corpus)
        food_corpus_tk_pd = lab3.pad_corpus(food_corpus_tk)
        natr_corpus_tk_pd = lab3.pad_corpus(natr_corpus_tk)
        food_corpus_tr, food_corpus_te = lab3.split_corpus(food_corpus_tk_pd)
        natr_corpus_tr, natr_corpus_te = lab3.split_corpus(natr_corpus_tk_pd)
        self.assertEqual(len(food_corpus_tr), 4888)
        self.assertEqual(len(food_corpus_te), 1222)
        self.assertEqual(len(natr_corpus_tr), 2610)
        self.assertEqual(len(natr_corpus_te), 653)
        self.assertEqual(food_corpus_te[3][5], 'by')
        self.assertEqual(natr_corpus_te[1][2], 'Project')

    @weight(20)
    def test_d1_4_cn(self):
        nltk.download('punkt')
        food_corpus_tk = lab3.tokenize_corpus(self.food_corpus)
        natr_corpus_tk = lab3.tokenize_corpus(self.natr_corpus)
        food_corpus_tk_pd = lab3.pad_corpus(food_corpus_tk)
        natr_corpus_tk_pd = lab3.pad_corpus(natr_corpus_tk)
        food_corpus_tr, food_corpus_te = lab3.split_corpus(food_corpus_tk_pd)
        natr_corpus_tr, natr_corpus_te = lab3.split_corpus(natr_corpus_tk_pd)
        food_ngrams, food_vocab_man = lab3.count_ngrams(food_corpus_tr, 3)
        natr_ngrams, natr_vocab_man = lab3.count_ngrams(natr_corpus_tr, 3)
        self.assertEqual(len(food_ngrams.keys()), 181387)
        self.assertEqual(len(natr_ngrams.keys()), 105612)
        self.assertEqual(food_ngrams[('sold', 'the')], 2)
        self.assertEqual(natr_ngrams[('extracting', 'the')], 2)
        self.assertEqual(len(food_vocab_man), 12728)
        self.assertEqual(len(natr_vocab_man), 8972)
        self.assertEqual(sorted(food_vocab_man)[3200], 'ANALYSTS')
        self.assertEqual(sorted(natr_vocab_man)[3210], 'NGX')

    @weight(5)
    def test_d1_5_es(self):
        nltk.download('punkt')
        food_corpus_tk = lab3.tokenize_corpus(self.food_corpus)
        natr_corpus_tk = lab3.tokenize_corpus(self.natr_corpus)
        food_corpus_tk_pd = lab3.pad_corpus(food_corpus_tk)
        natr_corpus_tk_pd = lab3.pad_corpus(natr_corpus_tk)
        food_corpus_tr, food_corpus_te = lab3.split_corpus(food_corpus_tk_pd)
        natr_corpus_tr, natr_corpus_te = lab3.split_corpus(natr_corpus_tk_pd)
        food_ngrams, food_vocab_man = lab3.count_ngrams(food_corpus_tr, 3)
        natr_ngrams, natr_vocab_man = lab3.count_ngrams(natr_corpus_tr, 3)
        self.assertEqual(lab3.estimate(food_ngrams, ['palm'], ['producer', 'of']), 0.25)
        self.assertEqual(lab3.estimate(natr_ngrams, ['basis'], ['tested', 'the']), 0.5)

    @weight(10)
    def test_d2_1_gp(self):
        nltk.download('punkt')
        food_corpus_tk = lab3.tokenize_corpus(self.food_corpus)
        natr_corpus_tk = lab3.tokenize_corpus(self.natr_corpus)
        food_train, food_vocab = padded_everygram_pipeline(3, food_corpus_tk[:int(0.8*len(food_corpus_tk))])
        natr_train, natr_vocab = padded_everygram_pipeline(3, natr_corpus_tk[:int(0.8*len(natr_corpus_tk))])
        food_test = sum([['<s>'] + x + ['</s>'] for x in food_corpus_tk[int(0.8*len(food_corpus_tk)):]],[])
        natr_test = sum([['<s>'] + x + ['</s>'] for x in natr_corpus_tk[int(0.8*len(natr_corpus_tk)):]],[])
        food_lm = Laplace(3)
        natr_lm = Laplace(3)
        food_lm.fit(food_train, food_vocab)
        natr_lm.fit(natr_train, natr_vocab)
        self.assertEqual(int(lab3.get_perplexity(food_lm, food_test[:2500])), 7318)
        self.assertEqual(int(lab3.get_perplexity(food_lm, natr_test[:2500])), 7309)
        self.assertEqual(int(lab3.get_perplexity(natr_lm, natr_test[:2500])), 5222)
        self.assertEqual(int(lab3.get_perplexity(natr_lm, food_test[:2500])), 5354)

    @weight(40)
    def test_d3_1_vary(self):
        nltk.download('punkt')
        food_corpus_tk = lab3.tokenize_corpus(self.food_corpus)
        n_gram_orders = [2, 3]
        train_corpus = food_corpus_tk[:int(0.8*len(food_corpus_tk))]
        test_corpus = food_corpus_tk[int(0.8*len(food_corpus_tk)): int(0.85*len(food_corpus_tk))]
        results = lab3.vary_ngram(train_corpus, test_corpus, n_gram_orders)
        self.assertEqual(int(results[2]), 7387)
        self.assertEqual(int(results[3]), 7428)
