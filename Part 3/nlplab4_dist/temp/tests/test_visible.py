import unittest
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
import sys
sys.path.append("../")
if __name__ == "__main__":
    from nlp_lab4 import lab4
else:
    import lab4

class TestStep(unittest.TestCase):

    @weight(5)
    def test_1_1(self):
        test_corpus = ["START All that glitters isn't gold END".split(" "), "START All's well that ends well END".split(" ")]
        test_corpus_words, num_corpus_words = lab4.distinct_words(test_corpus)
        # Correct answers
        ans_test_corpus_words = sorted(list(set(["START", "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", "END"])))
        ans_num_corpus_words = len(ans_test_corpus_words)
        # Test correct number of words
        self.assertEqual(num_corpus_words,ans_num_corpus_words)
        # Test correct words
        self.assertEqual(test_corpus_words,ans_test_corpus_words)

    @weight(15)
    def test_1_2(self):
        # Define toy corpus and get student's co-occurrence matrix
        test_corpus = ["START All that glitters isn't gold END".split(" "), "START All's well that ends well END".split(" ")]
        M_test, word2Ind_test = lab4.compute_co_occurrence_matrix(test_corpus, window_size=1)
        # Correct M and word2Ind
        M_test_ans = np.array(
            [[0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,],
             [0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,],
             [0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,],
             [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,],
             [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,],
             [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,],
             [0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,],
             [0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,],
             [1., 0., 0., 0., 1., 1., 0., 0., 0., 1.,],
             [0., 1., 1., 0., 1., 0., 0., 0., 1., 0.,]]
        )
        word2Ind_ans = {'All': 0, "All's": 1, 'END': 2, 'START': 3, 'ends': 4, 'glitters': 5, 'gold': 6, "isn't": 7, 'that': 8, 'well': 9}
        # Test correct word2Ind
        self.assertTrue(word2Ind_ans == word2Ind_test)
        # Test correct M shape
        self.assertTrue(M_test.shape == M_test_ans.shape)
        # Test correct M values
        self.assertTrue((M_test == M_test_ans).all())

    @weight(10)
    def test_1_3(self):
        # Define toy corpus and run student code
        test_corpus = ["START All that glitters isn't gold END".split(" "), "START All's well that ends well END".split(" ")]
        M_test, word2Ind_test = lab4.compute_co_occurrence_matrix(test_corpus, window_size=1)
        M_test_reduced = lab4.reduce_to_k_dim(M_test, k=2)
        # Test proper dimensions
        self.assertTrue(M_test_reduced.shape[0] == 10)
        self.assertTrue(M_test_reduced.shape[1] == 2)

if __name__ == "__main__":
    tests = TestStep()
    tests.test_1_1()
    