import unittest, lab2
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
import json
import os.path
import torch
from torch.autograd import Variable
from torch import optim
from sklearn.linear_model import LogisticRegression

class TestStep(unittest.TestCase):

    def setUp(self):
        self.y_tr,self.x_tr = lab2.read_data('lyrics-train.csv',preprocessor=lab2.bag_of_words)
        self.y_dv,self.x_dv = lab2.read_data('lyrics-dev.csv',preprocessor=lab2.bag_of_words)
        self.counts_tr = lab2.aggregate_counts(self.x_tr)
        self.counts_dv = lab2.aggregate_counts(self.x_dv)

        self.labels = set(self.y_tr)
        self.x_tr_pruned, self.vocab = lab2.prune_vocabulary(self.counts_tr, self.x_tr, 10)
        self.x_dv_pruned, self.vocab2 = lab2.prune_vocabulary(self.counts_tr, self.x_dv, 10)

        self.X_tr = lab2.make_numpy(self.x_tr_pruned,self.vocab)
        self.X_dv = lab2.make_numpy(self.x_dv_pruned,self.vocab)
        self.label_set = sorted(list(set(self.y_tr)))
        self.Y_tr = np.array([self.label_set.index(y_i) for y_i in self.y_tr])
        self.Y_dv = np.array([self.label_set.index(y_i) for y_i in self.y_dv])
        self.X_tr_var = Variable(torch.from_numpy(self.X_tr.astype(np.float32)))
        self.X_dv_var = Variable(torch.from_numpy(self.X_dv.astype(np.float32)))
        self.Y_tr_var = Variable(torch.from_numpy(self.Y_tr))
        self.Y_dv_var = Variable(torch.from_numpy(self.Y_dv))

    @weight(5)
    def test_d1_1_bow(self):
        self.assertEqual(len(self.x_tr), len(self.y_tr))
        self.assertEqual(self.x_tr[4]['all'],5)
        self.assertEqual(self.x_tr[41]['angels'],1)
        self.assertEqual(self.x_tr[410]['angels'],0)
        self.assertEqual(len(self.x_tr[1144]),124)

    @weight(5)
    def test_d1_2_agg(self):
        self.assertEqual(self.counts_dv['you'],5542)
        self.assertEqual(len(self.counts_dv),9006)
        self.assertEqual(self.counts_dv['money'],92)

    @weight(5)
    def test_d1_3a_oov(self):
        self.assertEqual(len(lab2.compute_oov(self.counts_dv,self.counts_tr)),2677)
        self.assertEqual(len(lab2.compute_oov(self.counts_tr,self.counts_dv)),30459)

    @weight(5)
    def test_d1_4_prune(self):
        x_tr_pruned, vocab = lab2.prune_vocabulary(self.counts_tr,self.x_tr,3)
        x_dv_pruned, vocab2 = lab2.prune_vocabulary(self.counts_tr,self.x_dv,3)

        self.assertEqual(len(vocab),len(vocab2))
        self.assertEqual(len(vocab),11824)

        self.assertEqual(len(self.x_dv[95].keys())-len(x_dv_pruned[95].keys()),8)

    @weight(5)
    def test_d2_1_featvec(self):
        label = '1980s'
        fv = lab2.make_feature_vector({'test':1,'case':2},label)
        self.assertEqual(len(fv),3)
        self.assertEqual(fv[(label,'test')],1)
        self.assertEqual(fv[(label,'case')],2)
        self.assertEqual(fv[(label,lab2.OFFSET)],1)

    @weight(5)
    def test_d2_2_predict(self):
        y_hat,scores = lab2.predict(self.x_tr_pruned[0],lab2.theta_hand,self.labels)
        self.assertEqual(scores['pre-1980'],0.1)
        self.assertAlmostEqual(scores['2000s'],1.3,places=5)
        self.assertEqual(y_hat,'2000s')
        self.assertEqual(scores['1980s'],0.0)

        y_hat = lab2.predict_all(self.x_dv_pruned,lab2.theta_hand,self.labels)
        self.assertAlmostEqual(lab2.acc(y_hat,self.y_dv),.3422222, places=5)

    @weight(5)
    def test_d3_1_corpus_counts(self):
        # public
        iama_counts = lab2.get_corpus_counts(self.x_tr_pruned,self.y_tr,"1980s");
        self.assertEqual(iama_counts['today'],50)
        self.assertEqual(iama_counts['yesterday'],14)
        self.assertEqual(iama_counts['internets'],0)

    @weight(15)
    def test_d3_2_pxy(self):
        # check that distribution normalizes to one
        log_pxy = lab2.estimate_pxy(self.x_tr_pruned,self.y_tr,"1980s",0.1,self.vocab)
        self.assertAlmostEqual(np.exp(list(log_pxy.values())).sum(),1)

        # check that values are correct
        self.assertAlmostEqual(log_pxy['money'],-7.6896,places=3)
        self.assertAlmostEqual(log_pxy['fly'],-8.6369,places=3)

        log_pxy_more_smooth = lab2.estimate_pxy(self.x_tr_pruned,self.y_tr,"1980s",10,self.vocab)
        self.assertAlmostEqual(log_pxy_more_smooth['money'],-7.8013635125541789,places=3)
        self.assertAlmostEqual(log_pxy_more_smooth['tonight'], -6.4054072405225515,places=3)

    @weight(15)
    def test_d3_3a_nb(self):
        theta_nb = lab2.estimate_nb(self.x_tr_pruned,self.y_tr,0.1)

        y_hat,scores = lab2.predict(self.x_tr_pruned[55],theta_nb,self.labels)
        self.assertAlmostEqual(scores['2000s'],-1840.5064690929203,places=3)
        self.assertEqual(y_hat,'1980s')

        y_hat,scores = lab2.predict(self.x_tr_pruned[155],theta_nb,self.labels)
        self.assertAlmostEqual(scores['1980s'], -2153.0199277981355, places=3)
        self.assertEqual(y_hat,'2000s')

    @weight(5)
    def test_d3_3b_nb(self):
        y_hat_dv = lab2.read_predictions('nb-dev.preds')
        self.assertGreaterEqual(lab2.acc(y_hat_dv,self.y_dv),.46)

    @weight(5)
    def test_d3_4a_nb_best(self):
        vals = np.logspace(-3,2,11)
        best_smoother, scores = lab2.find_best_smoother(self.x_tr_pruned,self.y_tr,self.x_dv_pruned,self.y_dv,[1e-3,1e-2,1e-1,1])
        self.assertGreaterEqual(scores[.1],.46)
        self.assertGreaterEqual(scores[.01],.45)

    @weight(5)
    def test_d4_1_numpy(self):
        X_dv = lab2.make_numpy(self.x_dv_pruned,self.vocab2)
        self.assertEqual(X_dv.sum(), 137687)
        self.assertEqual(X_dv.sum(axis=1)[4], 417)
        self.assertEqual(X_dv.sum(axis=1)[144], 175)

        self.assertEqual(X_dv.sum(axis=0)[10], 3)
        self.assertEqual(X_dv.sum(axis=0)[100], 0)

    @weight(10)
    def test_d4_2_model_acc(self):
        scikit_log_reg = lab2.better_model()
        logisticRegr=scikit_log_reg.fit(self.X_tr, self.Y_tr)
        dev_acc = logisticRegr.score(self.X_dv, self.Y_dv)
        self.assertGreaterEqual(dev_acc,0.5)

    @weight(10)
    def test_topfeat_LR(self):
        scikit_log_reg = LogisticRegression()
        logisticRegr=scikit_log_reg.fit(self.X_tr, self.Y_tr)
        # print(set(lab2.get_top_features_LR(scikit_log_reg, self.vocab,self.label_set,'pre-1980',k=10)[0]))
        self.assertTrue(set(lab2.get_top_features_LR(scikit_log_reg, self.vocab,self.label_set,'pre-1980',k=10)[0]) == set(['lord', 'boogie', 'very', 'feelin', 'darling', 'dancing', 'till', 'mornin', 'fool', 'percussion']))
        self.assertTrue(set(lab2.get_top_features_LR(scikit_log_reg, self.vocab,self.label_set,'pre-1980',k=10)[1]) == set(['step', 'under', 'meant', 'runaway', 'perfect', 'yo', 'open', 'front', 'body', 'hit']))

        self.assertTrue(set(lab2.get_top_features_LR(scikit_log_reg, self.vocab,self.label_set,'1980s',k=10)[0]) == set(['wall', 'america', 'standing', 'tumble', 'poison', 'shout', 'chance', 'heat', 'cut', 'took']))
        self.assertTrue(set(lab2.get_top_features_LR(scikit_log_reg, self.vocab,self.label_set,'1980s',k=10)[1]) == set(['floor', 'hes', 'god', 'percussion', 'thinkin', 'finally', 'window', 'mama', 'lord', 'sing']))

        self.assertTrue(set(lab2.get_top_features_LR(scikit_log_reg, self.vocab,self.label_set,'1990s',k=10)[0]) == set(['hit', 'yo', 'cuz', 'saw', 'dick', 'cradle', 'front', 'push', 'needed', 'rush']))
        self.assertTrue(set(lab2.get_top_features_LR(scikit_log_reg, self.vocab,self.label_set,'1990s',k=10)[1]) == set(['dancing', 'second', 'chance', 'born', 'use', 'those', 'pretty', 'meaning', 'today', 'other']))

        self.assertTrue(set(lab2.get_top_features_LR(scikit_log_reg, self.vocab,self.label_set,'2000s',k=10)[0]) == set(['wit', 'shut', 'shorty', 'club', 'three', 'jeans', 'side', 'ass', 'full', 'bitch']))
        self.assertTrue(set(lab2.get_top_features_LR(scikit_log_reg, self.vocab,self.label_set,'2000s',k=10)[1]) == set(['lovin', 'rhythm', 'hip', 'lover', 'must', 'honey', 'boogie', 'woman', 'youve', 'fool']))

    @weight(10)
    def test_topfeat_NB(self):
        vals = np.logspace(-3,2,11)
        best_smoother, scores = lab2.find_best_smoother(self.x_tr_pruned,self.y_tr,self.x_dv_pruned,self.y_dv,vals)
        theta_nb = lab2.estimate_nb(self.x_tr_pruned,self.y_tr,best_smoother)
        self.assertTrue(set(lab2.get_top_features_NB(theta_nb, self.label_set,'pre-1980',k=10)[0]) == set(['you', 'the', 'i', 'to', 'and', 'a', 'me', 'my', 'it', 'love']))
        self.assertTrue(set(lab2.get_top_features_NB(theta_nb, self.label_set,'1980s',k=10)[0]) == set(['you', 'the', 'i', 'to', 'me', 'a', 'and', 'it', 'my', 'love']))
        self.assertTrue(set(lab2.get_top_features_NB(theta_nb, self.label_set,'1990s',k=10)[0]) == set(['you', 'i', 'the', 'to', 'me', 'and', 'a', 'it', 'my', 'your']))
        self.assertTrue(set(lab2.get_top_features_NB(theta_nb, self.label_set,'2000s',k=10)[0]) == set(['you', 'i', 'the', 'me', 'and', 'to', 'a', 'it', 'my', 'in']))

    @weight(10)
    def test_PRF(self):
        scikit_log_reg = LogisticRegression()
        logisticRegr=scikit_log_reg.fit(self.X_tr, self.Y_tr)
        predictions = logisticRegr.predict(self.X_dv)
        a,b,c = lab2.get_PRF(predictions, self.Y_dv, self.label_set, 'pre-1980')
        self.assertTrue(abs(a-0.5078125) < 0.01)
        self.assertTrue(abs(b-0.5241935483870968) < 0.01)
        self.assertTrue(abs(c-0.5158730158730158) < 0.01)

        a,b,c = lab2.get_PRF(predictions, self.Y_dv, self.label_set, '1980s')
        self.assertTrue(abs(a-0.32967032967032966) < 0.01)
        self.assertTrue(abs(b-0.28846153846153844) < 0.01)
        self.assertTrue(abs(c-0.30769230769230765) < 0.01)

        a,b,c = lab2.get_PRF(predictions, self.Y_dv, self.label_set, '1990s')
        self.assertTrue(abs(a-0.391304347826087) < 0.01)
        self.assertTrue(abs(b-0.37894736842105264) < 0.01)
        self.assertTrue(abs(c-0.3850267379679144) < 0.01)

        a,b,c = lab2.get_PRF(predictions, self.Y_dv, self.label_set, '2000s')
        self.assertTrue(abs(a-0.6258992805755396) < 0.01)
        self.assertTrue(abs(b-0.6850393700787402) < 0.01)
        self.assertTrue(abs(c-0.6541353383458647) < 0.01)
