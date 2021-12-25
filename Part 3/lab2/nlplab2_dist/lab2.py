from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import torch
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression

# A list of labels.
OFFSET = '**OFFSET**'

# deliverable 1.1
def bag_of_words(text):
    '''
    Count the number of word occurences for each document in the corpus

    :param text: a document, as a single string
    :returns: a Counter for a single document
    :rtype: Counter
    '''
    count = Counter()

    for i in text.split():
        count[i] += 1

    return count

# deliverable 1.2
def aggregate_counts(bags_of_words):
    '''
    Aggregate word counts for individual documents into a single bag of words representation

    :param bags_of_words: a list of bags of words as Counters from the bag_of_words method
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    '''

    # YOUR CODE GOES HERE
    return sum(bags_of_words, Counter())

# deliverable 1.3
def compute_oov(bow1, bow2):
    '''
    Return a set of words that appears in bow1, but not bow2

    :param bow1: a bag of words
    :param bow2: a bag of words
    :returns: the set of words in bow1, but not in bow2
    :rtype: set
    '''
    return set(bow1).difference(set(bow2))

# deliverable 1.4
def prune_vocabulary(training_counts, target_data, min_counts):
    '''
    prune target_data to only words that appear at least min_counts times in training_counts

    :param training_counts: aggregated Counter for training data
    :param target_data: list of Counters containing dev bow's
    :returns: new list of Counters, with pruned vocabulary
    :returns: list of words in pruned vocabulary
    :rtype: list of Counters, set
    '''
    p_ctr, p_word = [], []

    for i, j in training_counts.items():
        if j >= min_counts:
            p_word.append(i)

    for ctr in target_data:
        new_ctr = Counter()
        
        for i, j in ctr.items():
            if i in p_word:
                new_ctr[i] = j

        p_ctr.append(new_ctr)

    return p_ctr, p_word

### helper code

def read_data(filename,label='Era',preprocessor=bag_of_words):
    df = pd.read_csv(filename)
    return df[label].values,[preprocessor(string) for string in df['Lyrics'].values]

def oov_rate(bow1,bow2):
    return len(compute_oov(bow1,bow2)) / len(bow1.keys())


# hint! use this.
def argmax(scores):
    items = list(scores.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]

# This will no longer work for our purposes since python3's max does not guarantee deterministic ordering
# argmax = lambda x : max(x.items(),key=lambda y : y[1])[0]

# deliverable 2.1
def make_feature_vector(base_features,label):
    '''
    take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    '''
    feature_dict = defaultdict()
    feature_dict[(label, OFFSET)] = 1

    for i in base_features:
        feature_dict[(label, i)] = base_features[i]
        
    return feature_dict

# deliverable 2.2
def predict(base_features,weights,labels):
    '''
    prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict
    '''

    score = defaultdict()

    for i in labels:
        temp_fvec, temp_score = make_feature_vector(base_features, i), 0
        
        for j in temp_fvec:
            if j in weights:
                temp_score += weights[j]*temp_fvec[j]
        
        score[i] = temp_score

    return argmax(score), score

def predict_all(x,weights,labels):
    '''
    Predict the label for all instances in a dataset

    :param x: base instances
    :param weights: defaultdict of weights
    :returns: predictions for each instance
    :rtype: numpy array

    '''
    All_label = []

    for i in x:
        All_label.append(predict(i, weights, labels)[0])

    return np.array(All_label)

theta_hand = defaultdict(float,
                         {('2000s','money'):0.1,
                          ('2000s','name'):0.2,
                          ('1980s','tonight'):0.1,
                          ('2000s','man'):0.1,
                          ('1990s','fly'):0.1,
                          ('pre-1980',OFFSET):0.1
                         })

# deliverable 3.1
def get_corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    Example:
    x = [Counter({'aa': 1, 'bb': 2, 'cc': 3}),
        Counter({'aa': 1, 'dd': 2, 'ee': 3}),
        Counter({'bb': 1, 'cc': 2, 'dd': 3})]
    y = [1, 2, 1]
    label = 1
    get_corpus_counts(x,y,label) = {'aa': 1, 'bb': 3, 'cc': 5, 'dd': 3}

    """
    count = Counter()

    for i in range(len(y)):
        if y[i] == label:
            count.update(x[i])

    return defaultdict(int, count)  

# deliverable 3.2
def estimate_pxy(x,y,label,smoothing,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict

    '''
    count = Counter()
    Tot_count = get_corpus_counts(x, y, label)
    deno = len(vocab)*smoothing + sum(Tot_count.values())
    
    for i in vocab:
        count[i] = np.log((Tot_count[i] + smoothing)/deno)
    
    return defaultdict(int, count)

# deliverable 3.3
def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: a defaultdict of features and weights. features are tuples (label,base_feature).
    :rtype: defaultdict

    Hint: See predict() for the exact return type information.

    """

    count = Counter()
    Tot_count = aggregate_counts(x)
    vocabulary = []

    for i, j in Tot_count.items():
        if j >= 10:
            vocabulary.append(i)

    for k in set(y):
        corpus_counts = get_corpus_counts(x, y, k)
        common_denominator = len(vocabulary)*smoothing + sum(corpus_counts.values()) 
        
        for l in vocabulary:
            count[(k, l)] = np.log((corpus_counts[l] + smoothing) / common_denominator)
        
        count[(k, OFFSET)] = np.log(len(list(y[k == y])) / len(y))

    return defaultdict(float, count)

# deliverable 3.4
def find_best_smoother(x_tr_pruned,y_tr,x_dv_pruned,y_dv,smoothers):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: 1) best smoothing value, 2) a dictionary of smoothing values and dev set accuracy.
    :rtype: 1) float, 2) dictionary

    '''
    result = defaultdict()

    for i in smoothers:
        prior = estimate_nb(x_tr_pruned, y_tr, i)
        pre = predict_all(x_dv_pruned, prior, y_dv)
        result[i] = acc(pre, y_dv)

    best = argmax(result)

    return best, result

def acc(y_hat,y):
    return (y_hat == y).mean()

def write_predictions(y_hat,filename):
    with open(filename,'w') as fout:
        for y_hat_i in y_hat:
            fout.write(y_hat_i + "\n")

def read_predictions(filename):
    with open(filename,'r') as fin:
        return [line.rstrip() for line in fin.readlines()]

## these are just for fun

def f1(y_hat,y,label):
    tp = sum((y_hat==label) & (y==label))
    fp = sum((y_hat==label) & (y!=label))
    fn = sum((y_hat!=label) & (y==label))
    #print tp,fp,fn
    r = tp/float(tp + fn + 1e-10)
    p = tp/float(tp + fp + 1e-10)
    f = 2 * r * p / (r + p + 1e-10)
    return f

def macro_f1(y_hat,y):
    all_labels = set(y)
    y_hat = np.array(y_hat)
    f1s = {label:f1(y_hat,y,label) for label in all_labels}
    return sum(f1s.values())/len(all_labels),f1s


# deliverable 4.1
def make_numpy(bags_of_words, vocab):
    '''
    Convert the bags of words into a 2D numpy array

    :param bags_of_words: list of Counters
    :param vocab: pruned vocabulary
    :returns: the bags of words as a 2D numpy array (length of bags_of_words by length of vocab)
    :rtype: numpy array
    '''
    Two_array = np.zeros((len(bags_of_words), len(vocab)))
    vocab = sorted(vocab)

    for i in range(len(bags_of_words)):
        for j in range(len(vocab)):
            if (vocab[j] in bags_of_words[i]):
                Two_array[i][j] = bags_of_words[i][vocab[j]]

    return Two_array

# deliverable 4.2
def better_model():
    # BEGIN SOLUTION
    scikit_log_reg = LogisticRegression(solver='lbfgs')   ## Tune parameters for this function.
    scikit_log_reg.C = 0.08
    
    return scikit_log_reg
    ### END SOLUTION

######################### helper code
def train_model(loss, model, X_tr_var, Y_tr_var,
                num_its = 200,
                X_dv_var = None,
                Y_dv_var = None,
                status_frequency=10,
                optim_args = {'lr':0.002,'momentum':0},
                param_file = 'best.params'):

    # initialize optimizer
    optimizer = optim.SGD(model.parameters(), **optim_args)

    losses = []
    accuracies = []

    for epoch in range(num_its):
        # set gradient to zero
        optimizer.zero_grad()
        # run model forward to produce loss
        output = loss.forward(model.forward(X_tr_var),Y_tr_var)
        # backpropagate and train
        output.backward()
        optimizer.step()

        #print(output.item())
        losses.append(output.item())

        # write parameters if this is the best epoch yet
        if X_dv_var is not None:
            # run forward on dev data
            _, Y_hat = model.forward(X_dv_var).max(dim=1)
            # compute dev accuracy
            acc = acc(Y_hat.data.numpy(),Y_dv_var.data.numpy())
            # save
            if len(accuracies) == 0 or acc > max(accuracies):
                state = {'state_dict':model.state_dict(),
                         'epoch':len(accuracies)+1,
                         'accuracy':acc}
                torch.save(state,param_file)
            accuracies.append(acc)

        # print status message if desired
        if status_frequency > 0 and epoch % status_frequency == 0:
            print("Epoch "+str(epoch+1)+": Dev Accuracy: "+str(acc))

    # load parameters of best model
    checkpoint = torch.load(param_file)
    model.load_state_dict(checkpoint['state_dict'])

    return model, losses, accuracies



def plot_results(losses, accuracies):
    fig,ax = plt.subplots(1,2,figsize=[12,2])
    ax[0].plot(losses)
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('iteration');
    ax[1].plot(accuracies);
    ax[1].set_ylabel('dev set accuracy')
    ax[1].set_xlabel('iteration');

# deliverable 5.1
def get_top_features_LR(scikit_log_reg, vocab,label_set,label,k):
    most_indicative_features = []
    least_indicative_features = []

    ### BEGIN SOLUTION
    vocab = sorted(vocab)
    cof = scikit_log_reg.coef_
    ind = None

    for i in range(len(label_set)):
        if label == label_set[i]:
            ind = i
            break

    f = [j  for i, j in sorted(zip(cof[ind], vocab))]

    most_indicative_features = f[-k: ]
    least_indicative_features = f[0 : k]

    return most_indicative_features, least_indicative_features
    ### END SOLUTION

# deliverable 5.2
def get_top_features_NB(theta_nb, label_set,label,k):
    most_indicative_features = []
    least_indicative_features = []

    ### BEGIN SOLUTION
    label_set = defaultdict()

    for i, j in theta_nb.keys():
        if (label == i) and (j != OFFSET):
            label_set[j] = theta_nb[(i, j)]

    f = [i  for i, j in sorted(label_set.items(), key = lambda item: item[1])]
    
    most_indicative_features = f[-k: ]
    least_indicative_features = f[0 : k]

    return most_indicative_features, least_indicative_features
    ### END SOLUTION

# deliverable 6
def get_PRF(Y_hat_dv, Y_dv, label_set, label):
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    ### BEGIN SOLUTION
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    ind = None

    for i in range(len(label_set)):
        if (label == label_set[i]):
            ind = i
            break

    for j in range(len(Y_hat_dv)):
        if (Y_dv[j] != ind) and (Y_hat_dv[j] == ind):
            FP += 1
        elif (Y_dv[j] != ind) and (Y_hat_dv[j] != ind):
            TN += 1
        elif (Y_dv[j] == ind) and (Y_hat_dv[j] == ind) :
            TP += 1
        else:
            FN += 1

    Precision = TP/(TP + FP)
    Recall = TP/(TP + FN)
    F_value = 2*(Precision*Recall)/(Precision + Recall)
    return Precision, Recall, F_value
    ### END SOLUTION
