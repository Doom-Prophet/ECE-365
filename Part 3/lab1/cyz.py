from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from nltk.corpus import udhr

def get_freqs(corpus, puncts):
    freqs = {}
    ### BEGIN SOLUTION
    chars = [e for e in corpus] 
    for i in range(len(chars)):
            if chars[i] in puncts or chars[i].isdigit():
                chars[i] = ' '
    words = ''.join(chars).lower().split()
    for word in words:
        if not word in freqs:
            freqs[word] = 1
        else:
            freqs[word] += 1
    ### END SOLUTION
    return freqs

def get_top_10(freqs):
    top_10 = []
    ### BEGIN SOLUTION
    top_10 = [e[0] for e in sorted(freqs.items(), key = lambda x: x[1])[-10:]][::-1]
    ### END SOLUTION
    return top_10

def get_bottom_10(freqs):
    bottom_10 = []
    ### BEGIN SOLUTION
    bottom_10 = [e[0] for e in sorted(freqs.items(), key = lambda x: x[1])[:10]]
    ### END SOLUTION
    return bottom_10

def get_percentage_singletons(freqs):
    ### BEGIN SOLUTION
    cnt = 0
    for key in freqs:
        if freqs[key] == 1:
           cnt += 1
    return cnt / len(freqs) * 100
    ### END SOLUTION
    pass

def get_freqs_stemming(corpus, puncts):
    ### BEGIN SOLUTION
    porter = PorterStemmer()
    freqs = {}
    chars = [e for e in corpus] 
    for i in range(len(chars)):
            if chars[i] in puncts or chars[i].isdigit():
                chars[i] = ' '
    words = ''.join(chars).lower().split()
    for word in words:
        word = porter.stem(word)
        if not word in freqs:
            freqs[word] = 1
        else:
            freqs[word] += 1
    return freqs
    ### END SOLUTION
    pass

def get_freqs_lemmatized(corpus, puncts):
    ### BEGIN SOLUTION
    wordnet_lemmatizer = WordNetLemmatizer()
    freqs = {}
    chars = [e for e in corpus] 
    for i in range(len(chars)):
            if chars[i] in puncts or chars[i].isdigit():
                chars[i] = ' '
    words = ''.join(chars).lower().split()
    for word in words:
        word = wordnet_lemmatizer.lemmatize(word, pos="v")
        if not word in freqs:
            freqs[word] = 1
        else:
            freqs[word] += 1
    return freqs
    ### END SOLUTION
    pass

def size_of_raw_corpus(freqs):
    ### BEGIN SOLUTION
    return len(freqs)
    ### END SOLUTION
    pass

def size_of_stemmed_raw_corpus(freqs_stemming):
    ### BEGIN SOLUTION
    return len(freqs_stemming)
    ### END SOLUTION
    pass

def size_of_lemmatized_raw_corpus(freqs_lemmatized):
    ### BEGIN SOLUTION
    return len(freqs_lemmatized)
    ### END SOLUTION
    pass

def percentage_of_unseen_vocab(a, b, length_i):
    ### BEGIN SOLUTION
    return len(set(a) - set(b)) / length_i
    ### END SOLUTION
    pass

def frac_80_perc(freqs):
    ### BEGIN SOLUTION
    items = sorted(freqs.items(), key = lambda x: x[1])[::-1]
    total = sum([e[1] for e in items])
    cnt = 0
    res = 0
    while cnt < total * 0.8:
        cnt += items[res][1]
        res += 1
    return res / len(freqs)
    ### END SOLUTION
    pass

def plot_zipf(freqs):
    ### BEGIN SOLUTION
    items = sorted(freqs.items(), key = lambda x: x[1])[::-1]
    x = [i + 1 for i in range(len(items))]
    y = [e[1] for e in items]
    plt.plot(x, y)
    ### END SOLUTION
    plt.show()  # put this line at the end to display the figure.

def get_TTRs(languages):
    TTRs = {}
    for lang in languages:
        words = udhr.words(lang)
        words = [e.lower() for e in words]
        TTRs[lang] = [None] * 13
        temp_set = set()
        ### BEGIN SOLUTION
        for i in range(13):
            temp_set = temp_set.union(set(words[100 * i: 100 * (i + 1)])) 
            TTRs[lang][i] = len(temp_set)
        ### END SOLUTION
    return TTRs

def plot_TTRs(TTRs):
    ### BEGIN SOLUTION
    x = [100 * (i + 1) for i in range(13)]
    for key in TTRs:
        plt.plot(x, TTRs[key], label = key)
    plt.legend()
    ### END SOLUTION
    plt.show()  # put this line at the end to display the figure.
