from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from nltk.corpus import udhr

def get_freqs(corpus, puncts):
    freqs = {}
    ### BEGIN SOLUTION
    words = [x for x in corpus] 
    for i in range(len(words)):
        if (words[i] in puncts) or (words[i].isdigit()):
            words[i] = ' '

    words = ''.join(words).lower().split()

    for j in words:
        if not j in freqs:
            freqs[j] = 1
        else:
            freqs[j] += 1
    ### END SOLUTION
    return freqs

def get_top_10(freqs):
    top_10 = []
    ### BEGIN SOLUTION
    temp = sorted(freqs.items(),key = lambda x:x[1],reverse = True)
    temp = temp[:10]
    for l in temp:
        top_10.append(l[0])
    ### END SOLUTION
    return top_10

def get_bottom_10(freqs):
    bottom_10 = []
    ### BEGIN SOLUTION
    temp = sorted(freqs.items(),key = lambda x:x[1],reverse = False)
    temp = temp[:10]
    for l in temp:
        bottom_10.append(l[0])
    ### END SOLUTION
    return bottom_10

def get_percentage_singletons(freqs):
    ### BEGIN SOLUTION
    count = 0
    for i in freqs.values():
        if i == 1:
            count += 1
    
    return 100*count/len(freqs.values())
    ### END SOLUTION

def get_freqs_stemming(corpus, puncts):
    ### BEGIN SOLUTION
    freqs = {}

    words = [x for x in corpus] 
    for i in range(len(words)):
        if (words[i] in puncts) or (words[i].isdigit()):
            words[i] = ' '

    words = ''.join(words).lower().split()
    porter = PorterStemmer()

    for j in words:
        j = porter.stem(j)
        if not j in freqs:
            freqs[j] = 1
        else:
            freqs[j] += 1
    ### END SOLUTION
    return freqs

def get_freqs_lemmatized(corpus, puncts):
    ### BEGIN SOLUTION
    freqs = {}

    words = [x for x in corpus] 
    for i in range(len(words)):
        if (words[i] in puncts) or (words[i].isdigit()):
            words[i] = ' '

    words = ''.join(words).lower().split()
    wordnet_lemmatizer = WordNetLemmatizer()

    for j in words:
        j = wordnet_lemmatizer.lemmatize(j, pos="v")
        if not j in freqs:
            freqs[j] = 1
        else:
            freqs[j] += 1
    ### END SOLUTION
    return freqs

def size_of_raw_corpus(freqs):
    ### BEGIN SOLUTION
    return len(freqs.keys())
    ### END SOLUTION

def size_of_stemmed_raw_corpus(freqs_stemming):
    ### BEGIN SOLUTION
    return len(freqs_stemming.keys())
    ### END SOLUTION

def size_of_lemmatized_raw_corpus(freqs_lemmatized):
    ### BEGIN SOLUTION
    return len(freqs_lemmatized.keys())
    ### END SOLUTION

def percentage_of_unseen_vocab(a, b, length_i):
    ### BEGIN SOLUTION
    return len(set(a) - set(b))/length_i
    ### END SOLUTION

def frac_80_perc(freqs):
    ### BEGIN SOLUTION
    temp = sorted(freqs.items(),key = lambda x:x[1],reverse = True)
    total = sum([x[1] for x in temp])

    value_count = 0
    key_count = 0
    index = 0
    while value_count < total * 0.8:
        value_count += temp[index][1]
        key_count += 1
        index += 1
        if value_count/total > 0.8:
            break
    
    return key_count/len(freqs)
    ### END SOLUTION

def plot_zipf(freqs):
    ### BEGIN SOLUTION
    items = sorted(freqs.items(), key = lambda x: x[1])[::-1]
    x = [i + 1 for i in range(len(items))]
    y = [e[1] for e in items]
    
    plt.figure()
    plt.title("Zipf's law on words")
    plt.plot(x, y)
    plt.xlabel('Rank of words')
    plt.ylabel('Frequency of word')
    ### END SOLUTION
    plt.show()  # put this line at the end to display the figure.

def get_TTRs(languages):
    TTRs = {}
    for lang in languages:
        words = udhr.words(lang)
        new_words = [x.lower() for x in words]
        TTRs[lang] = []
        for i in range(13):
            new_set = set(new_words[:(i+1)*100])
            TTRs[lang].append(len(new_set))
        ### BEGIN SOLUTION
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
