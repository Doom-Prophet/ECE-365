import numpy as np
import scipy.spatial.distance as dist
from scipy import stats
from sklearn import neighbors
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Question1(object):
    def bayesClassifier(self,data,pi,means,cov):
        delta = np.log(pi) + np.dot(np.dot(data,np.linalg.inv(cov)),means.T) - 0.5*np.sum(np.dot(np.linalg.inv(cov),means.T)*means.T,axis = 0)
        labels = np.argmax(delta,axis = 1)
        return labels


    def classifierError(self,truelabels,estimatedlabels):
        error = np.count_nonzero(estimatedlabels - truelabels)/len(truelabels)
        return error


class Question2(object):
    def trainLDA(self,trainfeat,trainlabel):
        nlabels = int(trainlabel.max())+1 # Assuming all labels up to nlabels exist.
        pi = np.zeros(nlabels)            # Store your prior in here
        means = np.zeros((nlabels,trainfeat.shape[1]))            # Store the class means in here
        cov = np.zeros((trainfeat.shape[1],trainfeat.shape[1]))       # Store the covariance matrix in here
        # Put your code below

        for i in range(nlabels):
            index = np.where(trainlabel == i)
            pi[i] = len(index[0])/len(trainlabel)

            sum = 0
            for j in index[0]:
                sum += trainfeat[j]
            means[i] = sum/len(index[0])

        for m in range(nlabels):
            index = np.where(trainlabel == m)
            for n in index[0]:
                cov += np.outer((trainfeat[n] - means[m]),(trainfeat[n] - means[m]).T)
        cov = cov/(len(trainlabel) - nlabels)

        # Don't change the output!
        return pi,means,cov

    def estTrainingLabelsAndError(self,trainingdata,traininglabels):
        q1 = Question1()
        lpi, lmeans, lcov = self.trainLDA(trainingdata,traininglabels)
        esttrlabels = q1.bayesClassifier(trainingdata,lpi, lmeans, lcov)
        trerror = q1.classifierError(traininglabels,esttrlabels)
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)

        # Don't change the output!
        return (esttrlabels, trerror)

    def estValidationLabelsAndError(self,trainingdata,traininglabels,valdata,vallabels):
        q1 = Question1()
        lpi, lmeans, lcov = self.trainLDA(trainingdata, traininglabels)
        estvallabels = q1.bayesClassifier(valdata, lpi, lmeans, lcov)
        valerror = q1.classifierError(vallabels, estvallabels)
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)

        # Don't change the output!
        return (estvallabels, valerror)


class Question3(object):
    def kNN(self,trainfeat,trainlabel,testfeat, k):
        D = dist.cdist(testfeat,trainfeat, metric = 'euclidean')
        labels = np.zeros(shape = testfeat.shape[0])
        for i in range(D.shape[0]):
            ind = np.argpartition(D[i,:],k)[:k]
            temp = trainlabel[ind]
            labels[i] = stats.mode(temp)[0][0]
        return labels

    def kNN_errors(self,trainingdata, traininglabels, valdata, vallabels):
        q1 = Question1()  #classifierError(self,truelabels,estimatedlabels)
        trainingError = np.zeros(4)
        validationError = np.zeros(4)
        k_array = [1,3,4,5]

        for i in range(len(k_array)):
            # Please store the two error arrays in increasing order with k
            # This function should call your previous self.kNN() function.
            # Put your code below
            result_t = self.kNN(trainingdata,traininglabels,trainingdata,k_array[i])
            trainingError[i] = q1.classifierError(traininglabels,result_t)
            result_v = self.kNN(trainingdata, traininglabels, valdata, k_array[i])
            validationError[i] = q1.classifierError(vallabels,result_v)

        # Don't change the output!
        return (trainingError, validationError)

class Question4(object):
    def sklearn_kNN(self,traindata,trainlabels,valdata,vallabels):
        model = neighbors.KNeighborsClassifier(1)
        t1 = time.time()
        model.fit(traindata, trainlabels)
        t2 = time.time()
        result = model.predict(valdata)
        t3 = time.time()
        classifier, valerror, fitTime, predTime = (model, np.count_nonzero(result - vallabels)/len(vallabels), t2-t1 , t3-t2)

        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

    def sklearn_LDA(self,traindata,trainlabels,valdata,vallabels):
        model = LinearDiscriminantAnalysis()
        t4 = time.time()
        model.fit(traindata, trainlabels)
        t5 = time.time()
        LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                                   solver='svd', store_covariance=False, tol=0.0001)
        t6 = time.time()
        result = model.predict(valdata)
        t7 = time.time()
        classifier, valerror, fitTime, predTime = (model, np.count_nonzero(result - vallabels)/len(vallabels), t5-t4, t7-t6)

        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

###
