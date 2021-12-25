import numpy as np
import time
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# You may use this function as you like.
error = lambda y, yhat: np.mean(y!=yhat)

class Question1(object):
    # The sequence in this problem is different from the one you saw in the jupyter notebook. This makes it easier to grade. Apologies for any inconvenience.
    def BernoulliNB_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a BernoulliNB classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below

        classifier = BernoulliNB()
        t1 = time.time()
        classifier.fit(traindata, trainlabels)
        t2 = time.time()
        pred = classifier.predict(valdata)
        t3 = time.time()

        fittingTime = t2 - t1
        valPredictingTime = t3 - t2
        validationError = error(pred, vallabels)

        pred = classifier.predict(traindata)
        trainingError = error(pred,trainlabels)


        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def MultinomialNB_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a MultinomialNB classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below

        classifier = MultinomialNB()
        t1 = time.time()
        classifier.fit(traindata, trainlabels)
        t2 = time.time()
        pred = classifier.predict(valdata)
        t3 = time.time()

        fittingTime = t2 - t1
        valPredictingTime = t3 - t2
        validationError = error(pred, vallabels)

        pred = classifier.predict(traindata)
        trainingError = error(pred, trainlabels)

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def LinearSVC_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a LinearSVC classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below

        classifier = LinearSVC()
        t1 = time.time()
        classifier.fit(traindata, trainlabels)
        t2 = time.time()
        pred = classifier.predict(valdata)
        t3 = time.time()

        fittingTime = t2 - t1
        valPredictingTime = t3 - t2
        validationError = error(pred, vallabels)

        pred = classifier.predict(traindata)
        trainingError = error(pred, trainlabels)

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def LogisticRegression_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a LogisticRegression classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below

        classifier = LogisticRegression()
        t1 = time.time()
        classifier.fit(traindata, trainlabels)
        t2 = time.time()
        pred = classifier.predict(valdata)
        t3 = time.time()

        fittingTime = t2 - t1
        valPredictingTime = t3 - t2
        validationError = error(pred, vallabels)

        pred = classifier.predict(traindata)
        trainingError = error(pred, trainlabels)

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def NN_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a Nearest Neighbor classifier using the given data.

        Make sure to modify the default parameter.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata              (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels            (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below

        classifier = KNeighborsClassifier(1)
        t1 = time.time()
        classifier.fit(traindata, trainlabels)
        t2 = time.time()
        pred = classifier.predict(valdata)
        t3 = time.time()

        fittingTime = t2 - t1
        valPredictingTime = t3 - t2
        validationError = error(pred, vallabels)

        pred = classifier.predict(traindata)
        trainingError = error(pred, trainlabels)

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def confMatrix(self,truelabels,estimatedlabels):
        """ Write a function that calculates the confusion matrix (cf. Fig. 2.1 in the notes).

        You may wish to read Section 2.1.1 in the notes -- it may be helpful, but is not necessary to complete this problem.

        Parameters:
        1. truelabels           (Nv, ) numpy ndarray. The ground truth labels.
        2. estimatedlabels      (Nv, ) numpy ndarray. The estimated labels from the output of some classifier.

        Outputs:
        1. cm                   (2,2) numpy ndarray. The calculated confusion matrix.
        """
        cm = np.zeros((2,2))
        # Put your code below
        TP = 0
        TN = 0
        FN = 0
        FP = 0

        for i in range(len(truelabels)):
            if truelabels[i] == 1 and estimatedlabels[i] == 1:
                TP += 1
            elif truelabels[i] == 1 and estimatedlabels[i] == -1:
                FN += 1
            elif truelabels[i] == -1 and estimatedlabels[i] == 1:
                FP += 1
            elif truelabels[i] == -1 and estimatedlabels[i] == -1:
                TN += 1

        cm[0][0] = TP
        cm[0][1] = FP
        cm[1][0] = FN
        cm[1][1] = TN

        return cm

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Run the classifier you selected in the previous part of the problem on the test data.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. testError            Float. The reported test error. It should be less than 1.
        3. confusionMatrix      (2,2) numpy ndarray. The resulting confusion matrix. This will not be graded.
        """
        # You can freely use the following line

        # Put your code below
        classifier, trainingError, testError, fittingTime, valPredictingTime = self.LogisticRegression_classifier(traindata, trainlabels, testdata, testlabels)
        pred = classifier.predict(testdata)
        confusionMatrix = self.confMatrix(testlabels, pred)
        # Do not change this sequence!
        return (classifier, testError, confusionMatrix)

class Question2(object):
    def crossValidationkNN(self, traindata, trainlabels, k):
        """ Write a function which implements 5-fold cross-validation to estimate the error of a classifier with cross-validation with the 0,1-loss for k-Nearest Neighbors (kNN).

        For this problem, take your folds to be 0:N/5, N/5:2N/5, ..., 4N/5:N for cross-validation.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. k                    Integer. The cross-validated error estimates will be outputted for 1,...,k.

        Outputs:
        1. err                  (k+1,) numpy ndarray. err[i] is the cross-validated estimate of using i neighbors (the zero-th component of the vector will be meaningless).
        """
        # Put your code below
        err = np.zeros(k+1)
        err[0] = 99999
        length = len(traindata)/5

        for i in range(1,k+1):
            temp_err = np.zeros(5)

            classifier = KNeighborsClassifier(i)
            classifier.fit(traindata[ int(1*length) : int(5*length)], trainlabels[ int(1*length) : int(5*length)])
            pred = classifier.predict(traindata[int(0*length):int(1*length)])
            temp_err[0] = error(pred,trainlabels[int(0*length):int(1*length)])

            temp = []
            for j in range(0,int(1*length)):
                temp.append(j)
            for k in range(int(2*length),int(5*length)):
                temp.append(k)

            classifier = KNeighborsClassifier(i)
            classifier.fit(traindata[temp], trainlabels[temp])
            pred = classifier.predict(traindata[int(1 * length):int(2 * length)])
            temp_err[1] = error(pred, trainlabels[int(1 * length):int(2 * length)])

            temp = []
            for j in range(0,int(2*length)):
                temp.append(j)
            for k in range(int(3*length),int(5*length)):
                temp.append(k)
            classifier = KNeighborsClassifier(i)
            classifier.fit(traindata[temp], trainlabels[temp])
            pred = classifier.predict(traindata[int(2 * length):int(3 * length)])
            temp_err[2] = error(pred, trainlabels[int(2 * length):int(3 * length)])

            temp = []
            for j in range(0,int(3*length)):
                temp.append(j)
            for k in range(int(4*length),int(5*length)):
                temp.append(k)
            classifier = KNeighborsClassifier(i)
            classifier.fit(traindata[temp], trainlabels[temp])
            pred = classifier.predict(traindata[int(3 * length):int(4 * length)])
            temp_err[3] = error(pred, trainlabels[int(3 * length):int(4 * length)])


            classifier = KNeighborsClassifier(i)
            classifier.fit(traindata[ int(0*length) : int(4*length)], trainlabels[ int(0*length) : int(4*length)])
            pred = classifier.predict(traindata[int(4*length):int(5*length)])
            temp_err[4] = error(pred,trainlabels[int(4*length):int(5*length)])

            err[i] = np.mean(temp_err)

        return err

    def minimizer_K(self, traindata, trainlabels, k):
        """ Write a function that calls the above function and returns 1) the output from the previous function, 2) the number of neighbors within  1,...,k  that minimizes the cross-validation error, and 3) the correponding minimum error.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. k                    Integer. The cross-validated error estimates will be outputted for 1,...,k.

        Outputs:
        1. err                  (k+1,) numpy ndarray. The output from crossValidationkNN().
        2. k_min                Integer (np.int64 or int). The number of neighbors within  1,...,k  that minimizes the cross-validation error.
        3. err_min              Float. The correponding minimum error.
        """
        err = self.crossValidationkNN(traindata, trainlabels, k)
        # Put your code below
        err_min = np.min(err)
        k_min = np.where(err == err_min)[0][0]

        # Do not change this sequence!
        return (err, k_min, err_min)

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Train a kNN model on the whole training data using the number of neighbors you found in the previous part of the question, and apply it to the test data.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data. Use the best k value that you choose.
        2. testError            Float. The reported test error. It should be less than 1.
        """
        # Put your code below
        err, k_min, err_min = self.minimizer_K(traindata, trainlabels, 30)

        classifier = KNeighborsClassifier(k_min)
        classifier.fit(traindata, trainlabels)
        pred = classifier.predict(testdata)

        testError = error(pred, testlabels)

        # Do not change this sequence!
        return (classifier, testError)

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class Question3(object):
    def LinearSVC_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-5},...,2^{15}.

        You should seaerch by hand.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. min_err              Float. The correponding minimum error.
        """
        # Put your code below

        temp_err = np.zeros([21])
        for i in range(-5, 16):
            classifier = LinearSVC(C=2**i)
            scores = cross_val_score(classifier, traindata, trainlabels)
            err = 1 - np.mean(scores)
            temp_err[i+5] = err

        min_err = np.min(temp_err)
        C_min = float(2)**(np.where(temp_err == min_err)[0][0] - 5)

        #Following lines are for testing and comparing, these lines are
        #copied from LogisticRegression_crossValidation part below

        # param_grid = {'C': [2 ** i for i in range(-5, 16)]}
        #
        # classifier = LinearSVC()
        # grid_search = GridSearchCV(classifier, param_grid)
        # grid_search.fit(traindata, trainlabels)
        #
        # var = grid_search.best_params_
        # C_min = var['C']
        #
        # classifier = LinearSVC(C=C_min)
        # scores = cross_val_score(classifier, traindata, trainlabels)
        #
        # min_err = 1 - np.mean(scores)
        # Do not change this sequence!
        return (C_min, min_err)

    def SVC_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-5},...,2^{15} and \gamma from 2^{-15},...,2^{3}.

        Use GridSearchCV to perform a grid search.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. gamma_min            Float. The hyper-parameter \gamma that minimizes the validation error.
        3. min_err              Float. The correponding minimum error.
        """
        # Put your code below

        param_grid =  {'C': [2**i for i in range(-5,16)], 'gamma': [2**i for i in range(-15,4)]}

        classifier = SVC()
        grid_search = GridSearchCV(classifier, param_grid)
        grid_search.fit(traindata, trainlabels)

        var = grid_search.best_params_
        C_min = var['C']
        gamma_min = var['gamma']

        classifier = SVC(C=C_min,gamma=gamma_min)
        scores = cross_val_score(classifier, traindata, trainlabels)

        min_err = 1 - np.mean(scores)

        # Do not change this sequence!
        return (C_min, gamma_min, min_err)

    def LogisticRegression_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-14},...,2^{14}.

        You may either use GridSearchCV or search by hand.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. min_err              Float. The correponding minimum error.
        """
        # Put your code below
        C_min = float("nan")
        maxi = 0

        for i in [2**j for j in range(-14, 15)]:
            classifier = LogisticRegression(max_iter=1000, C=i)
            score = cross_val_score(classifier, traindata, trainlabels, cv=10)
            rate = sum(score)/len(score)
            if maxi < rate:
                maxi = rate
                C_min = i

        min_err = 1 - maxi

        # Do not change this sequence!
        return (C_min, min_err)

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Train the best classifier selected above on the whole training set.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data. Use the best classifier that you choose.
        2. testError            Float. The reported test error. It should be less than 1.
        """
        # Put your code below
        C_min, gamma_min, min_err = self.SVC_crossValidation(traindata, trainlabels)
        classifier = SVC(C=C_min,gamma=gamma_min)
        classifier.fit(traindata, trainlabels)
        pred = classifier.predict(testdata)

        testError = error(pred,testlabels)

        # Do not change this sequence!
        return (classifier, testError)
