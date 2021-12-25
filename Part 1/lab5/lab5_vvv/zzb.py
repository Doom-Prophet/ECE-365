import numpy as np
from sklearn import neighbors
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.cluster import KMeans
import scipy.spatial.distance as dist
from matplotlib.colors import ListedColormap

class Question1(object):
    def pcaeig(self,data):
        """ Implement PCA via the eigendecomposition.
       
        Parameters:
        1. data     (N, d) numpy ndarray. Each row as a feature vector.

        Outputs:
        1. W        (d,d) numpy array. PCA transformation matrix (Note that each row of the matrix should be a principal component)
        2. s        (d,) numpy array. Vector consisting of the amount  of variance explained in the data by each PCA feature. 
        Note that the PCA features are ordered in decreasing amount of variance explained, by convention.
        """
        N = data.shape[0]
        # d = data.shape[1]
        sample_cov = np.dot(data.T, data)/N
        eig_val, eig_vec = np.linalg.eigh(sample_cov)
        W = np.fliplr(eig_vec).T
        s = np.flip(eig_val,2)
        # Remember to check your data types
        return (W,s)
    
    def pcadimreduce(self,data, W, k):
        """ Implements dimension reduction via PCA.
        
        Parameters:
        1. data     (N, d) numpy ndarray. Each row as a feature vector.
        2. W        (d,d) numpy array. PCA transformation matrix
        3. k        number. Number of PCA features to retain
        
        Outputs:
        1. reduced_data  (N,k) numpy ndarray, where each row contains PCA features corresponding to its input feature.
        """
        # print(data.shape)
        reduced_data = np.dot(data, W.T)[:,0:k]
        # print(reduced_data.shape)
        return reduced_data
    
    def pcareconstruct(self,pcadata, W, k):
        """ Implements dimension reduction via PCA.
        
        Parameters:
        1. pcadata     (N, k) numpy ndarray. Each row as a PCA vector. (e.g. generated from pcadimreduce)
        2. W        (d,d) numpy array. PCA transformation matrix
        3. k        number. Number of PCA features 
        
        Outputs:
        1. reconstructed_data  (N,d) numpy ndarray, where the i-th row contains the reconstruction of the original i-th input feature vector (in `data`) based on the PCA features contained in `pcadata`.
        """
        Wk = W[0:k, :]
        reconstructed_data = np.dot(pcadata, Wk)
        return reconstructed_data
    
    def pcasvd(self, data): 
        """Implements PCA via SVD.
        
        Parameters: 
        1. data     (N, d) numpy ndarray. Each row as a feature vector.
        
        Returns: 
        1. Wsvd     (d,d) numpy array. PCA transformation matrix (Note that each row of the matrix should be a principal component)
        2. ssvd       (d,) numpy array. Vector consisting of the amount  of variance explained in the data by each PCA feature. 
        Note that the PCA features are ordered in decreasing amount of variance explained, by convention.
        """
        u,s,v=np.linalg.svd(data)
        Wsvd=v
        ssvd=s**2/data.shape[0]
        return Wsvd, ssvd
        

class Question2(object):
    
    def unexp_var(self, X):
        """Returns an numpy array with the fraction of unexplained variance on X by retaining the first k principal components for k =1,...200.
        Parameters:
        1. X        The input image
        
        Returns:
        1. pca      The PCA object fit on X 
        2. unexpv   A (200,) numpy ndarray, where the i-th element contains the percentage of unexplained variance on X by retaining i+1 principal components
        """
        unexpv = np.zeros((200))
        pca = PCA(n_components=200)
        pca.fit(X)
        for k in range(1,201):
            unexpv[k-1] = 1-np.sum(pca.explained_variance_ratio_[0:k])
        return pca,unexpv
    
    def pca_approx(self, X_t, pca, i):
        """Returns an approimation of `X_t` using the the first `i`  principal components (learned from `X`).
        
        Parameters:
            1. X_t      The input image to be approximated
            2. pca      The PCA object to use for the transform
            3. i        Number of principal components to retain
            
        Returns: 
            1. recon_img    The reconstructed approximation of X_t using the first i principal components learned from X (As a sanity check it should be of size (1,4096))
        """
        feat_img = pca.transform(X_t.reshape(1,-1))
        num_feat = feat_img.shape[1]
        # print(num_feat)
        feat_img[0,i:] = np.zeros(shape=(1,num_feat-i))
        recon_img = pca.inverse_transform(feat_img)
        return recon_img
    
   

class Question3(object):
    
    def pca_classify(self, traindata,trainlabels, valdata, vallabels):
        """Returns validation errors using 1-NN on the PCA features using 1,2,...,256 PCA features, the minimum validation error, and number of PCA features used.
        
        Parameters: 
            traindata       
            trainlabels
            valdata
            valabels
            
        Returns:
            ve      numpy array of length 256 containing the validation errors using 1,...,256 features
            min_ve  minimum validation error
            pca_feat Number of PCA features to retain. Integer.
        """
        error = lambda y, yhat: np.mean(y != yhat)
        N_t,d = traindata.shape
        N_v = valdata.shape[0]
        pca = PCA(n_components=d)
        pca.fit(traindata)
        ve = np.zeros((256))
        for k in range(1,256+1):
            reduced_train = pca.transform(traindata)
            # print(reduced_train.shape)
            reduced_train[:,k:] = np.zeros(shape=(N_t,d-k))
            reduced_val = pca.transform(valdata)
            reduced_val[:,k:] = np.zeros(shape=(N_v,d-k))
            classifier = neighbors.KNeighborsClassifier(n_neighbors=1)
            classifier.fit(reduced_train, trainlabels)
            est_result = classifier.predict(reduced_val)
            ve[k-1] = error(vallabels, est_result)
        min_idx = np.argmin(ve)
        min_ve = ve[min_idx]
        pca_feat = min_idx+1
        # print(pca_feat)
        # print(type(pca_feat))
        pca_feat = int(pca_feat)
        return ve, min_ve, pca_feat
    
