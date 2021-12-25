import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
class Lab4(object):
    
    def expectation_maximization(self,read_mapping,tr_lengths,n_iterations) :
         #start code here
        len_tr = len(tr_lengths)
        len_readmap = len(read_mapping)
        record = [[] for i in range(len_tr)]
        P = [1 / len_tr] * len_tr
        Z = np.zeros((len_readmap, len_tr))

        for i in range(len_tr):
            record[i].append(P[i])

        for count in range(n_iterations):
            # E-step
            for i in range(len_readmap):
                summation = 0     # sum of p and j
                for j in read_mapping[i]:
                    summation += P[j]
                for k in range(len_tr):
                    if k in read_mapping[i]:
                        Z[i][k] = P[k] / summation
                    else:
                        Z[i][k] = 0
                        
            # M-step
            theta = [0] * len_tr
            
            for i in range(len_tr):
                for j in range(len_readmap):
                    theta[i] += Z[j][i]
                theta[i] = (theta[i] / len_readmap) / tr_lengths[i]
            
            sum_theta = sum(theta)   
            
            for k in range(len_tr):
                P[k] = theta[k] / sum_theta

            for i in range(len_tr):
                record[i].append(P[i])
                
        return record
        #end code here

    def prepare_data(self,lines_genes) :
        '''
        Input - list of strings where each string corresponds to expression levels of a gene across 3005 cells
        Output - gene expression dataframe
        '''
        #start code here
        len_genes = len(lines_genes)
        names = ['Gene_%d'%i for i in range(len_genes)]
        data = pd.DataFrame()

        for i in range(len_genes):
            read = lines_genes[i].split()
            for j in range(len(read)):
                read[j] = round(np.log(1 + int(read[j])), 5)
            data[names[i]] = read

        return data
        #end code here
    
    def identify_less_expressive_genes(self,df) :
        '''
        Input - gene expression dataframe
        Output - list of column names which are expressed in less than 25 cells
        '''
        #start code here
        result = []
        columns = df.columns.values.tolist()

        for i in columns:
            count = 0
            for j in df[i]:
                if j:
                    count += 1
            if count < 25:
                result.append(i)

        return result
        #end code here
    
    
    def perform_pca(self,df) :
        '''
        Input - df_new
        Output - numpy array containing the top 50 principal components of the data.
        '''
        #start code here
        PCA_data = PCA(n_components = 50, random_state = 365).fit_transform(np.array(df))
        length = PCA_data.shape[0]
        width = PCA_data.shape[1]

        for i in range(length):
            for j in range(width):
                PCA_data[i][j] = round(PCA_data[i][j], 5)

        return PCA_data
        #end code here
    
    def perform_tsne(self,pca_data) :
        '''
        Input - pca_data
        Output - numpy array containing the top 2 tsne components of the data.
        '''
        #start code here
        TSNE_data = TSNE(n_components = 2, random_state = 1000, perplexity = 50).fit_transform(pca_data)
        
        return TSNE_data
        #end code here