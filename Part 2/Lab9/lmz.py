import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
class Lab4(object):
    
    def expectation_maximization(self,read_mapping,tr_lengths,n_iterations) :
        #start code here
        K = len(tr_lengths)
        N = len(read_mapping)
        history = [[] for k in range(K)]
        
        # Initialize
        p_hat = [1/K] * K
        for k in range(K):
            history[k].append(p_hat[k])
        z_hat = np.zeros((N, K))
        
        # Iteration
        for iteration in range(n_iterations):
            # E-step
            for i in range(N):
                sum_pj = 0
                for j in read_mapping[i]:
                    sum_pj += p_hat[j]
                for k in range(K):
                    if k in read_mapping[i]:
                        z_hat[i][k] = p_hat[k]/sum_pj
                    else:
                        z_hat[i][k] = 0
                        
            # M-step
            theta = [0] * K
            for k in range(K):
                for i in range(N):
                    theta[k] += z_hat[i][k]
                theta[k] = theta[k]/N/tr_lengths[k]
            sum_theta = sum(theta)   
            for k in range(K):
                p_hat[k] = theta[k]/sum_theta
                
            # record p_hat
            for k in range(K):
                history[k].append(p_hat[k])
                
        return history
        #end code here

    def prepare_data(self,lines_genes) :
        '''
        Input - list of strings where each string corresponds to expression levels of a gene across 3005 cells
        Output - gene expression dataframe
        '''
        #start code here
        num_gene = len(lines_genes)
        gene_names = ['Gene_%d'%i for i in range(num_gene)]
        data_df = pd.DataFrame()
        for i in range(num_gene):
            line = lines_genes[i].split()
            for k in range(len(line)):
                line[k] = round(np.log(int(line[k])+1), 5)
            data_df[gene_names[i]] = line
        return data_df
        #end code here
    
    def identify_less_expressive_genes(self,df) :
        '''
        Input - gene expression dataframe
        Output - list of column names which are expressed in less than 25 cells
        '''
        #start code here
        drop_columns = []
        columns = df.columns.values.tolist()
        for name in columns:
            expr_count = 0
            for i in df[name]:
                if i:
                    expr_count += 1
            if expr_count < 25:
                drop_columns.append(name)
        return drop_columns
        #end code here
    
    
    def perform_pca(self,df) :
        '''
        Input - df_new
        Output - numpy array containing the top 50 principal components of the data.
        '''
        #start code here
        pca_data = PCA(n_components=50, random_state=365).fit_transform(np.array(df))
        for i in range(pca_data.shape[0]):
            for j in range(pca_data.shape[1]):
                pca_data[i][j] = round(pca_data[i][j], 5)
        return pca_data
        #end code here
    
    def perform_tsne(self,pca_data) :
        '''
        Input - pca_data
        Output - numpy array containing the top 2 tsne components of the data.
        '''
        #start code here
        tsne_data50 = TSNE(n_components=2, random_state=1000, perplexity=50).fit_transform(pca_data)
        return tsne_data50
        #end code here