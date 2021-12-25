import heapq
import pandas as pd
import statsmodels.api as sm 
import numpy as np
import statsmodels

class Lab3(object):
    
    def create_data(self,snp_lines) :
        '''
        Input - the snp_lines parsed at the beginning of the notebook
        Output - You should return the 53 x 3902 dataframe
        '''
        #start code here
        column = []
        table = []
        samples = np.zeros((53, 3902))

        for i in snp_lines:
            table.append(i.split())  

        for m in range(3902):
            for n in range(53):
                if table[m][n+9] == './.':
                    samples[n][m] = np.nan
                else:
                    samples[n][m] = int(table[m][n+9][0]) + int(table[m][n+9][2])
                    
        dataframe = pd.DataFrame(samples)
        
        for j in table:
            column.append(str(j[0])+':'+str(j[1]))

        dataframe.columns = column
        
        return dataframe
        #end code here

    def create_target(self,header_line) :
        '''
        Input - the header_line parsed at the beginning of the notebook
        Output - a list of values(either 0 or 1)
        '''
        #start code here
        labels = header_line.split()
        indicator = []

        for i in labels[9:]:
            if 'yellow' in i:
                indicator.append(1)
            elif 'dark' in i:
                indicator.append(0)
        
        return indicator
        #end code here
    
    def logistic_reg_per_snp(self,df) :
        '''
        Input - snp_data dataframe
        Output - list of pvalues and list of betavalues
        '''
        #start code here
        p_value = []
        betavalue = []
        
        for i in df.columns[:3902]:
            data = sm.add_constant(list(df[i]))
            statsmodel = sm.Logit(list(df['target']),data,missing='drop').fit(method='bfgs',disp=False)
            
            p_value.append(round(statsmodel.pvalues[1],9))
            betavalue.append(round(statsmodel.params[1],5))

        return p_value, betavalue
        #end code here
    
    
    def get_top_snps(self,snp_data,p_values) :
        '''
        Input - snp dataframe with target column and p_values calculated previously
        Output - list of 5 tuples, each with chromosome and position
        '''
        #start code here
        smallest = heapq.nsmallest(5, p_values)
        output = []

        for i in smallest:
            temp = snp_data.columns[p_values.index(i)].split(':')
            output.append((temp[0],temp[1]))

        return output
        #end code here