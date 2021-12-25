import pandas as pd
import statsmodels.api as sm 
import numpy as np
import statsmodels
import heapq

class Lab3(object):
    
    def create_data(self,snp_lines) :
        '''
        Input - the snp_lines parsed at the beginning of the notebook
        Output - You should return the 53 x 3902 dataframe
        '''
        #start code here
        columnNames = []
        snp_table = []
        for i in snp_lines:
            snp_table.append(i.split())
        for i in snp_table:
            columnNames.append(str(i[0])+':'+str(i[1]))
        
        
        snp_array = np.zeros((53, 3902))
        for i in range(3902):
            for j in range(53):
                if snp_table[i][j+9]== '0/0':
                    snp_array[j][i] = 0
                elif snp_table[i][j+9]== '1/0':
                    snp_array[j][i] = 1
                elif snp_table[i][j+9]== '0/1':
                    snp_array[j][i] = 1
                elif snp_table[i][j+9]== '1/1':
                    snp_array[j][i] = 2
                else:
                    snp_array[j][i] = np.nan
        
        snp_data = pd.DataFrame(snp_array)
        snp_data.columns = columnNames
        return snp_data
        #end code here

    def create_target(self,header_line) :
        '''
        Input - the header_line parsed at the beginning of the notebook
        Output - a list of values(either 0 or 1)
        '''
        #start code here
        splited_line = header_line.split()
        target = []
        for i in splited_line[9:]:
            if 'yellow' in i:
                target.append(1)
            elif 'dark' in i:
                target.append(0)
        return target
        #end code here
    
    def logistic_reg_per_snp(self,df) :
        '''
        Input - snp_data dataframe
        Output - list of pvalues and list of betavalues
        '''
        #start code here
        p_values = []
        betavalues = []
        
        for i in df.columns[:3902]:
            datalist = list(df[i])
            x_data = sm.add_constant(datalist)
            model = sm.Logit(list(df['target']),x_data,missing='drop').fit(method='bfgs',disp=False)
            p_values.append(round(model.pvalues[1],9))
            betavalues.append(round(model.params[1],5))
        return p_values, betavalues
        #end code here
    
    
    def get_top_snps(self,snp_data,p_values) :
        '''
        Input - snp dataframe with target column and p_values calculated previously
        Output - list of 5 tuples, each with chromosome and position
        '''
        #start code here
        smallest5 = heapq.nsmallest(5, p_values)
        smallest_index = []
        label_list = []
        output_list = []
        for i in smallest5:
            smallest_index.append(p_values.index(i))
        for i in smallest_index:
            label_list.append(snp_data.columns[i])
        for i in label_list:
            i = i.split(':')
            output_list.append((i[0],i[1]))
        return output_list
        #end code here