import numpy as np
from collections import OrderedDict

class Lab1(object):
    def parse_reads_illumina(self,reads) :
        '''
        Input - Illumina reads file as a string
        Output - list of DNA reads
        '''
        #start code here
        index = []
        dna = []
        for i in range(len(reads)):
            if reads[i] == '\n':
                index.append(i)

        for j in range(len(index)-1):
            if j % 4 == 0:
                dna.append(reads[ index[j]+1 : index[j+1] ])
        
        return dna
        
        #end code here

    def unique_lengths(self,dna_reads) :
        '''
        Input - list of dna reads
        Output - set of counts of reads
        '''
        #start code here
        length = set()
        for i in dna_reads:
            length.add(len(i))
        
        return length
        #end code here

    def check_impurity(self,dna_reads) :
        '''
        Input - list of dna reads
        Output - list of reads which have impurities, a set of impure chars 
        '''
        #start code here
        impurities = []
        additional = set()

        for i in dna_reads:
            words = {j for j in i}
            for k in words:
                if k not in ['A', 'C', 'G', 'T','a', 'c', 'g', 't']:
                    impurities.append(i)
                    additional.add(k)

        return impurities, additional
        #end code here

    def get_read_counts(self,dna_reads) :
        '''
        Input - list of dna reads
        Output - dictionary with key as read and value as the no. of times it occurs
        '''
        #start code here
        dic = {}

        for i in dna_reads:
            if i not in dic.keys():
                dic[i] = 1
            else:
                dic[i] += 1
        
        return dic
        #end code here

    def parse_reads_pac(self,reads_pac) :
        '''
        Input - pac bio reads file as a string
        
        Output - list of dna reads
        '''
        #start code here
        dna = []
        content = reads_pac.split('\n')
        buffer = ''

        for i in content:
            if '>' in i:
                if buffer:
                    dna.append(buffer)
                    buffer = ''
            else:
                buffer += i

        dna.append(buffer)
        return dna
        #end code here