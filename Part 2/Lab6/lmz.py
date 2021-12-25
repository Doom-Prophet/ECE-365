import numpy as np
from collections import OrderedDict

class Lab1(object):
    def parse_reads_illumina(self,reads) :
        '''
        Input - Illumina reads file as a string
        Output - list of DNA reads
        '''
        #start code here
        reads_illumina = reads.split('\n')
        dna_reads_illumina = []
        n = len(reads_illumina)//4
        for i in range(n):
            dna_reads_illumina.append(reads_illumina[4*i+1])
        return dna_reads_illumina
        #end code here

    def unique_lengths(self,dna_reads) :
        '''
        Input - list of dna reads
        Output - set of counts of reads
        '''
        #start code here
        counts_illumina = set()
        for i in dna_reads:
            counts_illumina.add(len(i))
        return counts_illumina
        #end code here

    def check_impurity(self,dna_reads) :
        '''
        Input - list of dna reads
        Output - list of reads which have impurities, a set of impure chars 
        '''
        #start code here
        impure_reads_illumina = []
        impure_chars_illumina = set()
        for i in dna_reads:
            set_e = {e for e in i}
            for j in set_e:
                if j not in ['A', 'C', 'G', 'T', 'a', 'c', 'g', 't']:
                    impure_reads_illumina.append(i)
                    impure_chars_illumina.add(j)
        return list(impure_reads_illumina), impure_chars_illumina
        #end code here

    def get_read_counts(self,dna_reads) :
        '''
        Input - list of dna reads
        Output - dictionary with key as read and value as the no. of times it occurs
        '''
        #start code here
        reads_counts_illumina = {}
        for i in dna_reads:
            if i not in reads_counts_illumina.keys():
                reads_counts_illumina[i] = 1
            else:
                reads_counts_illumina[i] += 1
        return reads_counts_illumina
        #end code here

    def parse_reads_pac(self,reads_pac) :
        '''
        Input - pac bio reads file as a string
        Output - list of dna reads
        '''
        #start code here
        dna_reads_pac = []
        reads_pac = reads_pac.split('\n')
        new_read = ''
        for i in reads_pac:
            if '>' in i:
                if new_read:
                    dna_reads_pac.append(new_read)
                    new_read = ''
            else:
                new_read += i
        dna_reads_pac.append(new_read)
        return dna_reads_pac
        #end code here