import numpy as np
from collections import OrderedDict

class Lab2(object):
    
    def smith_waterman_alignment(self,s1,s2,penalties) :
        '''
        Input - two sequences and a dictionary with penalities for match, mismatch and gap
        Output - an integer value which is the maximum smith waterman alignment score
        '''
        #start code here
        len1 = len(s1)
        len2 = len(s2)

        table = np.zeros((len1 + 1, len2 + 1))
        for i in range(len1 + 1):
            table[i][0] = 0
        for j in range(len2 + 1):
            table[0][j] = 0

        match = penalties['match']
        miss = penalties['mismatch']
        gap = penalties['gap']
        score_max = 0

        for i in range(len1):
            for j in range(len2):
                if s1[i] == s2[j]:
                    temp = table[i][j] + match
                else:
                    temp = table[i][j] + miss

                new_max = max(temp, table[i][j+1] + gap, table[i+1][j] + gap, 0)
                
                if new_max > score_max:
                    score_max = new_max

                table[i+1][j+1] = new_max
        
        return int(score_max)
        #end code here

    def print_smith_waterman_alignment(self,s1,s2,penalties) :
        '''
        Input - two sequences and a dictionary with penalities for match, mismatch and gap
        Output - a tuple with two strings showing the two sequences with '-' representing the gaps
        '''
        #start code here
        len1 = len(s1)
        len2 = len(s2)
        max1 = 0
        max2 = 0
        str1 = ''
        str2 = ''

        table = np.zeros((len1 + 1, len2 + 1))
        for i in range(len1 + 1):
            table[i][0] = 0
        for j in range(len2 + 1):
            table[0][j] = 0

        match = penalties['match']
        miss = penalties['mismatch']
        gap = penalties['gap']
        score_max = 0

        for i in range(len1):
            for j in range(len2):
                if s1[i] == s2[j]:
                    temp = table[i][j] + match
                else:
                    temp = table[i][j] + miss

                new_max = max(temp, table[i][j+1] + gap, table[i+1][j] + gap, 0)
                
                if new_max > score_max:
                    score_max = new_max
                    max1 = i + 1
                    max2 = j + 1

                table[i+1][j+1] = new_max
        
        while table[max1][max2]:
            if s1[max1 - 1] == s2[max2 - 1]:
                str1 += s1[max1 - 1]
                str2 += s2[max2 - 1]
                max1 -= 1
                max2 -= 1

            elif table[max1][max2-1] + gap == table[max1][max2]:
                str1 += "-"
                str2 += s2[max2-1]
                max2 -= 1

            elif table[max1-1][max2] + gap == table[max1][max2]:
                str1 += s1[max1-1]
                str2 += "-"
                max1 -= 1
                
        return (str1[::-1], str2[::-1])
        #end code here

    def find_exact_matches(self,list_of_reads,genome):
        
        '''
        Input - list of reads of the same length and a genome fasta file (converted into a single string)
        Output - a list with the same length as list_of_reads, where the ith element is a list of all locations (starting positions) in the genome where the ith read appears. The starting positions should be specified using the "chr2:120000" format
        '''
        
        #start code here
        genlist = []
        gen_separate = genome.split('>')[1:]
        gendict = {}    
        outstr = []

        for read in gen_separate:
            read = read.replace('\n','')
            for num in range(10):
                read = read.replace(str(num),'')

            genlist.append(read.replace('chr',''))
        
        for i in range(len(genlist)):
            chromo = genlist[i]
            if len(chromo) >= len(list_of_reads[0]):
                for pos in range(len(chromo) - len(list_of_reads[0]) + 1):
                    pattern = chromo[pos : pos + len(list_of_reads[0])]
                    index = "chr%d:%d"%(i + 1, pos + 1)

                    if pattern not in gendict:
                         gendict[pattern] = [index]
                    else:
                         gendict[pattern].append(index)

        for read in list_of_reads:
            if read in gendict:
                outstr.append(gendict[read])
            else:
                outstr.append([])
            
        return outstr
        #end code here
       
    
    def find_approximate_matches(self,list_of_reads,genome):
        '''
        Input - list of reads of the same length and a genome fasta file (converted into a single string)
        Output -  a list with the same length as list_of_reads, where the ith element is a list of all locations (starting positions) in the genome which have the highest smith waterman alignment score with ith read in list_of_reads
        '''
        
        #start code here
        genlist = []
        gen_separate = genome.split('>')[1:]
        gendict = {}
        penalties = {'match': 1,'mismatch': -1,'gap': -1}
        score_max = 0
        outstr = []

        for read in gen_separate:
            read = read.replace('\n','')
            for num in range(10):
                read = read.replace(str(num),'')
            genlist.append(read.replace('chr',''))
        leng = len(list_of_reads[0]) // 4
        
        for ch_index in range(len(genlist)):
            chromo = genlist[ch_index]
            for pos in range(len(chromo) - leng + 1):
                pattern = chromo[pos: pos + leng]
                index_tuple = (ch_index, pos)
                if pattern not in gendict:
                    gendict[pattern] = [index_tuple]
                else:
                    gendict[pattern].append(index_tuple)
                    
        for read in list_of_reads:
            score_max = 0
            outlist = []
            for pos in range(len(read)- leng + 1):
                read_pattern = read[pos : pos + leng]
                if read_pattern in gendict:
                    index_list = gendict[read_pattern]
                    for index_tuple in index_list:
                        ch_index = index_tuple[0]
                        pos_index = index_tuple[1]
                        chromo = genlist[ch_index]
                        if len(read)<=len(chromo):
                            R = min(len(chromo) - len(list_of_reads[0]), pos_index + len(list_of_reads[0]))
                            L = max(0, pos_index - len(list_of_reads[0]))
                            for head in range(L, R + 1):
                                genome_pattern = chromo[head : head + len(list_of_reads[0])]
                                result = self.smith_waterman_alignment(read,genome_pattern, penalties)
                                index = "chr%d:%d"%(ch_index + 1, head + 1)
                                if result > score_max:
                                    outlist = [index]
                                    score_max = result
                                elif result == score_max and (index not in outlist):
                                    outlist.append(index)
            outstr.append(outlist)

        return outstr
        #end code here
        