import numpy as np
from collections import OrderedDict

class Lab2(object):
    
    def smith_waterman_alignment(self,s1,s2,penalties) :
        '''
        Input - two sequences and a dictionary with penalities for match, mismatch and gap
        Output - an integer value which is the maximum smith waterman alignment score
        '''
        #start code here
        m = penalties['match']
        s = penalties['mismatch']
        d = penalties['gap']
        max_score = 0
        
        array = np.zeros((len(s1)+1, len(s2)+1))
        for i in range(len(s1)+1):
            array[i][0] = 0
        for j in range(len(s2)+1):
            array[0][j] = 0
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i]==s2[j]:
                    temp = array[i][j]+m
                else:
                    temp = array[i][j]+s
                new = max(temp, array[i][j+1]+d, array[i+1][j]+d, 0)
                if new > max_score:
                    max_score = new
                array[i+1][j+1] = new
                
        return int(max_score)
        #end code here

    def print_smith_waterman_alignment(self,s1,s2,penalties) :
        '''
        Input - two sequences and a dictionary with penalities for match, mismatch and gap
        Output - a tuple with two strings showing the two sequences with '-' representing the gaps
        '''
        #start code here
        m = penalties['match']
        s = penalties['mismatch']
        d = penalties['gap']
        max_score = 0
        
        array = np.zeros((len(s1)+1, len(s2)+1))
        for i in range(len(s1)+1):
            array[i][0] = 0
        for j in range(len(s2)+1):
            array[0][j] = 0
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i]==s2[j]:
                    temp = array[i][j]+m
                else:
                    temp = array[i][j]+s
                new = max(temp, array[i][j+1]+d, array[i+1][j]+d, 0)
                if new > max_score:
                    max_score = new
                    max_i = i+1
                    max_j = j+1
                array[i+1][j+1] = new
                
        i = max_i
        j = max_j
        output1 = ""
        output2 = ""
        while array[i][j]:
            if s1[i-1] == s2[j-1]:
                output1 = s1[i-1]+output1
                output2 = s2[j-1]+output2
                i -= 1
                j -= 1
            elif array[i][j-1]+d == array[i][j]:
                output1 = "-"+output1
                output2 = s2[j-1]+output2
                j -= 1
            elif array[i-1][j]+d == array[i][j]:
                output1 = s1[i-1]+output1
                output2 = "-"+output2
                i -= 1
            # else:
            #     output1 = s1[i-1]+output1
            #     output2 = s2[j-1]+output2
            #     i -= 1
            #     j -= 1
                
        return (output1, output2)
        #end code here

    def find_exact_matches(self,list_of_reads,genome):
        
        '''
        Input - list of reads of the same length and a genome fasta file (converted into a single string)
        Output - a list with the same length as list_of_reads, where the ith element is a list of all locations (starting positions) in the genome where the ith read appears. The starting positions should be specified using the "chr2:120000" format
        '''
        
        #start code here
        L = len(list_of_reads[0])
        genome_list = []
        genome_split = genome.split('>')[1:]
        number=['0','1','2','3','4','5','6','7','8','9']
        for rd in  genome_split:
            rd=rd.replace('\n','')
            for num in number:
                rd=rd.replace(num,'')
            genome_list.append(rd.replace('chr',''))
        
        genomeDict = {}
        
        for i in range(len(genome_list)):
            chromosome = genome_list[i]
            G = len(chromosome)
            if G>= L:
                for position in range(G-L+1):
                    segment = chromosome[position: position+L]
                    index = "chr%d:%d"%(i+1,position+1)
                    if segment not in genomeDict:
                        genomeDict[segment] = [index]
                    else:
                        genomeDict[segment].append(index)
                    
        output = []
        for read in list_of_reads:
            if read in genomeDict:
                output.append(genomeDict[read])
            else:
                output.append([])
            
        return output
        #end code here
       
    
    def find_approximate_matches(self,list_of_reads,genome):
        '''
        Input - list of reads of the same length and a genome fasta file (converted into a single string)
        Output -  a list with the same length as list_of_reads, where the ith element is a list of all locations (starting positions) in the genome which have the highest smith waterman alignment score with ith read in list_of_reads
        '''
        
        #start code here
        L = len(list_of_reads[0])
        genome_list = []
        genome_split = genome.split('>')[1:]
        number=['0','1','2','3','4','5','6','7','8','9']
        for rd in  genome_split:
            rd=rd.replace('\n','')
            for num in number:
                rd=rd.replace(num,'')
            genome_list.append(rd.replace('chr',''))
        k = L//4
        
        genomeDict = {}
        
        for ch_index in range(len(genome_list)):
            chromosome = genome_list[ch_index]
            G = len(chromosome)
            for position in range(G-k+1):
                segment = chromosome[position: position+k]
                index_tuple = (ch_index, position)
                if segment not in genomeDict:
                    genomeDict[segment] = [index_tuple]
                else:
                    genomeDict[segment].append(index_tuple)
                    
        penalties={'match':1,'mismatch':-1,'gap':-1}
        max_score = 0
        output = []
        for read in list_of_reads:
            max_score = 0
            output_list = []
            for position in range(len(read)-k+1):
                read_segment = read[position: position+k]
                if read_segment in genomeDict:
                    index_list = genomeDict[read_segment]
                    for index_tuple in index_list:
                        ch_index = index_tuple[0]
                        pos_index = index_tuple[1]
                        chromosome = genome_list[ch_index]
                        if len(read)<=len(chromosome):
                            left = max(0, pos_index-L)
                            right = min(len(chromosome)-L, pos_index+L)
                            for start in range(left, right+1):
                                genome_segment = chromosome[start: start+L]
                                score = self.smith_waterman_alignment(read,genome_segment,penalties)
                                index = "chr%d:%d"%(ch_index+1,start+1)
                                if score > max_score:
                                    output_list = [index]
                                    max_score = score
                                elif score == max_score and (index not in output_list):
                                    output_list.append(index)
            output.append(output_list)

        return output
        #end code here
        
        