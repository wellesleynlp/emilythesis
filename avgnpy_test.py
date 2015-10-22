# script to convert full PLP feature values into avgerages of PLP features
# 10.19.2015
# Emily Ahn

import numpy
from os import listdir

def writeOneDir(oldDir, newDir):
    for fileName in listdir(oldDir):
        old_fileName = oldDir + "/" + fileName
        avg = numpy.mean(numpy.loadtxt(old_fileName), axis=0)
        new_fileName = newDir + "/" + fileName
        numpy.savetxt(new_fileName, avg)
        



'''fileName = 'cslu_fae_corpus/npytxt/AR/FAR00013.npytxt'

with open(fileName) as f:
    fullText = f.readlines()
bigList = []
for line in fullText:
    nums = line.split()
    bigList.append(nums)
# now bigList holds double-array of all numbers

avgList = []
#for every column, find average
for n in range(52): 
    list_n = [float(bigList[j][n]) for j in range(52)]
    avg_n = numpy.mean(list_n)
    avgList.append(avg_n)

#print int(' '.join(str(x) for x in avgList))
for x in avgList:
    print x'''

'''def writeOneDir(oldDir, newDir):
    for fileName in listdir(oldDir):
        old_fileName = oldDir + "/" + fileName
        with open(old_fileName) as f:
            fullText = f.readlines()
        bigList = []
        for line in fullText:
            nums = line.split()
            bigList.append(nums)
        # now bigList holds double-array of all numbers
        
        avgList = []
        #for every column, find average
        for n in range(52): 
            list_n = [float(bigList[j][n]) for j in range(len(bigList))]
            avg_n = numpy.mean(list_n)
            avgList.append(avg_n)
        
        new_fileName = newDir + "/" + fileName
        with open(new_fileName, 'w') as f2:
            for x in avgList:
                f2.write(str(x) + ' ')'''
        
        

writeOneDir('cslu_fae_corpus/npytxt/AR', 'cslu_fae_corpus/npytxt_avg/AR')
writeOneDir('cslu_fae_corpus/npytxt/MA', 'cslu_fae_corpus/npytxt_avg/MA')
writeOneDir('cslu_fae_corpus/npytxt/HI', 'cslu_fae_corpus/npytxt_avg/HI')
    

