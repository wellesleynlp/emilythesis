"""Convert CSV transcriptions to label files"""
import sys
import os
import string

filename = sys.argv[1]
lang = filename[:2]

for line in open(filename):
    wavfile, trans = line.split(',', 1)
    trans = filter(lambda c: c in string.lowercase or c in string.whitespace or c=="'" or c=='-', trans.lower())
    trans = trans.replace('-', ' ').replace("'", "\\'")
    o = open(os.path.join(lang, wavfile.split('.')[0]+'.lab'), 'w')
    o.write(trans)
    o.close()

    
