"""Convert CSV transcriptions to label files"""
import sys
import os
import string

if __name__=='__main__':
    filename = sys.argv[1]
    lang = filename[:2]
    oovs = set()
    cmudictwords = set([line.split()[0] for line in open('cmudict.forhtk.txt').readlines()])
    for line in open(filename):
        wavfile, trans = line.split(',', 1)
        trans = filter(lambda c: c in string.lowercase or c in string.whitespace or c in "'-.", trans.lower())
        trans = trans.replace('-', ' ').replace('.', ' ').replace("'", "\\'")
        transwords = trans.split()
        with open(os.path.join(lang, wavfile.split('.')[0]+'.lab'), 'w') as o:
            for word in transwords:
                if word not in cmudictwords:
                    oovs.add(word)
            if len(oovs)==0:  # write lab file only if no OOVs
                o.write(trans)
    if len(oovs)>0:
        print 'There are', len(oovs), 'OOV words. Add to dictionary or correct the CSV transcriptions, and try again.'
        for oov in oovs:
            print oov
    else:
        print 'Success!'

    
