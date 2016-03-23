"""Check goodness of CMU Dictionary before aligning.
"""

def check():
    phoneset = set('AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH'.split())
    for line in open('cmudict.forhtk.txt').readlines():
        for phone in line.split()[1:]:
            if phone[0] in 'AEIOU':
                if phone[:2] not in phoneset or phone[-1] not in '012':
                    print line, ':', phone, 'invalid'
            elif phone not in phoneset:
                print line,':', phone, 'invalid'
    
if __name__=='__main__':
    check()
    
    
