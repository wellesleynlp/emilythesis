import textgrid
import sys
import os
import numpy as np

__author__='Emily Ahn and Sravana Reddy'

lang = sys.argv[1] # lang = "AR"
filename = sys.argv[2] # filename = "FAR00013" # NO EXTENSION
aligndir = 'alignments'
speechdir = 'cslu_fae_corpus/npytxt'

tg = textgrid.TextGrid()
tg.read(os.path.join(aligndir, lang, filename+'.TextGrid'))

# file_data is np matrix, where each index represents 10ms (or 0.01 sec)
file_data = np.load(os.path.join(speechdir, lang, filename+'.npytxt'))

def create_phone_dict():
	phone_dict = {}

	for intv in tg.tiers[0].intervals:
		plp_starti = intv.minTime*100
		plp_endi = intv.maxTime*100
		plp_ft = [] #2-D array
		for window in range(plp_starti, plp_endi):
			#insert 52-D vector for that window
			plp_ft.append(file_data[window]) 
		# check if intv.mark exists in phone_dict yet
		if intv.mark not in phone_dict:
			phone_dict[intv.mark] = plp_ft
		else:
			phone_dict[intv.mark].extend(plp_ft)
	# print plp_ft
	# break # test only for 1st intv
	return phone_dict


create_phone_dict()
