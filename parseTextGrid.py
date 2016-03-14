import textgrid
import sys
import os
import numpy as np

__author__='Emily Ahn and Sravana Reddy'

""" Date created: 3/10/2016
	Date modified: 3/13/2016
	*******************************************************************************************
	Given forced alignments (transcriptions) of all speech files in 1 lang, create .npytxt files
	sorted by filename and phoneme, where each file contains plp features of only that phoneme.
"""

def create_phone_dict(file_data, tg):
	""" For 1 file, take in all plp features (file_data) and corresponding TextGrid (tg). 
		Return dictionary where each phone maps to corresponding windows of plp features.
	"""
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

def phone_dict_tofile(phone_dict, lang, filename):
	""" For 1 file and its phone dict, store information in folder under phones/lang, 
		with the format: filename+phone+.npy
	"""
	# initialize folder
	if not os.path.isdir(os.path.join('phones', lang)):
        os.mkdir(os.path.join('phones', lang))
    for phone in phone_dict:
    	np_array = np.array(phone_dict[phone])
    	write_name = os.path.join('phones', lang, filename+phone+'.npy')
    	np.save(write_name, np_array)

if __name__=='__main__':
	aligndir = 'alignments'
	speechdir = 'cslu_fae_corpus/npytxt'
	homedir = sys.argv[1]

	for lang in os.listdir("cslu_fae_corpus/speech"):
		for filename in os.listdir(homedir, "cslu_fae_corpus/speech/" + lang):
			if filename.endswith('.wav'):
				tg = textgrid.TextGrid()
				tg.read(os.path.join(homedir, aligndir, lang, filename+'.TextGrid'))

				# file_data is np matrix, where each index represents 10ms (or 0.01 sec)
				file_data = np.load(os.path.join(homedir, speechdir, lang, filename+'.npytxt'))
				phone_dict = create_phone_dict(file_data, tg)
				phone_dict_tofile(phone_dict, lang, filename)
