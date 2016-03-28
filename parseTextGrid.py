import textgrid
import sys
import os
import numpy as np

__author__='Emily Ahn and Sravana Reddy'

""" Date created: 3/10/2016
	Date modified: 3/27/2016
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
		# do not include '' and 'sil' intervals
		if (not intv.mark=='') or (not intv.mark=='sil'):
			plp_starti = int(intv.minTime*100)
			plp_endi = int(intv.maxTime*100)
			plp_ft = [] #2-D array
			for window in range(plp_starti, plp_endi):
				#insert 52-D vector for that window
				plp_ft.append(file_data[window]) 
			# check if intv.mark exists in phone_dict yet
			if intv.mark not in phone_dict:
				phone_dict[intv.mark] = plp_ft
			else:
				phone_dict[intv.mark].extend(plp_ft)
	return phone_dict

def phone_dict_tofile(phone_dict, lang, filename, homedir):
	""" For 1 file and its phone dict, store information in folder under phones/lang, 
		with the format: filename+phone+.npy
	"""
	# initialize folder
	if not os.path.isdir(os.path.join(homedir, 'phones', lang)):
		os.makedirs(os.path.join(homedir, 'phones', lang))
	# store each phone as its own file
	for phone in phone_dict:
		np_array = np.array(phone_dict[phone])
		write_name = os.path.join(homedir, 'phones', lang, filename+phone+'.npy')
		np.save(write_name, np_array)

if __name__=='__main__':
	aligndir = 'alignments'
	speechdir = 'cslu_fae_corpus/plptxt'
	homedir = sys.argv[1] # i.e. '/home/sravana/data'
	lang = sys.argv[2] # i.e. 'HI' or 'KO'

	# for lang in os.listdir(os.path.join(homedir, "cslu_fae_corpus/speech")):
	for filename in os.listdir(os.path.join(homedir, "cslu_fae_corpus/speech/" + lang)):
		if filename.endswith('.wav'):
			tg = textgrid.TextGrid()
			file_noext = os.path.splitext(filename)[0]
			tg.read(os.path.join(aligndir, lang, file_noext+'.TextGrid'))

			# file_data is np matrix, where each index represents 10ms (or 0.01 sec)
			file_data = np.loadtxt(os.path.join(homedir, speechdir, lang, file_noext+'.npytxt'))
			phone_dict = create_phone_dict(file_data, tg)
			phone_dict_tofile(phone_dict, lang, file_noext, homedir+'/cslu_fae_corpus')
