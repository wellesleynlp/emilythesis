# concatenate filenames (with delimeter file) into a string
# use string to enter sox command into terminal


from os import listdir

dirNames = ['cslu_fae_corpus/speech/AR', 'cslu_fae_corpus/speech/CZ', 'cslu_fae_corpus/speech/IN']
#files_AR = listdir(dirNames[0])
#files_CZ= listdir(dirNames[1])
#files_IN = listdir(dirNames[2])

single = ""
# enter index # to change name
for filename in listdir(dirNames[2]):
    single += filename + " marwick.wav "
print "sox " + single