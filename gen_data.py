#!/usr/bin/env python
import os, sys

# Set data download page
web_link = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/'

# Set data paths and names
data_path = 'data/'
all_data = ['a9a','real-sim','news20','rcv1','gis.train','gis.test',\
			'eps.train','eps.test','kddb.train','kddb.test','webspam']

# Set data paths in web_link
# Comment it, if you do not run it.
data_dict = {}
#data_dict['a9a'] = 'a9a'
data_dict['real-sim'] = 'real-sim.bz2'
#data_dict['news20'] = 'news20.binary.bz2'
#data_dict['rcv1'] = 'rcv1_test.binary.bz2'
#data_dict['gis.train'] = 'gisette_scale.bz2'
#data_dict['gis.test'] = 'gisette_scale.t.bz2'
#data_dict['eps.train'] = 'epsilon_normalized.bz2'
#data_dict['eps.test'] = 'epsilon_normalized.t.bz2'
#data_dict['kdda.train'] = 'kdda.bz2'
#data_dict['kdda.test'] = 'kdda.t.bz2'
#data_dict['kddb.train'] = 'kddb.bz2'
#data_dict['kddb.test'] = 'kddb.t.bz2'
#data_dict['webspam'] = 'webspam_wc_normalized_trigram.svm.bz2'


data_list = []
for data in all_data:
	if data in data_dict.keys():
		data_list.append(data)
print 'Ready to generate', ', '.join(data_list), '...'

for data in data_list:

	pathdata = os.path.join(data_path,data_dict[data])

	if not data.startswith('yahoo'):
		print '\n-------------------------------------'
		print 'Download', data_dict[data]
		link = '%s%s' %(web_link,data_dict[data])
		cmd = 'wget -O %s %s' %(pathdata,link)
		print cmd
		os.system(cmd)

	if not os.path.exists(pathdata):
		print 'Download failed!!'
		continue

	if pathdata.endswith('.bz2'):
		print '\n-------------------------------------'
		print 'Uncompress', data_dict[data]
		cmd = 'bunzip2 %s' %(pathdata)
		print cmd
		os.system(cmd)
		pathdata = pathdata[0:-4]

	datafile = os.path.join(data_path, data)

	if os.path.exists(pathdata):
		print '\n-------------------------------------'
		print 'Rename %s to %s' %(pathdata,datafile)
		cmd = 'mv %s %s' %(pathdata,datafile)
		print cmd
		os.system(cmd)
	else:
		print 'File does not exist!! (%s)' %(pathdata)
		continue

	if not data.startswith('gis') or\
	   not data.startswith('eps') or\
	   not data.startswith('kddb'):
		print '\n-------------------------------------'
		print 'Count the size of', data
		cmd = 'wc -l %s' %(datafile)
		print cmd
		fp = os.popen(cmd)
		total_size = int(fp.readline().strip().split()[0])
		test_size = total_size/5
		train_size = total_size-test_size
		fp.close()
		print total_size

		print '\n-------------------------------------'
		print 'Do a 80/20 split for', data, '(',train_size,'/',test_size,')'
		test_data = '%s.test' %(datafile)
		train_data = '%s.train' %(datafile)
		cmd = 'python subset.py -s 2 %s %s %s.fix %s %s'\
			%(datafile,test_size,datafile,test_data,train_data)
		print cmd
		os.system(cmd)

		if not os.path.exists(test_data) or\
		   not os.path.exists(train_data):
			print 'Split failed!!'
			continue

