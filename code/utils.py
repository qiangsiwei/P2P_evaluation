# -*- encoding:utf-8 -*-

import re
import csv
import time
import fileinput
import numpy as np


def to_unicode(string):
	if isinstance(string, unicode):
		return string
	elif isinstance(string, str):
		return str(string).decode('utf-8')
	elif isinstance(string, float):
		return str(int(string)).decode('utf-8')
	else:
		return u''


def excel_to_txt(filename='../result/自贸区重合特征.xlsx'):
	import xlrd
	data = xlrd.open_workbook(filename)
	table = data.sheet_by_index(0)
	with open('../data/static_info_zmq.txt','w') as outfile:
		for i in range(table.nrows):
			outfile.write(u'\t'.join([re.sub(ur'[\t\n]','',to_unicode(field)) for field in table.row_values(i)]).encode('utf-8')+'\n')


if __name__ == '__main__':
	start = time.clock()

	excel_to_txt()

	elapsed = time.clock()-start
	print '\nTime used:', elapsed

