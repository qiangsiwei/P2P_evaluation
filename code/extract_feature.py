# -*- encoding:utf-8 -*-

import re
import csv
import time
import fileinput
import numpy as np

from utils import *


ZMQ_Nline = 67

def parsefile(filename_ppd='../data/static_info_ppd.csv', filename_zmq='../data/static_info_zmq.txt', write_overlap=False):
	
	def parse(f_tags, f_score, f_averageProfit, f_registMoney, f_autobid, f_stockTransfer, f_fundsToken, f_guaranteeMode, f_guaranteeOrg, f_lauchTime, f_category, f_lng, f_lat):

		def extr_tags(x):
			types = ['国资系', '上市公司系', '银行系', '民营系'] # 公司类别
			others = ['投之家合作平台', '股权上市', '接受过风投', '争议', '加入第三方征信', '加入协会']
			r = [0]*7; tags = x.split(',')
			for i in xrange(len(tags)):
				if tags[i] in types: r[0] = types.index(tags[i])+1
				if tags[i] in others: r[others.index(tags[i])+1] = 1
			return r

		def extr_ones(x, cut):
			x = x.strip(cut)
			return 0 if x=='' else float(x)

		def extr_autobid(x):
			return 0 if x=='' else 1 if x=='支持' else -1

		def extr_stockTransfer(x):
			return -1 if x=='' else 0 if x=='随时' else 12 if x=='1年' else 300 if x=='不可转让' else x.strip('个月')
		
		def extr_fundsToken(x):
			return 0 if x=='' or x=='无托管' else 1

		def extr_ifGuarantee(x):
			return 0 if x=='' else 1

		def extr_lauchTime(x):
			import datetime
			def date_difference(d1, d2):
				if '-' in d1 and '-' in d2:
					d1 = datetime.datetime.strptime(d1 + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
				else:
					d1 = datetime.datetime.strptime(d1 + ' 00:00:00', '%Y年%m月%d日 %H:%M:%S')
				d2 = datetime.datetime.strptime(d2 + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
				return (d2-d1).days/30
			return 6 if x =='' else date_difference(x, '2016-05-01')

		def extr_category(x):
			_dict = {'股份合作企业':0, '私营企业':1, '港、澳、台投资企业':2, '股份制企业':3, \
					 '集体所有制企业':4, '外商投资企业':5, '国有企业':6, '联营企业':7}
			return _dict.get(x,-1)

		return extr_tags(f_tags)+\
			   [extr_ones(f_score,''),\
				extr_ones(f_averageProfit,'%'),\
				extr_ones(f_registMoney,' 万元'),\
				extr_autobid(f_autobid),\
				extr_stockTransfer(f_stockTransfer),\
				extr_fundsToken(f_fundsToken),\
				extr_ifGuarantee(f_guaranteeMode),\
				extr_ifGuarantee(f_guaranteeOrg),\
				extr_lauchTime(f_lauchTime),\
				extr_category(f_category),\
				extr_ones(f_lng,''),\
				extr_ones(f_lat,'')]

	ppd_platforms = {}; zmq_platforms = {}

	# 生成ppd静态特征
	with open(filename_ppd, 'rb') as csvfile_ppd:
		reader = csv.DictReader(csvfile_ppd)
		for row in reader:
			parsed = parse(row['tags'],row['score'],row['averageProfit'],row['registMoney'],row['autobid'],row['stockTransfer'],row['fundsToken'],\
						   row['guaranteeMode'],row['guaranteeOrg'],row['lauchTime'],row['category'],row['lng'],row['lat'])
			ppd_platforms[row['platName']] = parsed
	
	# 生成zmq静态特征
	for line in fileinput.input(filename_zmq):
		if fileinput.lineno() == 1:
			field_names = line.strip().split('\t')
		else:
			row = {name:field for name, field in zip(field_names,line.strip().split('\t'))}
			parsed = parse('','',row.get('平均收益',''),row.get('注册资本',''),row.get('自动投标',''),row.get('债权转让',''),row.get('资金托管',''),\
						   row.get('保障模式',''),'',row.get('上线时间',''),row.get('公司类型',''),'','')
			zmq_platforms[row['平台名称']] = parsed
		if fileinput.lineno() == ZMQ_Nline:
			break
	fileinput.close()

	if write_overlap:
		print '重合平台：', len(set(ppd_platforms.keys())&set(zmq_platforms.keys()))
		with open('overlap_platforms.txt','w') as outfile:
			outfile.write('\n'.join(list(set(ppd_platforms.keys())&set(zmq_platforms.keys()))))
	
	return ppd_platforms, zmq_platforms


def parsecomment(filename_ppd='../data/comment_ppd.csv', filename_zmq='../data/comment_zmq.txt'):
	from snownlp import SnowNLP

	ppd_platforms = {}; zmq_platforms = {}
	with open(filename_ppd, 'rb') as csvfile_ppd:
		reader = csv.DictReader(csvfile_ppd)
		for row in reader:
			if row['content']:
				gentime = (lambda x:x.tm_year*12+x.tm_mon)(time.gmtime(int(row['timestamp'])))
				ppd_platforms[row['platName']] = ppd_platforms.get(row['platName'],{})
				ppd_platforms[row['platName']][gentime] = ppd_platforms[row['platName']].get(gentime,[])+[{'推荐':1,'一般':0.5}.get(row['attitude'],0)]
				# ppd_platforms[row['platName']][gentime] = ppd_platforms[row['platName']].get(gentime,[])+[SnowNLP(row['content']).sentiments]
	return ppd_platforms, zmq_platforms


if __name__ == '__main__':
	start = time.clock()

	parsefile()
	# parsecomment()

	elapsed = time.clock()-start
	print '\nTime used:', elapsed

