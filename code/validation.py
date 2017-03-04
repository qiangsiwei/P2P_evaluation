# -*- encoding:utf-8 -*-

import re
import csv
import time
import fileinput
import numpy as np

from extract_feature import *
from train_model import *


def model_compare():
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.cross_validation import KFold
	from sklearn.metrics import roc_auc_score

	(x_static, x_seq_neg, x_seq_neu, x_seq_pos), y = generate_training_data()
	x = np.hstack([x_static, x_seq_neg, x_seq_neu, x_seq_pos])

	for train_index, test_index in KFold(len(y), n_folds=5):
		clf = RandomForestClassifier(n_estimators=500, max_depth=3)
		clf.fit(x[train_index],y[train_index])
		y_pred = clf.predict_proba(x[test_index])[:,1]; y_true = y[test_index]
		print roc_auc_score(y_true, y_pred)
		# print 1.*(clf.predict(x[test_index]) == y[test_index]).sum()/len(y[test_index])


def feature_importance():
	from sklearn.ensemble import RandomForestClassifier

	feature_names = ['公司类别', '是否投之家合作', '是否股权上市', '是否风投', '是否争议', '是否第三方征信', '是否加入协会', \
					 '综合评分', '平均利率', '注册资金', '自动投注', '股权转让', '是否托管',\
					 '保障类型', '保障机构', '注册时间', '企业类别', '经度', '纬度']

	(x_static, x_seq_neg, x_seq_neu, x_seq_pos), y = generate_training_data()
	x = np.hstack([x_static, x_seq_neg, x_seq_neu, x_seq_pos])

	clf = RandomForestClassifier(n_estimators=500, max_depth=3)
	clf.fit(x,y)
	feature_importances = clf.feature_importances_[:19]
	for i, _index in enumerate(feature_importances.argsort()[::-1]):
		print '{0}\t{1}\t{2:.5f}'.format(i+1, feature_names[_index], feature_importances[_index])


if __name__ == '__main__':
	start = time.clock()

	# model_compare()
	feature_importance()

	elapsed = time.clock()-start
	print '\nTime used:', elapsed

