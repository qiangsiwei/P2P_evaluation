# -*- encoding:utf-8 -*-

import re
import csv
import time
import fileinput
import numpy as np

from extract_feature import *


def generate_training_data(labelfile='../data/label_ppd.txt', DeltaTime=27, Thres_Neg=0.4, Thres_Pos=0.5, checkpoint=2016*12+4, return_platNames=False, data_source='ppd'):
	NegativeInstances = set([]) 
	for line in fileinput.input(labelfile):
		if fileinput.lineno() == 1:
			field_names = line.strip().split('\t')
		else:
			row = {name:field for name, field in zip(field_names,line.strip().split('\t'))}
			NegativeInstances = NegativeInstances|set([row['name']])
	fileinput.close()

	static_ppd, static_zmq = parsefile()
	dynamic_ppd, dynamic_zmq = parsecomment()
	
	if data_source == 'ppd':
		static, dynamic = static_ppd, dynamic_ppd
	elif data_source == 'zmq':
		static, dynamic = static_zmq, dynamic_zmq
	else:
		raise Exception('Data Source Not Supported.')
	
	platNames, x_static, x_seq_neg, x_seq_neu, x_seq_pos, y = [], [], [], [], [], []
	for platName, static in static.iteritems():
		platNames.append(platName)
		seq_neg = [(np.array(dynamic[platName].get(checkpoint-i,[]))<=Thres_Neg).sum() for i in xrange(DeltaTime)][::-1] if platName in dynamic else [0]*DeltaTime
		seq_neu = [(Thres_Neg<(np.array(dynamic[platName].get(checkpoint-i,[])))*(np.array(dynamic[platName].get(checkpoint-i,[]))<Thres_Pos)).sum() for i in xrange(DeltaTime)][::-1] if platName in dynamic else [0]*DeltaTime
		seq_pos = [(np.array(dynamic[platName].get(checkpoint-i,[]))>=Thres_Pos).sum() for i in xrange(DeltaTime)][::-1] if platName in dynamic else [0]*DeltaTime
		x_static.append(static); x_seq_neg.append(seq_neg); x_seq_neu.append(seq_neu); x_seq_pos.append(seq_pos); y.append(platName in NegativeInstances)
	
	if return_platNames:
		return platNames, map(lambda x:np.array(x),[x_static, x_seq_neg, x_seq_neu, x_seq_pos]), np.array(y)
	else:
		return map(lambda x:np.array(x),[x_static, x_seq_neg, x_seq_neu, x_seq_pos]), np.array(y)


def train_model(BaseDim=19, DeltaTime=27, simulate_missing_rate=False):
	from keras.models import Model
	from keras.layers import Input, Embedding, LSTM, Dense, Dropout, merge
	from keras.layers.embeddings import Embedding
	from keras.utils.visualize_util import plot
	from sklearn.cross_validation import KFold
	from sklearn.metrics import roc_auc_score

	(x_static, x_seq_neg, x_seq_neu, x_seq_pos), y = generate_training_data()
	
	if simulate_missing_rate:
		x_static = x_static[:,[8,9,10,11,12,13,15,16]]; BaseDim = len(x_static[0])
		x_seq_neg, x_seq_neu, x_seq_pos = np.zeros(x_seq_neg.shape), np.zeros(x_seq_neu.shape), np.zeros(x_seq_pos.shape)

	# 交叉验证
	for train_index, test_index in KFold(len(y), n_folds=5):
		x_static_train, x_seq_neg_train, x_seq_neu_train, x_seq_pos_train, y_train = \
		map(lambda x:x[train_index], [x_static, x_seq_neg, x_seq_neu, x_seq_pos, y])

		# 建立模型
		base_dim = BaseDim; series_dims = [DeltaTime]*3
		base_input = Input(shape=(base_dim,), name='base_input')
		base_dense = Dense(64, input_dim=base_dim, init='uniform', activation='relu')(base_input)
		base_model = Dropout(0.5)(base_dense)
		series_inputs, series_models = [], []
		for i in xrange(len(series_dims)):
			series_input = Input(shape=(series_dims[i],), dtype='int32', name='series_input' + str(i))
			series_inputs.append(series_input)
			series_embedding = Embedding(output_dim=128, input_dim=10000, input_length=series_dims[i])(series_input)
			lstm_out = LSTM(32)(series_embedding)
			series_models.append(lstm_out)
		x = merge([base_model] + series_models, mode='concat')
		x = Dense(64, activation='relu')(x)
		main_loss = Dense(1, activation='sigmoid', name='main_output')(x)
		model = Model(input=[base_input]+series_inputs, output=main_loss)
		# plot(model, to_file='train_model.png')
		# print model.summary()
		model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=["accuracy"])
		model.fit([x_static_train, x_seq_neg_train, x_seq_neu_train, x_seq_pos_train], y_train, nb_epoch=10, batch_size=32)

		x_static_test, x_seq_neg_test, x_seq_neu_test, x_seq_pos_test, y_test = \
		map(lambda x:x[test_index], [x_static, x_seq_neg, x_seq_neu, x_seq_pos, y])

		pred = model.predict([x_static_test, x_seq_neg_test, x_seq_neu_test, x_seq_pos_test], batch_size=32)
		y_pred = [float(pred[i][0]) for i in xrange(len(pred))]; y_true = [int(y_test[i]) for i in xrange(len(pred))]
		print roc_auc_score(y_true, y_pred)


if __name__ == '__main__':
	start = time.clock()

	# train_model()
	train_model(simulate_missing_rate=True)

	elapsed = time.clock()-start
	print '\nTime used:', elapsed

