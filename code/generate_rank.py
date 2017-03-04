# -*- encoding:utf-8 -*-

import re
import csv
import time
import fileinput
import numpy as np

from extract_feature import *
from train_model import *


def generate_rankscore(BaseDim=19, DeltaTime=27):
	from keras.models import Model
	from keras.layers import Input, Embedding, LSTM, Dense, Dropout, merge
	from keras.layers.embeddings import Embedding
	from keras.utils.visualize_util import plot
	from sklearn.cross_validation import KFold
	from sklearn.metrics import roc_auc_score

	# 置空特征
	(x_static, x_seq_neg, x_seq_neu, x_seq_pos), y = generate_training_data()
	x_static = x_static[:,[8,9,10,11,12,13,15,16]]; BaseDim = len(x_static[0])
	x_seq_neg, x_seq_neu, x_seq_pos = np.zeros(x_seq_neg.shape), np.zeros(x_seq_neu.shape), np.zeros(x_seq_neu.shape)

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
	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=["accuracy"])
	model.fit([x_static, x_seq_neg, x_seq_neu, x_seq_pos], y, nb_epoch=10, batch_size=32)

	platNames, (x_static, x_seq_neg, x_seq_neu, x_seq_pos), y = generate_training_data(data_source='zmq', return_platNames=True)
	x_static = x_static[:,[8,9,10,11,12,13,15,16]]
	x_seq_neg, x_seq_neu, x_seq_pos = np.zeros(x_seq_neg.shape), np.zeros(x_seq_neu.shape), np.zeros(x_seq_neu.shape)
	rankscores = model.predict([x_static, x_seq_neg, x_seq_neu, x_seq_pos], batch_size=32)
	overlap_names = open('overlap_platforms.txt').read().split('\n')
	for name, score in zip(platNames, rankscores[:,0]):
		if not name in overlap_names:
			print '{0}\t{1}'.format(name, 1-score)


def generate_rankscore_overlap(BaseDim=19, DeltaTime=27):
	from keras.models import Model
	from keras.layers import Input, Embedding, LSTM, Dense, Dropout, merge
	from keras.layers.embeddings import Embedding
	from keras.utils.visualize_util import plot
	from sklearn.cross_validation import KFold
	from sklearn.metrics import roc_auc_score

	platNames, (x_static, x_seq_neg, x_seq_neu, x_seq_pos), y = generate_training_data(return_platNames=True)
	overlap_names = open('overlap_platforms.txt').read().split('\n')
	overlap_index = np.array([platNames.index(name) for name in overlap_names])

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
	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=["accuracy"])

	model.fit([x_static, x_seq_neg, x_seq_neu, x_seq_pos], y, nb_epoch=10, batch_size=32)
	rankscores = model.predict([x_static[overlap_index], x_seq_neg[overlap_index], x_seq_neu[overlap_index], x_seq_pos[overlap_index]], batch_size=32)

	for name, score in zip(overlap_names, rankscores[:,0]):
		print '{0}\t{1}'.format(name, 1-score)


if __name__ == '__main__':
	start = time.clock()

	generate_rankscore()
	# generate_rankscore_overlap()

	elapsed = time.clock()-start
	print '\nTime used:', elapsed

