# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math
import numpy as np
from collections import Counter, defaultdict
import pickle
import pdb

class Reader(object):
	def __init__(self, data_dir, vocab_size):
		self.BOS = "<s>"
		self.EOS = "</s>"
		self.UNK = "<unk>"
		self.data_dir = data_dir
		self.vocab_size = vocab_size

		self.build_vocab()
	
	def build_vocab(self):
		vocab_path = os.path.join(self.data_dir, "vocabulary.pkl")
		train_dir = os.path.join(self.data_dir, "train")
		valid_dir = os.path.join(self.data_dir, "valid")
		test_dir = os.path.join(self.data_dir, "test")
		
		if os.path.isfile(vocab_path):
			#self.words = open(vocab_path).read().replace("\n", " ").split()
			vocab_file = open(vocab_path, 'rb')
			self.words = pickle.load(vocab_file)
			vocab_file.close()
		else:
			data = []
			train_files = os.listdir(train_dir)
			for f in [os.path.join(train_dir, train_file) for train_file in train_files]:
				data.extend(self.read_words(f))
			valid_files = os.listdir(valid_dir)
			for f in [os.path.join(valid_dir, valid_file) for valid_file in valid_files]:
				data.extend(self.read_words(f))
			test_files = os.listdir(test_dir)
			for f in [os.path.join(test_dir, test_file) for test_file in test_files]:
				data.extend(self.read_words(f))
			
			counter = Counter(data)  # sort words '.': 5, ',': 4......
			count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
			words = list(zip(*count_pairs)[0])
			print("total vocab size: %d"%len(words))
			
			# make sure <unk> is in vocabulary
			if self.UNK not in words:
				words.insert(0, self.UNK)
			# make sure EOS is id 0
			words.insert(0, self.EOS)
			self.words = words[:self.vocab_size]
			assert len(self.words) == self.vocab_size

			# Save the vocabulary with pickle for future use
			vocab_file = open(vocab_path, 'wb')
			pickle.dump(self.words, vocab_file)
			vocab_file.close()
	
		print("vocab size: %d"%len(self.words))
		self.word2id = dict(zip(self.words, range(self.vocab_size))) 
	
	def read_words(self, file_path):  # return 1-D list
		words = []
		with open(file_path) as f:
			for line in f:
				words.extend(line.strip().split())
		return words

	def from_line_to_id(self, line):
		word_list = line.strip().split()
		word_list.append(self.EOS)
		id_list = [self.word2id[word] for word in word_list]
		return id_list

	def get_simple_batch(self, data_file, batch_size, num_step):
		raw_data = []
		with open(data_file, 'r') as f:
			for line in f:
				id_list = self.from_line_to_id(line)
				raw_data.extend(id_list)
		batch_len = len(raw_data) // batch_size
		batch_data = np.array(raw_data[:batch_len * batch_size]).reshape([batch_size, batch_len])
		batch_num = (batch_len-1) // num_step
		for i in range(batch_num):
			batch_input = batch_data[:, i*num_step:(i+1)*num_step]
			batch_output = batch_data[:, i*num_step+1:(i+1)*num_step+1]
			yield batch_input, batch_output

	def padding_batch(self, batch_data):
		batch_size = len(batch_data)
		padding_len = len(max(batch_data, key=len))
		result = np.zeros([batch_size, padding_len])
		result_len = np.zeros(batch_size)
		for i, data in enumerate(batch_data):
			data_len = len(data)
			result_len[i] = data_len 
			result[i][:data_len] = data
		return result, result_len
	
	def get_dynamic_batch(self, data_file, batch_size):
		batch_inputs = []
		batch_outputs = []
		with open(data_file, 'r') as f:
			for line in f:
				id_list = self.from_line_to_id(line)
				batch_inputs.append(id_list[:-1])
				batch_outputs.append(id_list[1:])
				if len(batch_inputs) == batch_size:
					inputs, inputs_len = self.padding_batch(batch_inputs)
					outputs, _ = self.padding_batch(batch_outputs)
					batch_inputs = []
					batch_outputs = []
					yield inputs, inputs_len, outputs 

