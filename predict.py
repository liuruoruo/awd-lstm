# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import time
import pdb
import random
import math
import pickle
import argparse
import numpy as np
import tensorflow as tf

from dynamic_lstm import DynamicLSTM
from reader import Reader 
from queue import Queue
import config

def test_from_input():
	reader = Reader()
	model = DynamicLSTM(None, is_training=False, reuse=False)
	model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
	model_saver = tf.train.Saver(model_variables)
	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		ckpt_path = tf.train.latest_checkpoint(config.model_dir)
		if ckpt_path:
			model_saver.restore(sess, tf.train.latest_checkpoint(config.model_dir))
			print("Read model parameters from %s" % tf.train.latest_checkpoint(config.model_dir))
		else:
			print("model doesn't exists")
		
		user_input = raw_input("input: ")
		while user_input:
			inputs, inputs_len = reader.get_batch_from_input(user_input)
			feed_dict={	model.x: inputs, model.x_len: inputs_len }
			x, inputs, last_state, last_output, output_prob = sess.run([model.x, 
																																	model.inputs,
																																	model.last_state,
																																	model.last_output,
																																	model.output_prob],
																																	feed_dict = feed_dict)
			pdb.set_trace()
			user_input = raw_input("input: ")


def test_from_file(filepath):
	reader = Reader()
	model = DynamicLSTM(None, is_training=False, reuse=False)
	model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
	model_saver = tf.train.Saver(model_variables)
	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		ckpt_path = tf.train.latest_checkpoint(config.model_dir)
		if ckpt_path:
			model_saver.restore(sess, tf.train.latest_checkpoint(config.model_dir))
			print("Read model parameters from %s" % tf.train.latest_checkpoint(config.model_dir))
		else:
			print("model doesn't exists")

		data_gen = reader.get_custom_line_from_file(filepath)
		for inputs, inputs_len in data_gen:
			feed_dict={	model.x: inputs, model.x_len: inputs_len}
			prob = sess.run([model.output_prob], feed_dict=feed_dict)


def test_top_3(filepath):
	reader = Reader()
	queue = Queue("test", config.batch_size)
	model = DynamicLSTM(queue, is_training=False, reuse=False)
	model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
	model_saver = tf.train.Saver(model_variables)
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
		sess.run(init_op)
		
		ckpt_path = tf.train.latest_checkpoint(config.model_dir)
		if ckpt_path:
			model_saver.restore(sess, tf.train.latest_checkpoint(config.model_dir))
			print("Read model parameters from %s" % tf.train.latest_checkpoint(config.model_dir))
		else:
			print("model doesn't exists")
		
		data_gen = reader.get_custom_batch_from_file(filepath, config.batch_size)
		correct = 0
		total = 0
		line_num = 0
		for inputs, inputs_len, outputs in data_gen:
			line_num += 1
			sess.run(queue.enqueue_op, feed_dict={	queue.inputs: inputs, 
																							queue.inputs_len: inputs_len, 
																							queue.outputs: outputs })
			probs = sess.run(model.output_prob)
			pdb.set_trace()
			for i in range(config.batch_size):
				max_val = probs[i][outputs[i][0]]
				if max_val >= probs[i][outputs[i][1]] and max_val >= probs[i][outputs[i][2]]:
					correct += 1
			total += config.batch_size
		print("total: %d correct: %d correct rate: %.4f" %(total, correct, float(correct)/float(total)))



def main(_):
	"""	
	if len(sys.argv) > 1:
		test_from_file(sys.argv[1])
	else:
		test_from_input()
	"""
	#test_from_file(sys.argv[1])
	test_top_3(sys.argv[1])

if __name__ == "__main__":
	tf.app.run()
