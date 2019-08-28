# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
import sys
import time
import pdb
import random
import math
import threading
import numpy as np
import tensorflow as tf
from reader import Reader

flags = tf.app.flags
logging = tf.logging

flags.DEFINE_string('data_path', "/search/odin/imer/liuyujia/data", "Data directory.")
flags.DEFINE_string('dataset', "ptb", "Subdirectory of data directory.")
flags.DEFINE_string('model_name', "awd", "Model name by which model is saved.")

flags.DEFINE_integer('num_layer', 1, "How many layers to use in lstm model.")
flags.DEFINE_integer('embed_size', 1500, "Dimension for embedding vector.")
flags.DEFINE_integer('layer_size', [1500], "Layer dimension for each layer.")
flags.DEFINE_integer('vocab_size', 10000, "Vocabulary size.")
flags.DEFINE_integer('batch_size', 20, "Number of samples in each batch.")
flags.DEFINE_integer('num_step', 10, "Number of samples in each batch.")
flags.DEFINE_integer('top_k', 3, "Number of candidates when computing accuracy.")
flags.DEFINE_integer('step_per_log', 100, "Number of steps every log.")

flags.DEFINE_float('init_scale', 0.04, "Scale used when initialize variables.")
flags.DEFINE_float('embed_keep_prob', 0.9, "Percentage of state to keep through the dropout layer.")
flags.DEFINE_float('input_keep_prob', 0.5, "Percentage of input to keep through the dropout layer.")
flags.DEFINE_float('state_keep_prob', 0.5, "Percentage of input to keep through the dropout layer.")
flags.DEFINE_float('intra_layer_keep_prob', 0.5, "Percentage of output to keep through the dropout layer.")
flags.DEFINE_float('output_keep_prob', 0.5, "Percentage of output to keep through the dropout layer.")
flags.DEFINE_float('init_learning_rate', 1.0, "Initial learning rate.")
flags.DEFINE_float('max_grad_norm', 5.0, "Scale used when initialize variables.")
flags.DEFINE_float('ar_weight', 2.0, "Scale used when initialize variables.")
flags.DEFINE_float('tar_weight', 1.0, "Scale used when initialize variables.")
flags.DEFINE_float('weight_decay', 1.2e-6, "Scale used when initialize variables.")


flags.DEFINE_bool("enable_embedding_dropout", False, "dropout embedding matrix")
flags.DEFINE_bool("enable_weight_dropped_lstm", True, "dropout embedding matrix")
flags.DEFINE_bool("enable_ar_tar", False, "dropout embedding matrix")
flags.DEFINE_bool("enable_weight_decay", False, "dropout embedding matrix")
flags.DEFINE_bool("restore_model", False, "Restore model")

FLAGS = flags.FLAGS

class LSTM_Model():
	"""  Importing and running isolated TF graph """
	def __init__(self, name):
		# Create local graph and use it in the session
		self.graph = tf.Graph()
		self.sess = tf.Session(graph=self.graph, 
			config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
		self.is_training = name == "train"
		
		with self.graph.as_default():
			# Build graph
			with tf.name_scope("helper"):
				self.global_loss = tf.get_variable(
					name="global_loss", shape=[], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)	

				self.global_step = tf.Variable(0, name='global_step', trainable=False)  	
			
			with tf.name_scope("input"):
				self.x = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, FLAGS.num_step], name='x')
				self.y = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, FLAGS.num_step], name='y')
				# Count word num in a batch to measure speed
				
			initializer = tf.random_uniform_initializer(-FLAGS.init_scale,FLAGS.init_scale)
			# Create optimizer for training
			self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
			optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
			#optimizer = tf.train.AdamOptimizer()

			with tf.variable_scope("model", initializer=initializer):
				with tf.name_scope("input"):
					y_flat = tf.reshape(self.y, [-1])

				with tf.name_scope("embedding"):
					embedding = tf.get_variable("embedding", [FLAGS.vocab_size, FLAGS.embed_size])
					# Embedding Dropout
					if self.is_training:
						if FLAGS.enable_embedding_dropout:
							embedding = tf.nn.dropout(embedding, FLAGS.embed_keep_prob, noise_shape=[FLAGS.vocab_size, 1])
					inputs = tf.nn.embedding_lookup(embedding, self.x)

				with tf.name_scope("rnn"):	
					def lstm_cell(layer):
						if FLAGS.enable_weight_dropped_lstm:
							# Weight-Dropped LSTM
							from awd_cell import BasicLSTMCell
							if self.is_training:
								lstm_cell = BasicLSTMCell(FLAGS.layer_size[layer], forget_bias=0.0,
									state_keep_prob=FLAGS.state_keep_prob)
								# Variational Dropout
								if layer == 0:
									lstm_cell = tf.contrib.rnn.DropoutWrapper(
										lstm_cell, 
										input_keep_prob=FLAGS.input_keep_prob,
										variational_recurrent=True,
										input_size=FLAGS.embed_size,
										dtype=tf.float32)
								else:
									lstm_cell = tf.contrib.rnn.DropoutWrapper(
										lstm_cell, 
										input_keep_prob=FLAGS.intra_layer_keep_prob,
										variational_recurrent=True,
										input_size=FLAGS.layer_size[layer-1],
										dtype=tf.float32)
							else:
								lstm_cell = BasicLSTMCell(FLAGS.layer_size[layer], forget_bias=0.0)

						else:
							lstm_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.layer_size[layer], forget_bias=0.0)
							if self.is_training:
								# Variational Dropout
								if layer == 0:
									lstm_cell = tf.contrib.rnn.DropoutWrapper(
										lstm_cell, 
										input_keep_prob=FLAGS.input_keep_prob,
										#state_keep_prob=FLAGS.state_keep_prob,
										variational_recurrent=True,
										input_size=FLAGS.embed_size,
										dtype=tf.float32)
								else:
									lstm_cell = tf.contrib.rnn.DropoutWrapper(
										lstm_cell, 
										input_keep_prob=FLAGS.intra_layer_keep_prob,
										state_keep_prob=FLAGS.state_keep_prob,
										variational_recurrent=True,
										input_size=FLAGS.layer_size[layer-1],
										dtype=tf.float32)
						return lstm_cell
					
					cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(i) for i in range(FLAGS.num_layer)])

					# For each layer, get the initial state. states will be a tuple of LSTMStateTuples.
					state = self._get_state_variables(FLAGS.batch_size, cell)
					raw_outputs, last_state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, initial_state=state, dtype=tf.float32)
					# Add an operation to update the train states with the last state tensors.
					update_state = self._get_state_update_op(state, last_state)		
					with tf.control_dependencies(update_state):
						raw_outputs = tf.identity(raw_outputs)
					
					# Output dropout	
					if self.is_training:
						outputs = tf.nn.dropout(raw_outputs, FLAGS.output_keep_prob, noise_shape=[FLAGS.batch_size, 1, FLAGS.layer_size[-1]])
					else:
						outputs = raw_outputs
							
					outputs_flat = tf.reshape(outputs, [-1, FLAGS.layer_size[-1]])
					# Independent hidden size and embed size
					if FLAGS.layer_size[-1] != FLAGS.embed_size:
						softmax_w = tf.get_variable("softmax_w", [FLAGS.layer_size[-1], FLAGS.embed_size])
						softmax_b = tf.get_variable("softmax_b", [FLAGS.embed_size],
							initializer=tf.zeros_initializer())
						outputs_flat = tf.nn.bias_add(tf.matmul(outputs_flat, softmax_w), softmax_b)
					logits_flat = tf.matmul(outputs_flat, embedding, transpose_b=True)

				with tf.name_scope("loss"):
					self.loss = tf.reduce_sum(
						tf.nn.sparse_softmax_cross_entropy_with_logits(
							labels = y_flat,
							logits = logits_flat))
					
					if self.is_training:
						if FLAGS.enable_ar_tar:
							# Activiation Regularization	
							ar_loss = tf.nn.l2_loss(outputs_flat) * FLAGS.ar_weight
							# Temporal Activation Regularization (slowness)
							tar_loss = tf.nn.l2_loss(raw_outputs[:,1:,:] - raw_outputs[:,:-1,:]) * FLAGS.tar_weight
							self.loss = self.loss + ar_loss + tar_loss
						if FLAGS.enable_weight_decay:
							l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()
						                    if 'bias' not in v.name ]) * FLAGS.weight_decay
							self.loss = self.loss + l2_loss

			# Initialize average weights for NT-ASGD
			self._init_average_weights()

			grads = optimizer.compute_gradients(self.loss)
			_grads, _vars = zip(*grads)
			_grads, _ = tf.clip_by_global_norm(_grads, FLAGS.max_grad_norm)
			# Apply the gradients to adjust the shared variables.
			self.train_op = optimizer.apply_gradients(zip(_grads, _vars), global_step=self.global_step)
			with tf.control_dependencies([self.train_op]):
				self.train_and_average_op = self._update_averages_op()
			
			# assign_op that we assign average to variables when do valid
			self.assign_averages = self._assign_averages_op()
			# We must accumulate the loss and word num for validation phase at every step
			self.global_loss_update = tf.assign_add(self.global_loss, self.loss)

			model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
			# Add global_step to restore if necessary
			model_vars.append(self.global_step)
			self.model_saver = tf.train.Saver(model_vars)
			self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
			
	def _get_state_variables(self, batch_size, cell):
		# For each layer, get the initial state and make a variable out of it
		# to enable updating its value.
		state_variables = []
		for state_c, state_h in cell.zero_state(batch_size, tf.float32):
			state_variables.append(tf.contrib.rnn.LSTMStateTuple(
				tf.Variable(state_c, trainable=False),
				tf.Variable(state_h, trainable=False)))
		# Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
		return tuple(state_variables)

	def _get_state_update_op(self, state_variables, new_states):
		# Add an operation to update the train states with the last state tensors
		update_ops = []
		for state_variable, new_state in zip(state_variables, new_states):
			# Assign the new state to the state variables on this layer
			update_ops.extend(
				[state_variable[0].assign(new_state[0]),
				state_variable[1].assign(new_state[1])])
		# Return a tuple in order to combine all update_ops into a single operation.
		# The tuple's actual value should not be used.
		return tf.tuple(update_ops)	
	
	def init(self):	
		self.sess.run(self.init_op)		
		print("Variables initialized ...")
		
	def restore(self, model_dir):
		# Should call init before restore model in case some variables not in this ckpt
		ckpt_path = tf.train.latest_checkpoint(model_dir)
		if ckpt_path:
			self.model_saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))
			print("Read model parameters from %s" % tf.train.latest_checkpoint(model_dir))
		else:
			print("model doesn't exists")

	def save(self, model_path):
		self.model_saver.save(self.sess, model_path)

	# update variables in average collection after every train_op
	def assign_averages(self):
		self.sess.run(self.assign_averages)

	def run(self, data, learning_rate=0.0, mu=0.0):
		data_x, data_y = data
		if self.is_training:
			if mu > 0.0:
				return self.sess.run([self.loss, self.global_step, self.train_and_average_op], 
					feed_dict={ self.x: data_x, self.y: data_y, self.learning_rate: learning_rate, self.mu: mu })
			else:	
				return self.sess.run([self.loss, self.global_step, self.train_op], 
					feed_dict={ self.x: data_x, self.y: data_y, self.learning_rate: learning_rate})
		else:
			return self.sess.run([self.global_loss_update], 
				feed_dict={ self.x: data_x, self.y: data_y})

	def get_global_loss(self):
		return self.sess.run(self.global_loss)
	
	# Initialize variables for every trainable_variables in average collection
	def _init_average_weights(self):
		self._averages = {}
		with tf.name_scope("average"):
			for w in tf.trainable_variables():
				w_avg = tf.get_variable(w.op.name+"/average", shape=w.get_shape().as_list(), dtype=tf.float32, 
					initializer=tf.zeros_initializer(), trainable=False)
				self._averages[w] = w_avg
	
	def _update_averages_op(self):
		self.mu = tf.placeholder(tf.float32)
		update_list = []
		for w in tf.trainable_variables():
			w_avg = self._averages[w]
			update_op = w_avg.assign_add((w-w_avg)*self.mu)
			update_list.append(update_op)
		return tf.group(*update_list)
	
	def _assign_averages_op(self):
		assign_list = []
		model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model")
		for w in tf.trainable_variables():
			w_avg = self._averages[w]
			assign_op = w.assign(w_avg)
			assign_list.append(assign_op)
		return tf.group(*assign_list)

def main(_):
		
	# Collect data file information
	data_dir = os.path.join(FLAGS.data_path, FLAGS.dataset)
	train_filename = "ptb.train.txt"
	train_file = os.path.join(data_dir, "train", train_filename)
	valid_filename = "ptb.valid.txt"
	valid_file = os.path.join(data_dir, "valid", valid_filename)
	# Model directory to save and restore model parameters
	model_dir = os.path.join(FLAGS.model_name, "model")
	# Make sure model directory exists:
	if not os.path.isdir(model_dir):
		os.mkdir(model_dir) 	
	# Reader to read batch data form file
	reader = Reader(data_dir, FLAGS.vocab_size)		
	# Create train and valid graph in two separately graphs
	train_model = LSTM_Model("train")
	valid_model = LSTM_Model("valid")

	# Init train and valid model first
	train_model.init()
	if FLAGS.restore_model:
		train_model.restore(model_dir)

	learning_rate = FLAGS.init_learning_rate	
	epoch_num = 0
	mu = 0.0
	is_triggered = False
	triggered_step = 0
	valid_ppl_history = []
	train_data_gen = reader.get_simple_batch(train_file, FLAGS.batch_size, FLAGS.num_step)
	while True:
		epoch_num += 1
		# Training phase
		print("Training starts with learning rate {}".format(learning_rate))
		while True:
			try:					
				start_time = time.time()
				batch_loss, train_step, _ = train_model.run(train_data_gen.next(), learning_rate=learning_rate, mu=mu) 
				end_time = time.time()
			
			except StopIteration:
				print("One epoch data is finished")
				# Refresh generator for next epoch
				train_data_gen = reader.get_simple_batch(train_file, FLAGS.batch_size, FLAGS.num_step)
				print("Saving model...")
				# Save graph and model parameters
				model_path = os.path.join(model_dir, FLAGS.model_name+"_"+str(epoch_num))
				train_model.save(model_path)	
				break
		
			if train_step % FLAGS.step_per_log == 0:
				batch_word_num = FLAGS.batch_size * FLAGS.num_step
				train_speed = batch_word_num // (end_time-start_time)
				train_ppl = np.exp(batch_loss / (FLAGS.batch_size*FLAGS.num_step))
				print("Train step: {} Train ppl: {:.2f} Train speed: {}".format(
					train_step, train_ppl, train_speed))
				break
			
		# Validation phase
		print("start validation")
		# Restore model parameter from training
		valid_model.init()
		if is_triggered:
			valid_model.assign_averages()
		else:
			valid_model.restore(model_dir)
		valid_data_gen = reader.get_simple_batch(valid_file, FLAGS.batch_size, FLAGS.num_step)
		valid_step = 0
		start_time = time.time()
		while True:
			try:					
				valid_model.run(valid_data_gen.next()) 
				valid_step += 1

			except StopIteration:
				end_time = time.time()
				print("valid is finished in {} seconds".format(end_time-start_time))
				break
	
		valid_ppl = np.exp(valid_model.get_global_loss()/(valid_step*FLAGS.batch_size*FLAGS.num_step))
		print("Epoch: {} Valid ppl: {:.2f}".format(epoch_num, valid_ppl))
			
		# If converged, finish training
		if len(valid_ppl_history) >= 3:
			if is_triggered:
				mu = 1 / max(1, train_step-triggered_step) 
			elif valid_ppl > valid_ppl_history[-1]:
				is_triggered = True
				triggered_step = train_step
				print("start asgd at train_step: {}".format(train_step))
			if valid_ppl > max(valid_ppl_history[-3:]):
				print("Training is converged")
				break
		
		valid_ppl_history.append(valid_ppl)
				
if __name__ == "__main__":
	tf.app.run()

