import collections

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

from tensorflow.python.ops.rnn_cell import RNNCell, LSTMStateTuple
from tensorflow.python.ops.rnn_cell_impl import _Linear

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class BasicLSTMCell(RNNCell):
	"""Basic LSTM recurrent network cell.
	The implementation is based on: http://arxiv.org/abs/1409.2329.
	We add forget_bias (default: 1) to the biases of the forget gate in order to
	reduce the scale of forgetting in the beginning of the training.
	It does not allow cell clipping, a projection layer, and does not
	use peep-hole connections: it is the basic baseline.
	For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
	that follows.
	"""

	def __init__(self, num_units, forget_bias=1.0,
							 state_keep_prob=1.0,
							 state_is_tuple=True, activation=None, reuse=None):
		"""Initialize the basic LSTM cell.
		Args:
			num_units: int, The number of units in the LSTM cell.
			forget_bias: float, The bias added to forget gates (see above).
				Must set to `0.0` manually when restoring from CudnnLSTM-trained
				checkpoints.
			state_is_tuple: If True, accepted and returned states are 2-tuples of
				the `c_state` and `m_state`.	If False, they are concatenated
				along the column axis.	The latter behavior will soon be deprecated.
			activation: Activation function of the inner states.	Default: `tanh`.
			reuse: (optional) Python boolean describing whether to reuse variables
				in an existing scope.  If not `True`, and the existing scope already has
				the given variables, an error is raised.
			When restoring from CudnnLSTM-trained checkpoints, must use
			CudnnCompatibleLSTMCell instead.
		"""
		super(BasicLSTMCell, self).__init__(_reuse=reuse)
		if not state_is_tuple:
			logging.warn("%s: Using a concatenated state is slower and will soon be "
									 "deprecated.  Use state_is_tuple=True.", self)
		if not (state_keep_prob >= 0.0 and state_keep_prob <= 1.0):
			raise ValueError("state_keep_prob is expecting value in range 0 to 1: %f" % state_keep_prob)

		self._num_units = num_units
		self._forget_bias = forget_bias
		self._state_keep_prob = state_keep_prob
		self._state_is_tuple = state_is_tuple
		self._activation = activation or math_ops.tanh
		self._linear = None

		# Create mask for recurrent weights
		self._mask_tensor = nn_ops.dropout(array_ops.ones([num_units, 4 * num_units]), keep_prob=state_keep_prob)
		#self._mask_tensor = random_ops.random_uniform([num_units, 4*num_units])

	@property
	def state_size(self):
		return (LSTMStateTuple(self._num_units, self._num_units)
						if self._state_is_tuple else 2 * self._num_units)

	@property
	def output_size(self):
		return self._num_units
	

	def call(self, inputs, state):
		"""Long short-term memory cell (LSTM).
		Args:
			inputs: `2-D` tensor with shape `[batch_size x input_size]`.
			state: An `LSTMStateTuple` of state tensors, each shaped
				`[batch_size x self.state_size]`, if `state_is_tuple` has been set to
				`True`.  Otherwise, a `Tensor` shaped
				`[batch_size x 2 * self.state_size]`.
		Returns:
			A pair containing the new hidden state, and the new state (either a
				`LSTMStateTuple` or a concatenated state, depending on
				`state_is_tuple`).
		"""
		sigmoid = math_ops.sigmoid
		# Parameters of gates are concatenated into one multiply for efficiency.
		if self._state_is_tuple:
			c, h = state
		else:
			c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

		if self._linear is None:
			self._linear = _Linear([inputs, h], 4 * self._num_units, True)
			if self._state_keep_prob < 1.0:
				weights = self._linear._weights
				input_size = weights.get_shape().as_list()[0] - self._num_units
				input_weights, state_weights = array_ops.split(weights, [input_size, self._num_units])
				state_weights = state_weights * self._mask_tensor
				self._linear._weights = array_ops.concat([input_weights, state_weights], 0)
		# i = input_gate, j = new_input, f = forget_gate, o = output_gate
		i, j, f, o = array_ops.split(
				value=self._linear([inputs, h]), num_or_size_splits=4, axis=1)

		new_c = (
				c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
		new_h = self._activation(new_c) * sigmoid(o)

		if self._state_is_tuple:
			new_state = LSTMStateTuple(new_c, new_h)
		else:
			new_state = array_ops.concat([new_c, new_h], 1)
		return new_h, new_state


