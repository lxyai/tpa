import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl

# class HierarchicalAttentionMechanism:
#     def __call__(self, s_vector, h_matrix_list):
#         pass


class HierarchicalAttentionCellWrapper(rnn_cell_impl.RNNCell):
    def __init__(self,
                 cell,
                 exogenous_length,
                 exogenous_vec_list,
                 output_size=None,
                 mid_size=None,
                 reuse=None):

        super(HierarchicalAttentionCellWrapper, self).__init__(_reuse=reuse)
        self._cell = cell
        self._exogenous_length = exogenous_length
        self._exogenous_vec_list = exogenous_vec_list
        if output_size is None:
            self._output_size = self._cell.output_size
        else:
            self._output_size = output_size
        if mid_size is None:
            self._mid_size = self._output_size
        else:
            self._mid_size = mid_size

    @property
    def state_size(self):
        size = (self._cell.output_size, self._cell.state_size)

        return size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        cell_prev_outputs, cell_states = state
        attn_list = []
        for i, vec in enumerate(self._exogenous_vec_list):
            with tf.variable_scope('score{}'.format(i)):
                vec_hidden_size = vec.get_shape().as_list()[2]
                e = self.score(self._mid_size, vec_hidden_size, cell_prev_outputs, vec)
                a = tf.expand_dims(tf.nn.softmax(e), axis=2)
                attn = tf.reduce_sum(tf.multiply(vec, a), axis=1)
                attn_list.append(attn)
        attn_list.append(inputs)
        final_attn = tf.concat(attn_list, axis=1)
        rnn_output, rnn_states = self._cell(final_attn, cell_states)
        return rnn_output, (rnn_output, rnn_states)





    def score(self, mid_size, h_size, S, H):
        # T: [mid_size cell_output_size]
        # U: [mid_size, h_size]
        # V: [mid_size]
        # S: [batch_size, cell_output_size]
        # H: [batch_size, attn_length, h_size]

        # T = tf.Variable(tf.random_uniform([mid_size, self._cell.output_size()], 0, 1.0),
        #                 trainable=True, dtype=tf.float32)
        # U = tf.Variable(tf.random_uniform([mid_size, h_size], 0, 1.0),
        #                 trainable=True, dtype=tf.float32)
        # V = tf.Variable(tf.random_uniform([mid_size], 0, 1.0),
        #                 trainable=True, dtype=tf.float32)
        init = tf.random_uniform_initializer(-1.0, 1.0, dtype=tf.float32)
        T = tf.get_variable('T', shape=[mid_size, self._cell.output_size], initializer=init)
        U = tf.get_variable('U', shape=[mid_size, h_size], initializer=init)
        V = tf.get_variable('V', shape=[mid_size], initializer=init)
        # TS: [batch_size, mid_size]
        # UH: [batch_size, attn_length, mid_size]
        TS = tf.tensordot(S, T, [[1], [1]])
        UH = tf.tensordot(H, U, [[2], [1]])
        # tanh(TS + UH): [batch_size, attn_length, mid_size]
        tanh_TS_UH = tf.tanh(
            tf.add(tf.expand_dims(TS, axis=1), UH)
        )
        # E: [batch_size, attn_length]
        E = tf.tensordot(tanh_TS_UH, V, [[2], [0]])
        return E

