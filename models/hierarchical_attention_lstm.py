import tensorflow as tf
from utils.hierarchical_attn_wrapper import HierarchicalAttentionCellWrapper


class TPABIDIRECTIONALLSTM:
    def __init__(self, para):
        self.para = para
        self.dtype = tf.float32
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self._build_graph()
        if self.para.mode == 'train':
            self._build_optimizer()
        self.saver = tf.train.Saver(max_to_keep=3)

    def _build_graph(self):
        print('Building {} graph...'.format(self.para.mode))
        x_length = self.para.data_config.x.range[1] - self.para.data_config.x.range[0] + 1
        y_length = self.para.data_config.y.range[1] - self.para.data_config.y.range[0] + 1
        z_length = self.para.data_config.z.range[1] - self.para.data_config.z.range[0] + 1

        y_channel = len(self.para.data_config.y.key)
        assert x_length == self.para.attention_len

        self.x = tf.placeholder(dtype=tf.float32,
                                shape=[None, self.para.attention_len, len(self.para.data_config.x.key)],
                                name='x')

        self.y = tf.placeholder(dtype=tf.float32,
                                shape=[None,
                                       y_length,
                                       len(self.para.data_config.y.key)],
                                name='y')

        self.z = tf.placeholder(dtype=tf.float32,
                                shape=[None,
                                       z_length,
                                       len(self.para.data_config.z.key)],
                                name='z')

        rnn_inputs = tf.expand_dims(self.x, axis=3)
        full_connection_inputs = self._build_multi_conv_layer(rnn_inputs, self.para['filters'])
        attn_vec = tf.squeeze(tf.layers.dense(full_connection_inputs, 1))
        all_rnn_states, final_rnn_state = tf.contrib.rnn

    def _build_single_conv_layer(self, inputs, filters):
        conv_out = tf.layers.conv2d(inputs,
                                    filters=filters,
                                    kernel_size=[1, 3],
                                    padding='valid',
                                    activation=tf.nn.relu)

        max_pool_out = tf.layers.max_pooling2d(conv_out,
                                               pool_size=[1, 3],
                                               strides=1,
                                               padding='valid')
        return max_pool_out

    def _build_multi_conv_layer(self, inputs, filters):
        for i, channel in enumerate(filters):
            with tf.variable_scope('feature_extract{}'.format(i)):
                inputs = self._build_single_conv_layer(inputs, channel)
        return inputs

    def _build_multi_lstm(self):
        return tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(i) for i in self.para['units_of_cells']])

        # with tf.variable_scope('TPABlock'):
        #     self.rnn_inputs_embed = tf.nn.relu(
        #         tf.layers.dense(self.x, self.para.num_units)
        #     )
        #
        #     self.rnn_inputs_embed = tf.unstack(self.rnn_inputs_embed, axis=1)
        #
        #     self.all_rnn_states, self.final_rnn_states = tf.nn.static_rnn(
        #         cell=self._build_rnn_cell(),
        #         inputs=self.rnn_inputs_embed,
        #         sequence_length=tf.constant([self.para.attention_len for _ in range(self.para.batch_size)], dtype=tf.int32),
        #         dtype=self.dtype
        #     )
        #
        #     self.final_rnn_states = tf.concat(
        #         [self.final_rnn_states[i][1] for i in range(self.para.num_layers)],
        #         axis=1
        #     )
        #
        #     self.future_tpa_outputs = tf.reshape(
        #         tf.layers.dense(self.final_rnn_states, y_length * self.para.num_units),
        #         [-1, y_length, self.para.num_units]
        #     )
        #
        # with tf.variable_scope('LSTMNetWork'):
        #     self.lstm_inputs = tf.layers.dense(self.z, self.para.num_units)
        #     self.all_lstm_outputs, self.final_lstm_states = tf.nn.bidirectional_dynamic_rnn(
        #         cell_fw=tf.contrib.rnn.LSTMCell(self.para.num_units),
        #         cell_bw=tf.contrib.rnn.LSTMCell(self.para.num_units),
        #         inputs=self.lstm_inputs,
        #         dtype=self.dtype
        #     )
        #     self.future_lstm_outputs = tf.slice(
        #         tf.concat(self.all_lstm_outputs, axis=2),
        #         [0, x_length, 0],
        #         [-1, y_length, -1]
        #     )
        #
        # self.final_outputs = tf.layers.dense(
        #     tf.concat([self.future_tpa_outputs, self.future_lstm_outputs], axis=2),
        #     y_channel
        # )

        # if self.para.highway > 0:
        #     self.reg_outputs = tf.expand_dims(
        #         tf.layers.dense(self.x[:, -self.para.highway:, 0], y_length, use_bias=False), axis=2)

        # self.final_outputs = self.merge_outputs + self.reg_outputs

        self.loss = self._compute_loss(self.final_outputs, self.y)

    def _build_optimizer(self):
        print('Building optimizer...')
        trainable_variables = tf.trainable_variables()
        if self.para.decay > 0:
            lr = tf.train.exponential_decay(
                self.para.learning_rate,
                self.global_step,
                self.para.decay,
                0.995,
                staircase=True
            )
        else:
            lr = self.para.learning_rate
        self.opt = tf.train.AdamOptimizer(lr)
        gradients = tf.gradients(self.loss, trainable_variables)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.para.max_gradient_norm)
        self.update = self.opt.apply_gradients(
            zip(clip_gradients, trainable_variables),
            global_step=self.global_step
        )

    def _compute_loss(self, outputs, labels):
        print('Building loss')
        loss = tf.losses.absolute_difference(labels=labels, predictions=outputs)
        return loss

    def _build_rnn_cell(self):
        return tf.contrib.rnn.MultiRNNCell([self._build_single_cell() for _ in range(self.para.num_layers)])

    def _build_lstm_cell(self):
        return tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(self.para.num_units) for _ in range(2)])

    def _build_single_cell(self):
        cell = tf.contrib.rnn.LSTMBlockCell(self.para.num_units)
        if self.para.mode == 'train':
            cell = tf.contrib.rnn.DropoutWrapper(
                cell=cell,
                input_keep_prob=(1.0 - self.para.dropout),
                output_keep_prob=(1.0 - self.para.dropout),
                state_keep_prob=(1.0 - self.para.dropout)
            )
        cell = TemporalPatternAttentionCellWrapper(cell, self.para.attention_len)
        return cell