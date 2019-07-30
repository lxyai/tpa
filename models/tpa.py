import tensorflow as tf
from utils.attention_wrapper import TemporalPatternAttentionCellWrapper


class SingleTPA:
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

        with tf.variable_scope('TPABlock'):
            self.rnn_inputs_embed = tf.nn.relu(
                tf.layers.dense(self.x, self.para.num_units)
            )

        self.rnn_inputs_embed = tf.unstack(self.rnn_inputs_embed, axis=1)

        self.all_rnn_states, self.final_rnn_states = tf.nn.static_rnn(
            cell=self._build_rnn_cell(),
            inputs=self.rnn_inputs_embed,
            sequence_length=tf.constant([self.para.attention_len for _ in range(self.para.batch_size)], dtype=tf.int32),
            dtype=self.dtype
        )

        self.final_rnn_states = tf.concat(
            [self.final_rnn_states[i][1] for i in range(self.para.num_layers)],
            axis=1
        )

        self.future_tpa_outputs = tf.reshape(
            tf.layers.dense(self.final_rnn_states, y_length * y_channel),
            [-1, y_length, y_channel]
        )

        self.loss = self._compute_loss(self.future_tpa_outputs, self.y)

    def _build_optimizer(self):
        print('Building {} optimizer...'.format(self.para.mode))
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
