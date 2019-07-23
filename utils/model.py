import tensorflow as tf
from utils.attention_wrapper import TemporalPatternAttentionCellWrapper

class PolyRNN:
    def __init__(self, para):
        self.para = para
        self.dtype = tf.float32
        self.mode = tf.placeholder(dtype=tf.string, name='mode')
        self.train_mode = tf.constant('train', dtype=tf.string, name='train_mode')
        self.validation_mode = tf.constant('validation', dtype=tf.string, name='validation_mode')
        self.test_mode = tf.constant('test', dtype=tf.string, name='test_mode')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self._build_graph()
        if self.para.mode == 'train':
            self._build_optimizer()
        self.saver = tf.train.Saver(max_to_keep=3)

    def _build_graph(self):
        print('Building graph...')
        self.x = tf.placeholder(dtype=tf.float32,
                                shape=[None, self.para.attention_len, len(self.para.data_config.x.key)],
                                name='x')

        self.y = tf.placeholder(dtype=tf.float32,
                                shape=[None, 1, len(self.para.data_config.y.key)],
                                name='y')

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

        self.rnn_outputs = []

        for i in range(self.para.data_config.y.range[1] - self.para.data_config.y.range[0] + 1):
            self.rnn_outputs.append(tf.layers.dense(self.final_rnn_states, len(self.para.data_config.y.key),
                                                        name='rnn_output{}'.format(i)))
        self.all_rnn_outputs = tf.stack(self.rnn_outputs, axis=1)

        self.loss = self._compute_loss(self.all_rnn_outputs, self.y)


        # if self.para.highway > 0:
        #     reg_outputs = tf.transpose(self.x[:, -self.para.highway:, :], [0, 2, 1])
        #     reg_outputs = tf.layers.dense(reg_outputs, 1)
        #     self.all_rnn_outputs += tf.squeeze(reg_outputs)

        # if self.para.mode == 'train' or self.para.mode == 'validation':
        #     self.labels = tf.squeeze(self.y, axis=1)
        #     self.loss = self._compute_loss(self.all_rnn_outputs, self.labels)
        # elif self.para.mode == 'test':
        #     self.labels = tf.squeeze(self.y, axis=1)

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

    def _build_single_cell(self):
        cell = tf.contrib.rnn.LSTMBlockCell(self.para.num_units)
        # if self.para.mode == 'train':
        #     cell = tf.contrib.rnn.DropoutWrapper(
        #         cell=cell,
        #         input_keep_prob=(1.0 - self.para.dropout),
        #         output_keep_prob=(1.0 - self.para.dropout),
        #         state_keep_prob=(1.0 - self.para.dropout)
        #     )

        def test():
            return cell

        cell = tf.cond(tf.equal(self.mode, self.train_mode), self._build_dropout_wrapper(cell), test())
        cell = TemporalPatternAttentionCellWrapper(cell, self.para.attention_len)
        return cell

    def _build_dropout_wrapper(self, cell):
        cell = tf.contrib.rnn.DropoutWrapper(
            cell=cell,
            input_keep_prob=(1.0 - self.para.dropout),
            output_keep_prob=(1.0 - self.para.dropout),
            state_keep_prob=(1.0 - self.para.dropout)
        )
        return cell
