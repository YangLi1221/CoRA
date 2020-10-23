import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer as xavier
from utilize import bn_dense_layer_v2

FLAGS = tf.app.flags.FLAGS

class NN(object):
    def __init__(self, is_training, init_vec):
        self.word = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name='input_word')
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name='input_pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name='input_pos2')
        self.mask = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name='input_mask')
        self.len = tf.placeholder(dtype=tf.int32, shape=[None], name='input_mask')
        self.label_index = tf.placeholder(dtype=tf.int32, shape=[None], name='label_index')
        self.label = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.num_classes], name='input_label')
        self.scope = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size + 1], name='scope')
        self.en1 = tf.placeholder(dtype=tf.int32, shape=[None], name='input_en1')
        self.en2 = tf.placeholder(dtype=tf.int32, shape=[None], name='input_en2')
        self.sen_hier1 = tf.placeholder(dtype=tf.int32, shape=[None], name='hier1_label_index')
        self.sen_hier2 = tf.placeholder(dtype=tf.int32, shape=[None], name='hier2_label_index')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.hier = init_vec['relation_levels'].shape[1]
        self.relation_levels = tf.constant(init_vec['relation_levels'], shape=[FLAGS.num_classes, self.hier], dtype=tf.int32, name='relation_levels')

        word_size = FLAGS.word_size
        vocab_size = FLAGS.vocabulary_size - 2

        with tf.variable_scope("embedding_lookup", initializer=xavier(), dtype=tf.float32):
            temp_word_embedding = self._GetVar(init_vec=init_vec, key='wordvec', name='temp_word_embedding', shape=[vocab_size, word_size], trainable=True)
            unk_word_embedding = self._GetVar(init_vec=init_vec, key='unkvec', name='unk_embedding', shape=[word_size], trainable=True)
            word_embedding = tf.concat([temp_word_embedding, tf.reshape(unk_word_embedding, [1, word_size]),
                                        tf.reshape(tf.constant(np.zeros(word_size), dtype=tf.float32), [1, word_size])], 0)
            temp_pos1_embedding = self._GetVar(init_vec=init_vec, key='pos1vec', name='temp_pos1_embedding', shape=[FLAGS.pos_num, FLAGS.pos_size], trainable=True)
            temp_pos2_embedding = self._GetVar(init_vec=init_vec, key='pos2vec', name='temp_pos2_embedding', shape=[FLAGS.pos_num, FLAGS.pos_size], trainable=True)
            pos1_embedding = tf.concat([temp_pos1_embedding, tf.reshape(tf.constant(np.zeros(FLAGS.pos_size, dtype=np.float32)), [1, FLAGS.pos_size])], 0)
            pos2_embedding = tf.concat([temp_pos2_embedding, tf.reshape(tf.constant(np.zeros(FLAGS.pos_size, dtype=np.float32)), [1, FLAGS.pos_size])], 0)

            input_word = tf.nn.embedding_lookup(word_embedding, self.word)
            input_pos1 = tf.nn.embedding_lookup(pos1_embedding, self.pos1)
            input_pos2 = tf.nn.embedding_lookup(pos2_embedding, self.pos2)
            input_en1 = tf.nn.embedding_lookup(word_embedding, self.en1)
            input_en2 = tf.nn.embedding_lookup(word_embedding, self.en2)

            self.input_embedding = tf.concat(values=[input_word, input_pos1, input_pos2], axis=2)
            self.input_entity = tf.concat(values=[input_en1, input_en2], axis=1)
            self.sentence = input_word
        self.hidden_size, self.sentence_encoder = self._GetEncoder(FLAGS.model, is_training)


    def _GetVar(self, init_vec, key, name, shape=None, initializer=xavier(), trainable=True):
        if init_vec is not None and key in init_vec:
            print('using pretrained {} and is {}'.format(key, 'trainable' if trainable else 'not trainable'))
            return tf.get_variable(name=name, initializer=init_vec[key], trainable=trainable)
        else:
            return tf.get_variable(name=name, shape=shape, initializer=initializer, trainable=trainable)

    def _GetEncoder(self, model, is_training):
        return FLAGS.hidden_size * 3, self.extractor

    def extractor(self, is_training, init_vec=None):
        with tf.variable_scope("sentence_encoder", dtype=tf.float32, initializer=xavier(), reuse=tf.AUTO_REUSE):
            entity = tf.expand_dims(self.input_entity, axis=1) * tf.ones(shape=[1, FLAGS.max_length, 1], dtype=tf.float32)
            word_with_entity = tf.concat([self.sentence, entity], 2)
            dim_word_entity = word_with_entity.shape[2]
            t_cnn = 0.05
            "gate entity aware"
            pos_info = bn_dense_layer_v2(self.input_embedding, dim_word_entity, True, 0., 'pos_info', 'tanh', wd=0., keep_prob=1., is_train=is_training)
            word_gated_cnn = bn_dense_layer_v2(word_with_entity / t_cnn, dim_word_entity, True, 0., 'word_gated', 'sigmoid', False, wd=0., keep_prob=1., is_train=is_training)
            final_vector_cnn = word_gated_cnn * word_with_entity + (1-word_gated_cnn) * pos_info
            "pcnn"
            mask_embedding = tf.constant([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
            pcnn_mask = tf.nn.embedding_lookup(mask_embedding, self.mask)
            cnn_input = tf.expand_dims(final_vector_cnn, axis=1)
            with tf.variable_scope('conv2d_pos'):
                conv_kernel = self._GetVar(init_vec=None, key='convkernel', name='kernel_pos', shape=[1, 3, dim_word_entity, FLAGS.hidden_size], trainable=True)
                conv_bias = self._GetVar(init_vec=None, key='convbias', name='bias_pos', shape=[FLAGS.hidden_size], trainable=True)
            x = tf.layers.conv2d(inputs=cnn_input, filters=FLAGS.hidden_size, kernel_size=[1,3], strides=[1,1], padding='same', reuse=tf.AUTO_REUSE)
            x = tf.reshape(x, [-1, FLAGS.max_length, FLAGS.hidden_size, 1])
            pcnn_x = tf.reshape(pcnn_mask, [-1, 1, FLAGS.max_length, 3]) * tf.transpose(x, [0, 2, 1, 3])
            output = tf.nn.relu(tf.reshape(tf.reduce_max(pcnn_x, 2), [-1, FLAGS.hidden_size * 3]))
        return output

class CoRA(NN):
    def __init__(self, is_training, init_vec):
        NN.__init__(self, is_training, init_vec)
        x = self.sentence_encoder(is_training, init_vec)
        with tf.variable_scope('bag-vote', initializer=xavier(), dtype=tf.float32):
            hier3_relation_matrix = self._GetVar(init_vec=None, key=None, name='hier3_relation_matrix', initializer=tf.orthogonal_initializer(), shape=[FLAGS.num_classes, self.hidden_size])
            hier2_relation_matrix = self._GetVar(init_vec=None, key=None, name='hier2_relation_matrix', initializer=tf.orthogonal_initializer(), shape=[FLAGS.num_hier2_classes, self.hidden_size])
            hier1_relation_matrix = self._GetVar(init_vec=None, key=None, name='hier1_relation_matrix', initializer=tf.orthogonal_initializer(), shape=[FLAGS.num_hier1_classes, self.hidden_size])

            "hierarchical_rank1"
            hier1_logits = tf.matmul(x, hier1_relation_matrix, transpose_b=True)
            hier1_index = tf.nn.softmax(hier1_logits, -1)
            hier1_relation = tf.matmul(hier1_index, hier1_relation_matrix)
            "gate"
            concat_hier1 = tf.concat([x, hier1_relation], 1)
            alpha_hier1 = bn_dense_layer_v2(concat_hier1, self.hidden_size, True, scope='gate_hier1', activation='sigmoid', is_train=is_training)
            context_hier1 = alpha_hier1 * x + (1 - alpha_hier1) * hier1_relation
            "MLP linear"
            middle_hier1 = bn_dense_layer_v2(context_hier1, 1024, False, scope='mlp_activation_hier1', activation='relu', is_train=is_training)
            output_hier1 = bn_dense_layer_v2(middle_hier1, self.hidden_size, False, scope='mlp_linear_hier1', activation='linear', is_train=is_training)
            "add&norm"
            output_hier1 += x
            output_hier1 = tf.contrib.layers.layer_norm(output_hier1)

            "hierarchical_rank2"
            hier2_logits = tf.matmul(x, hier2_relation_matrix, transpose_b=True)
            hier2_index = tf.nn.softmax(hier2_logits, -1)
            hier2_relation = tf.matmul(hier2_index, hier2_relation_matrix)
            "gate_hier2"
            concat_hier2 = tf.concat([x, hier2_relation], 1)
            alpha_hier2 = bn_dense_layer_v2(concat_hier2, self.hidden_size, True, scope='gate_hier2', activation='sigmoid', is_train=is_training)
            context_hier2 = alpha_hier2 * x + (1 - alpha_hier2) * hier2_relation
            "MLP linear"
            middle_hier2 = bn_dense_layer_v2(context_hier2, 1024, False, scope='mlp_activation_hier2', activation='relu', is_train=is_training)
            output_hier2 = bn_dense_layer_v2(middle_hier2, self.hidden_size, False, scope='mlp_linear_hier2', activation='linear', is_train=is_training)
            "add&norm"
            output_hier2 += x
            output_hier2 = tf.contrib.layers.layer_norm(output_hier2)

            "hierarchical_rank3"
            hier3_logits = tf.matmul(x, hier3_relation_matrix, transpose_b=True)
            hier3_index = tf.nn.softmax(hier3_logits, -1)
            hier3_relation = tf.matmul(hier3_index, hier3_relation_matrix)
            "gate_hier3"
            concat_hier3 = tf.concat([x, hier3_relation], 1)
            alpha_hier3 = bn_dense_layer_v2(concat_hier3, self.hidden_size, True, scope='gate_hier3', activation='sigmoid', is_train=is_training)
            context_hier3 = alpha_hier3 * x + (1 - alpha_hier3) * hier3_relation
            "MLP linear"
            middle_hier3 = bn_dense_layer_v2(context_hier3, 1024, False, scope='mlp_activation_hier3', activation='relu', is_train=is_training)
            output_hier3 = bn_dense_layer_v2(middle_hier3, self.hidden_size, False, scope='mlp_linear_hier3', activation='linear', is_train=is_training)
            "add&norm"
            output_hier3 += x
            output_hier3 = tf.contrib.layers.layer_norm(output_hier3)

            output_hier = tf.concat([output_hier1, output_hier2, output_hier3], 1)
            prob_bag_hier3 = bn_dense_layer_v2(output_hier, 1, True, scope='self-attn-hier3', activation='linear', is_train=is_training) #->(bs, 1)

            tower_repre = []
            for i in range(FLAGS.batch_size):
                prob_hier3 = tf.nn.softmax(tf.reshape(prob_bag_hier3[self.scope[i]:self.scope[i+1]], [1, -1]))
                sen_hier3 = tf.reshape(tf.matmul(prob_hier3, output_hier[self.scope[i]:self.scope[i+1]]), [self.hidden_size*3])
                tower_repre.append(sen_hier3)
            stack_repre = tf.stack(tower_repre)

            fusion_repre = tf.layers.dropout(stack_repre, rate=1-self.keep_prob, training=is_training)

            with tf.variable_scope("loss", dtype=tf.float32, initializer=xavier()):
                discrimitive_matrix = self._GetVar(init_vec=None, key='discmat', name='discrimitive_matrix', initializer=tf.orthogonal_initializer(), shape=[FLAGS.num_classes, 3*self.hidden_size])
                bias = self._GetVar(init_vec=None, key='disc_bias', name='bias', shape=[FLAGS.num_classes])
                logits = tf.matmul(fusion_repre, discrimitive_matrix, transpose_b=True) + bias
                regularizer = tf.contrib.layers.l2_regularizer(FLAGS.weight_decay)
                l2_loss = tf.contrib.layers.apply_regularization(regularizer=regularizer,
                                                                 weights_list=tf.trainable_variables())
                n_hier1 = tf.cast(FLAGS.num_hier1_classes - 1, tf.float32)
                p_hier1 = 1.0 - 0.1
                q_hier1 = 0.1 / n_hier1
                soft_hier1 = tf.one_hot(tf.cast(self.sen_hier1, tf.int32), depth=FLAGS.num_hier1_classes, on_value=p_hier1, off_value=q_hier1)
                hier1_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=soft_hier1, logits=hier1_logits))

                n_hier2 = tf.cast(FLAGS.num_hier2_classes - 1, tf.float32)
                p_hier2 = 1.0 - 0.1
                q_hier2 = 0.1 / n_hier2
                soft_hier2 = tf.one_hot(tf.cast(self.sen_hier2, tf.int32), depth=FLAGS.num_hier2_classes, on_value=p_hier2, off_value=q_hier2)

                hier2_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=soft_hier2, logits=hier2_logits))
                n_hier3 = tf.cast(FLAGS.num_classes - 1, tf.float32)
                p_hier3 = 1.0 - 0.1
                q_hier3 = 0.1 / n_hier3
                soft_hier3 = tf.one_hot(tf.cast(self.label_index, tf.int32), depth=FLAGS.num_classes, on_value=p_hier3, off_value=q_hier3)
                hier3_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=soft_hier3, logits=hier3_logits))

                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=logits)) + hier3_loss + hier2_loss + hier1_loss + l2_loss
                self.output = tf.nn.softmax(logits)
                tf.summary.scalar('loss', self.loss)
                self.predictions = tf.argmax(logits, 1, name="predictions")
                self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

        if not is_training:
            self.test_output = self.output