﻿using Tensorflow;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Examples.Text
{
    public class CharCnn : ITextModel
    {
        public CharCnn(int alphabet_size, int document_max_len, int num_class)
        {
            var learning_rate = 0.001f;
            var filter_sizes = new int[] { 7, 7, 3, 3, 3, 3 };
            var num_filters = 256;
            var kernel_initializer = tf.truncated_normal_initializer(stddev: 0.05f);

            var x = tf.placeholder(tf.int32, new TensorShape(-1, document_max_len), name: "x");
            var y = tf.placeholder(tf.int32, new TensorShape(-1), name: "y");
            var is_training = tf.placeholder(tf.@bool, new TensorShape(), name: "is_training");
            var global_step = tf.Variable(0, trainable: false);
            var keep_prob = tf.where(is_training, 0.5f, 1.0f);

            var x_one_hot = tf.one_hot(x, alphabet_size);
            var x_expanded = tf.expand_dims(x_one_hot, -1);

            // ============= Convolutional Layers =============
            Tensor pool1 = null, pool2 = null;
            Tensor conv3 = null, conv4 = null, conv5 = null, conv6 = null;
            Tensor h_pool = null;

            tf_with(tf.name_scope("conv-maxpool-1"), delegate
            {
                var conv1 = keras.layers.Conv2D(
                    filters: num_filters,
                    kernel_size: new[] { filter_sizes[0], alphabet_size },
                    kernel_initializer: kernel_initializer,
                    activation: tf.nn.relu).Apply(x_expanded);

                pool1 = keras.layers.max_pooling2d(conv1,
                    pool_size: new[] { 3, 1 },
                    strides: new[] { 3, 1 });
                pool1 = tf.transpose(pool1, new[] { 0, 1, 3, 2 });
            });

            tf_with(tf.name_scope("conv-maxpool-2"), delegate
            {
                var conv2 = keras.layers.Conv2D(
                    filters: num_filters,
                    kernel_size: new[] { filter_sizes[1], num_filters },
                    kernel_initializer: kernel_initializer,
                    activation: tf.nn.relu).Apply(pool1);

                pool2 = keras.layers.max_pooling2d(conv2,
                    pool_size: new[] { 3, 1 },
                    strides: new[] { 3, 1 });
                pool2 = tf.transpose(pool2, new[] { 0, 1, 3, 2 });
            });

            tf_with(tf.name_scope("conv-3"), delegate
            {
                conv3 = keras.layers.Conv2D(
                    filters: num_filters,
                    kernel_size: new[] { filter_sizes[2], num_filters },
                    kernel_initializer: kernel_initializer,
                    activation: tf.nn.relu).Apply(pool2);
                conv3 = tf.transpose(conv3, new[] { 0, 1, 3, 2 });
            });

            tf_with(tf.name_scope("conv-4"), delegate
            {
                conv4 = keras.layers.Conv2D(
                    filters: num_filters,
                    kernel_size: new[] { filter_sizes[3], num_filters },
                    kernel_initializer: kernel_initializer,
                    activation: tf.nn.relu).Apply(conv3);
                conv4 = tf.transpose(conv4, new[] { 0, 1, 3, 2 });
            });

            tf_with(tf.name_scope("conv-5"), delegate
            {
                conv5 = keras.layers.Conv2D(
                    filters: num_filters,
                    kernel_size: new[] { filter_sizes[4], num_filters },
                    kernel_initializer: kernel_initializer,
                    activation: tf.nn.relu).Apply(conv4);
                conv5 = tf.transpose(conv5, new[] { 0, 1, 3, 2 });
            });

            tf_with(tf.name_scope("conv-maxpool-6"), delegate
            {
                conv6 = keras.layers.Conv2D(
                    filters: num_filters,
                    kernel_size: new[] { filter_sizes[5], num_filters },
                    kernel_initializer: kernel_initializer,
                    activation: tf.nn.relu).Apply(conv5);

                var pool6 = keras.layers.max_pooling2d(conv6,
                    pool_size: new[] { 3, 1 },
                    strides: new[] { 3, 1 });
                pool6 = tf.transpose(pool6, new[] { 0, 2, 1, 3 });

                h_pool = tf.reshape(pool6, new[] { -1, 34 * num_filters });
            });

            // ============= Fully Connected Layers =============
            Tensor fc2_out = null;
            Tensor logits = null;
            Tensor predictions = null;

            tf_with(tf.name_scope("fc-1"), delegate
            {
                /*fc1_out = tf.layers.dense(h_pool,
                    1024,
                    activation: tf.nn.relu(),
                    kernel_initializer: kernel_initializer);*/
            });

            tf_with(tf.name_scope("fc-2"), delegate
            {
                /*fc2_out = tf.layers.dense(fc1_out,
                    1024,
                    activation: tf.nn.relu(),
                    kernel_initializer: kernel_initializer);*/
            });

            tf_with(tf.name_scope("fc-3"), delegate
            {
                logits = keras.layers.dense(fc2_out,
                    num_class,
                    kernel_initializer: kernel_initializer);
                predictions = tf.argmax(logits, -1, output_type: tf.int32);
            });

            tf_with(tf.name_scope("loss"), delegate
            {
                var y_one_hot = tf.one_hot(y, num_class);
                var loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits: logits, labels: y_one_hot));
                var optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step: global_step);
            });

            tf_with(tf.name_scope("accuracy"), delegate
            {
                var correct_predictions = tf.equal(predictions, y);
                var accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name: "accuracy");
            });
        }
    }
}
