using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Tensorflow;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Layers;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples.ImageProcessing.YOLO
{
    class common
    {
        public static Tensor convolutional(Tensor input_layer, TensorShape filters_shape,
            bool downsample = false, bool activate = true,
            bool bn = true)
        {
            int strides;
            string padding;

            if (downsample)
            {
                (int pad_h, int pad_w) = ((int)Math.Floor((filters_shape[0] - 2) / 2.0f) + 1, (int)Math.Floor((filters_shape[1] - 2) / 2.0f) + 1);
                var paddings = tf.constant(new int[,] { { 0, 0 }, { pad_h, pad_h }, { pad_w, pad_w }, { 0, 0 } });
                input_layer = tf.pad(input_layer, paddings, "CONSTANT");
                strides = 2;
                padding = "valid";
            }
            else
            {
                strides = 1;
                padding = "same";
            }

            var conv2d_layer = tf.keras.layers.Conv2D(filters_shape[-1],
                kernel_size: filters_shape[0],
                strides: strides,
                padding: padding,
                use_bias: !bn,
                kernel_regularizer: tf.keras.regularizers.l2(0.0005f),
                kernel_initializer: tf.random_normal_initializer(stddev: 0.01f),
                bias_initializer: tf.constant_initializer(0f));
            var conv = conv2d_layer.Apply(input_layer);
            if (bn)
            {
                var batch_layer = new BatchNormalization(new BatchNormalizationArgs
                {
                });
                conv = batch_layer.Apply(conv);
            }

            if (activate)
                conv = tf.nn.leaky_relu(conv, alpha: 0.1f);

            return conv;
        }

        public static Tensor upsample(Tensor input_data, string name, string method = "deconv")
        {
            Debug.Assert(new[] { "resize", "deconv" }.Contains(method));
            Tensor output = null;
            if (method == "resize")
            {
                tf_with(tf.variable_scope(name), delegate
                {
                    var input_shape = tf.shape(input_data);
                    output = tf.image.resize_nearest_neighbor(input_data, new Tensor[] { input_shape[1] * 2, input_shape[2] * 2 });
                });
            }
            else if(method == "deconv")
            {
                throw new NotImplementedException("upsample.deconv");
            }

            return output;
        }

        public static Tensor residual_block(Tensor input_data, int input_channel, int filter_num1, 
            int filter_num2, string name)
        {
            var short_cut = input_data;

            return tf_with(tf.variable_scope(name), scope =>
            {
                input_data = convolutional(input_data, (1, 1, input_channel, filter_num1));
                input_data = convolutional(input_data, (3, 3, filter_num1, filter_num2));

                var residual_output = input_data + short_cut;

                return residual_output;
            });
        }
    }
}
