using BERT;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Common.Types;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.NumPy;
using Tensorflow.Operations.Initializers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace BERT
{
    class BertEmbedding : Layer
    {
        BertConfig config;
        IVariableV1 weight;
        IVariableV1 token_type_embeddings;
        IVariableV1 position_embeddings;
        ILayer LayerNorm;
        ILayer dropout;


        public BertEmbedding(BertConfig config) : base(config)
        {
            this.config = config;
            LayerNorm = keras.layers.LayerNormalization(axis: -1, epsilon: config.layer_norm_eps);
            dropout = keras.layers.Dropout(config.hidden_dropout_prob);

            StackLayers(LayerNorm);
        }

        public override void build(KerasShapesWrapper input_shape)
        {
            tf_with(ops.name_scope("word_embeddings"), scope =>
            {
                weight = add_weight(name: "weight",
                                    shape: (config.vocab_size, config.hidden_size),
                                    initializer: new TruncatedNormal(config.initializer_range)
                    );

            });
            tf_with(ops.name_scope("token_type_embeddings"), scope =>
            {
                token_type_embeddings = add_weight(name: "token_type_embedding",
                                    shape: (config.type_vocab_size, config.hidden_size),
                                    initializer: new TruncatedNormal(config.initializer_range)
                    );

            });
            tf_with(ops.name_scope("position_embeddings"), scope =>
            {
                position_embeddings = add_weight(name: "position_embedding",
                                    shape: (config.max_position_embeddings, config.hidden_size),
                                    initializer: new TruncatedNormal(config.initializer_range)
                    );

            });

            base.build(input_shape);
        }
        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var input_ids = inputs[0];
            var position_ids = inputs[1];
            var token_type_ids = inputs[2];

            var inputs_embeds = tf.gather(weight.AsTensor(), input_ids);
            var position_embeds = tf.gather(position_embeddings.AsTensor(), indices: position_ids);
            var token_type_embeds = tf.gather(token_type_embeddings.AsTensor(), indices: token_type_ids);
            var final_embeddings = inputs_embeds + token_type_embeds + position_embeds;
            final_embeddings = LayerNorm.Apply(final_embeddings);
            final_embeddings = dropout.Apply(final_embeddings, training: training ?? false);
            return final_embeddings;
        }
    }

    class BertSelfOutput : Layer
    {
        BertConfig config;
        ILayer dense;
        ILayer LayerNorm;
        ILayer dropout;
        public BertSelfOutput(BertConfig config) : base(config)
        {
            this.config = config;
            dense = keras.layers.Dense(
                units: config.hidden_size, kernel_initializer: new TruncatedNormal(config.initializer_range));
            LayerNorm = keras.layers.LayerNormalization(axis: -1, epsilon: config.layer_norm_eps);
            dropout = keras.layers.Dropout(config.hidden_dropout_prob);

            StackLayers(dense, LayerNorm);
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var hidden_states = inputs[0];
            var input_tensor = inputs[1];

            hidden_states = dense.Apply(inputs: hidden_states);
            hidden_states = dropout.Apply(inputs: hidden_states, training: training ?? false);

            hidden_states = LayerNorm.Apply(tf.add(hidden_states, input_tensor));

            return hidden_states;
        }
    }

    class BertAttention : Layer
    {
        BertConfig config;
        BertSelfAttention self_attention;
        BertSelfOutput dense_output;
        public BertAttention(BertConfig config) : base(config)
        {
            this.config = config;
            self_attention = new BertSelfAttention(config);
            dense_output = new BertSelfOutput(config);
            StackLayers(self_attention, dense_output);
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var input_tensor = inputs[0];
            var attention_mask = inputs[1];

            var self_outputs = self_attention.Apply(new Tensor[] { input_tensor, attention_mask }, training: training ?? false);
            var attention_output = dense_output.Apply(new Tensor[] { self_outputs, input_tensor }, training: training ?? false);

            return attention_output;
        }
    }

    class BertEncoder : Layer
    {
        BertConfig config;
        List<ILayer> layers;
        public BertEncoder(BertConfig config) : base(config)
        {
            this.config = config;
            layers = new List<ILayer>();
            for (int i = 0; i < config.num_hidden_layers; i++) layers.Add(new BertLayer(config));
            StackLayers(layers.ToArray());
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var hidden_states = inputs[0];
            var attention_mask = inputs[1];

            foreach (var layer in layers)
            {
                var layer_outputs = layer.Apply(new Tensor[] { hidden_states, attention_mask });
                hidden_states = layer_outputs;
            }

            return hidden_states;
        }

    }

    class BertIntermediate : Layer
    {
        BertConfig config;
        ILayer dense;
        public BertIntermediate(BertConfig config) : base(config)
        {
            this.config = config;
            dense = keras.layers.Dense(
                units: config.intermediate_size, kernel_initializer: new TruncatedNormal(config.initializer_range));
            StackLayers(dense);
        }
        public static Tensors gelu(Tensor x)
        {
            var cdf = 0.5 * (1.0 + tf.tanh(
                (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))));
            return x * cdf;
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var hidden_states = inputs;
            hidden_states = dense.Apply(hidden_states);
            hidden_states = gelu(hidden_states);
            return hidden_states;
        }
    }

    class BertOutput : Layer
    {
        ILayer dense;
        ILayer LayerNorm;
        ILayer dropout;
        public BertOutput(BertConfig config) : base(config)
        {
            dense = keras.layers.Dense(
                units: config.hidden_size, kernel_initializer: new TruncatedNormal(config.initializer_range));
            LayerNorm = keras.layers.LayerNormalization(axis: -1, epsilon: config.layer_norm_eps);
            dropout = keras.layers.Dropout(config.hidden_dropout_prob);

            StackLayers(dense, LayerNorm);

        }
        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var hidden_states = inputs[0];
            var input_tensor = inputs[1];

            hidden_states = dense.Apply(new Tensor[] { hidden_states });
            hidden_states = dropout.Apply(new Tensor[] { hidden_states }, training: training ?? false);

            hidden_states = LayerNorm.Apply(tf.add(hidden_states, input_tensor));

            return hidden_states;
        }
    }

    class BertSelfAttention : Layer
    {
        BertConfig config;
        int attention_head_size;
        int all_head_size;
        double sqrt_att_head_size;
        ILayer query;
        ILayer key;
        ILayer value;
        ILayer dropout;
        public BertSelfAttention(BertConfig config) : base(config)
        {
            this.config = config;
            attention_head_size = config.hidden_size / config.num_attention_heads;
            all_head_size = config.num_attention_heads * attention_head_size;
            sqrt_att_head_size = Math.Sqrt(attention_head_size);
            query = keras.layers.Dense(units: all_head_size, kernel_initializer: new TruncatedNormal(config.initializer_range));
            key = keras.layers.Dense(units: all_head_size, kernel_initializer: new TruncatedNormal(config.initializer_range));
            value = keras.layers.Dense(units: all_head_size, kernel_initializer: new TruncatedNormal(config.initializer_range));
            dropout = keras.layers.Dropout(config.hidden_dropout_prob);

            StackLayers(query, key, value);

        }

        public Tensor transpose_for_scores(Tensor tensor, int batch_size)
        {
            tensor = tf.reshape(tensor: tensor, shape: (batch_size, -1, config.num_attention_heads, attention_head_size));
            return tf.transpose(tensor, perm: new int[] { 0, 2, 1, 3 });
        }
        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var hidden_states = inputs[0];
            var attention_mask = inputs[1];

            var batch_size = hidden_states.shape[0];
            var seq_len = hidden_states.shape[1];

            var mixed_query_layer = query.Apply(inputs: hidden_states);
            var key_layer = transpose_for_scores(key.Apply(inputs: hidden_states), (int)batch_size);
            var value_layer = transpose_for_scores(value.Apply(inputs: hidden_states), (int)batch_size);

            var query_layer = transpose_for_scores(mixed_query_layer, (int)batch_size);

            var attention_scores = tf.reshape(tf.batch_matmul(
                tf.reshape(query_layer, (batch_size * config.num_attention_heads, seq_len, attention_head_size)),
                tf.reshape(
                    tf.transpose(key_layer, new int[] { 0, 1, 3, 2 }),
                (batch_size * config.num_attention_heads, attention_head_size, seq_len))),
                (batch_size, config.num_attention_heads, seq_len, seq_len));

            var dk = tf.cast(np.array(sqrt_att_head_size), dtype: attention_scores.dtype);

            attention_scores = tf.divide(attention_scores, dk);
            attention_scores = tf.add(attention_scores, attention_mask);

            var attention_probs = tf.nn.softmax(logits: attention_scores, axis: -1);
            attention_probs = dropout.Apply(inputs: attention_probs, training: training ?? false);

            var attention_output = tf.reshape(
                tf.batch_matmul(tf.reshape(attention_probs, (batch_size * config.num_attention_heads, seq_len, seq_len)),
                tf.reshape(value_layer, (batch_size * config.num_attention_heads, seq_len, attention_head_size))),
                (batch_size, config.num_attention_heads, seq_len, attention_head_size));

            attention_output = tf.transpose(attention_output, perm: new int[] { 0, 2, 1, 3 });
            attention_output = tf.reshape(attention_output, (batch_size, -1, all_head_size));

            return attention_output;
        }
    }

    class BertLayer : Layer
    {
        BertConfig config;
        BertAttention attention;
        BertIntermediate intermediate;
        BertOutput bert_output;
        public BertLayer(BertConfig config) : base(config)
        {
            this.config = config;
            attention = new BertAttention(config);
            intermediate = new BertIntermediate(config);
            bert_output = new BertOutput(config);
            StackLayers(attention, intermediate, bert_output);
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var hidden_states = inputs[0];
            var attention_mask = inputs[1];

            var attention_output = attention.Apply(new Tensor[] { hidden_states, attention_mask }, training: training ?? false);

            var intermediate_output = intermediate.Apply(attention_output, training: training ?? false);
            var layer_output = bert_output.Apply(new Tensor[] { intermediate_output, attention_output }, training: training ?? false);
            return layer_output;
        }
    }

    class BertPooler : Layer
    {
        BertConfig config;
        ILayer dense;
        public BertPooler(BertConfig config) : base(config)
        {
            this.config = config;
            dense = keras.layers.Dense(
                units: config.hidden_size,
                kernel_initializer: new TruncatedNormal(config.initializer_range),
                activation: keras.activations.Tanh);
            StackLayers(dense);

        }

        public static Tensors gelu(Tensor x)
        {
            var cdf = 0.5 * (1.0 + tf.tanh(
                (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))));
            return x * cdf;
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var hidden_states = inputs[0];

            var shape = hidden_states.shape;
            var first_token_tensor = tf.slice<int, int>(hidden_states, new int[] { 0, 0, 0 }, new int[] { (int)shape[0], 1, (int)shape[2] });

            return gelu(dense.Apply(tf.reshape(first_token_tensor, (shape[0], shape[2]))));
        }
    }
    class BertMainLayer : Layer
    {
        BertConfig config;
        BertEmbedding embeddings;
        BertEncoder encoder;
        BertPooler pooler;
        public BertMainLayer(BertConfig config) : base(config)
        {
            this.config = config;
            embeddings = new BertEmbedding(config);
            encoder = new BertEncoder(config);
            pooler = new BertPooler(config);
            StackLayers(embeddings,encoder, pooler);
        }

        public Tensors get_other_ids(Tensor input_ids)
        {
            var batch_size = input_ids.shape[0];
            var attention_mask = tf.reshape(tf.cast(tf.fill(input_ids.shape, 1), dtype: tf.int32), (batch_size, -1));
            var token_type_ids = tf.reshape(tf.cast(tf.fill(input_ids.shape, 0), dtype: tf.int32), (batch_size, -1));
            var position_ids = tf.expand_dims(tf.range(0, input_ids.shape[1]), axis: 0);
            return new Tensor[] { input_ids, attention_mask, token_type_ids, position_ids };
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            if(inputs.Length == 1) inputs = get_other_ids(inputs[0]);

            var input_ids = inputs[0];
            var attention_mask = inputs[1];
            var token_type_ids = inputs[2];
            var position_ids = inputs[3];

            var input_shape = input_ids.shape; //bsz seq_len dim
            var embedding_output = embeddings.Apply(new Tensor[] { input_ids, position_ids, token_type_ids });

            var attention_mask_shape = attention_mask.shape;

            var extended_attention_mask = tf.reshape(
                attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1]));

            extended_attention_mask = tf.cast(extended_attention_mask, dtype: embedding_output.dtype);

            var one_cst = tf.constant(1.0, dtype: embedding_output.dtype);

            var ten_thousand_cst = tf.constant(-10000.0, dtype: embedding_output.dtype);
            extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst);

            var encoder_outputs = encoder.Apply(new Tensor[] { embedding_output, extended_attention_mask });

            var pooled_output = pooler.Apply(encoder_outputs);
            return pooled_output;
        }
    }
}
