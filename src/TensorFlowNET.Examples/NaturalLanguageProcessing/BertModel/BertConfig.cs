using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.Keras.ArgsDefinition;

namespace BERT
{
    class BertConfig : LayerArgs
    {
        public int vocab_size;
        public int hidden_size;
        public int num_hidden_layers;
        public int num_attention_heads;
        public int intermediate_size;
        public string hidden_act;
        public float hidden_dropout_prob;
        public float attention_probs_dropout_prob;
        public int max_position_embeddings;
        public int type_vocab_size;
        public float initializer_range;
        public float layer_norm_eps;
        public int pad_token_id;
        public string position_embedding_type;
        public BertConfig(int vocab_size = 30522,
                          int hidden_size = 768,
                          int num_hidden_layers = 12,
                          int num_attention_heads = 12,
                          int intermediate_size = 3072,
                          string hidden_act = "gelu",
                          double hidden_dropout_prob = 0.1,
                          double attention_probs_dropout_prob = 0.1,
                          int max_position_embeddings = 512,
                          int type_vocab_size = 2,
                          double initializer_range = 0.02,
                          double layer_norm_eps = 1e-12,
                          int pad_token_id = 0,
                          string position_embedding_type = "absolute")
        {
            this.vocab_size = vocab_size;
            this.hidden_size = hidden_size;
            this.num_hidden_layers = num_hidden_layers;
            this.num_attention_heads = num_attention_heads;
            this.intermediate_size = intermediate_size;
            this.hidden_act = hidden_act;
            this.hidden_dropout_prob = (float)hidden_dropout_prob;
            this.attention_probs_dropout_prob = (float)attention_probs_dropout_prob;
            this.max_position_embeddings = max_position_embeddings;
            this.type_vocab_size = type_vocab_size;
            this.initializer_range = (float)initializer_range;
            this.layer_norm_eps = (float)layer_norm_eps;
            this.pad_token_id = pad_token_id;
            this.position_embedding_type = position_embedding_type;

        }
    }
}
