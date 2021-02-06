import numpy as np
import tensorflow as tf
# import tensorflow_text as text
print(tf.__version__)

docs = tf.constant([u'Everything not saved will be lost.'])
tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(docs)
f1 = text.wordshape(tokens, text.WordShape.HAS_TITLE_CASE)
bigrams = text.ngrams(tokens, 2, reduction_type=text.Reduction.STRING_JOIN)

print(docs)
a = input()