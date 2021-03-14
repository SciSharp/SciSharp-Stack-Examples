# https://confusedcoders.com/data-science/deep-learning/how-to-build-deep-neural-network-for-custom-ner-with-keras
# https://www.kaggle.com/nikkisharma536/ner-with-bilstm-and-crf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd
df = pd.read_csv("/kaggle/input/entity-annotated-corpus/ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)
df.head()