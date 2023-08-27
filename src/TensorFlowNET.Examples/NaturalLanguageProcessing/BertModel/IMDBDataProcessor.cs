using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Utils;
using Tensorflow.Keras.Datasets;
using System.Data;
using Tensorflow.NumPy;
using BERT;
using Tensorflow;
using HDF5CSharp;
using System.IO;
using System.ComponentModel.DataAnnotations;
using System.Net;

namespace BERT
{
    internal interface IMDBDataPreProcessor
    {
        public static (int[,], int[]) ProcessData(string path, int max_len, int label = 0)
        {
            string url = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt";
            string vocab_file = "./vocab.txt";
            { 
                using (WebClient client = new WebClient())
                {
                    byte[] fileData = client.DownloadData(url); 
                    using (Stream stream = File.Create(vocab_file))
                    {
                        stream.Write(fileData, 0, fileData.Length);
                    }
                }
            }
            BertTokenizer tokenizer = new BertTokenizer(vocab_file);
            DirectoryInfo root = new DirectoryInfo(path);
            FileInfo[] files = root.GetFiles();
            List<List<int>> res = new List<List<int>>();
            List<int> res_y = new List<int>();
            foreach (var item in files)
            {
                
                string str = File.ReadAllText(item.ToString());
                str = str.Replace("<br /><br />", " ");
                var ids = tokenizer.convert_tokens_to_ids(tokenizer._tokenize(str));
                ids = tokenizer.build_inputs_with_special_tokens(ids);
                if (ids.Count < max_len)
                {
                    var tmp = new List<int>(new int[max_len - ids.Count]);
                    for (int i = 0; i < tmp.Count; i++) tmp[i] = 0;
                    ids.extend(tmp);
                }
                else { ids = ids.GetRange(0, max_len); ids[ids.Count - 1] = 102; }
                res_y.Add(label);
                res.Add(ids);
            }

            var res_array = new int[res.Count, max_len];
            for (int i = 0; i < res.Count; i++)
                for (int j = 0; j < max_len; j++) res_array[i, j] = res[i][j];

            return (res_array, res_y.ToArray());
        }
    }
}
