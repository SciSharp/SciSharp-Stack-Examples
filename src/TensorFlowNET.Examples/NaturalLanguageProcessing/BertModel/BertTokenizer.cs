using Newtonsoft.Json.Linq;
using OneOf.Types;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using static System.Net.Mime.MediaTypeNames;

namespace BERT
{
    class BasicTokenizer
    {
        bool do_lower_case;
        bool strip_accents;
        public List<string> never_split;
        public BasicTokenizer(bool do_lower_case, bool strip_accents = true, List<string> never_split = null)
        {
            this.do_lower_case = do_lower_case;
            this.strip_accents = strip_accents;
            this.never_split = never_split == null ? new List<string>() : never_split;
        }

        public static bool _is_control(char c)
        {
            //Checks whether `char` is a control character.
            if (c == '\t' || c == '\n' || c == '\r') return false;
            UnicodeCategory cat = CharUnicodeInfo.GetUnicodeCategory(c);
            if (cat.ToString().StartsWith("Other")) return true;
            return false;
        }

        public static bool _is_whitespace(char c)
        {
            //Checks whether `char` is a whitespace character.
            if (c == '\t' || c == '\n' || c == '\r') return false;
            UnicodeCategory cat = CharUnicodeInfo.GetUnicodeCategory(c);
            if (cat.ToString() == "SpaceSeparator") return true;
            return false;
        }

        public static bool _is_punctuation(char c)
        {
            int cp = (int)c;
            if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
            UnicodeCategory cat = CharUnicodeInfo.GetUnicodeCategory(c);
            if (cat.ToString().Contains("Punctuation")) return true;
            return false;
        }

        public static List<string> whitespace_tokenize(string text)
        {
            //Runs basic whitespace cleaning and splitting on a piece of text.
            text = text.Trim();
            var tokens = text.Split(' ');
            return new List<string>(tokens);
        }
        public string _run_strip_accents(string text)
        {
            //Strips accents from a piece of text.
            // Normalize(text); // Not importance.
            List<char> output = new List<char>();
            foreach (var c in text)
            {
                var cat = CharUnicodeInfo.GetUnicodeCategory(c);
                if (cat == UnicodeCategory.NonSpacingMark) continue;
                output.Add(c);
            }
            return string.Join("", output);
        }

        public List<string> _run_split_on_punc(string text, List<string> never_split = null)
        {
            //Splits punctuation on a piece of text.
            if (never_split != null && never_split.Contains(text)) return new List<string>(new string[] { text });
            char[] chars = text.ToArray();
            int i = 0;
            bool start_new_word = true;
            List<List<char>> output = new List<List<char>>();
            while (i < chars.Length)
            {
                char chr = chars[i];
                if (_is_punctuation(chr))
                {
                    var tmp = new List<char>();
                    tmp.Add(chr);
                    output.Add(new List<char>(tmp));
                    start_new_word = true;
                }
                else
                {
                    if (start_new_word) output.Add(new List<char>());
                    start_new_word = false;
                    output[output.Count - 1].Add(chr);
                }
                i++;
            }
            var res = new List<string>();
            foreach (var x in output) res.Add(string.Join("", x));
            return res;
        }

        public List<string> tokenize(string text, List<string> never_split = null)
        {
            text = _clean_text(text);
            var orig_tokens = whitespace_tokenize(text);
            List<string> split_tokens = new List<string>();
            foreach (string token in orig_tokens)
            {
                var _token = token;
                if (!never_split.Contains(token))
                {
                    if (do_lower_case)
                    {
                        _token = _token.ToLower();
                        if (this.strip_accents != false)
                        {
                            _token = this._run_strip_accents(_token);
                        }
                    }
                    else if (strip_accents)
                    {
                        _token = this._run_strip_accents(_token);
                    }
                    split_tokens.extend(this._run_split_on_punc(_token, never_split));
                }
            }
            var output_tokens = whitespace_tokenize(string.Join(" ", split_tokens));
            return output_tokens;
        }

        protected string _clean_text(string text)
        {
            //Performs invalid character removal and whitespace cleanup on text.
            var output = new List<char>();
            foreach (char c in text)
            {
                int cp = (int)c;
                if (cp == 0 || cp == 0xFFFD || _is_control(c)) continue;
                if (_is_whitespace(c)) output.Add(' ');
                else output.Add(c);
            }
            return string.Join("", output);
        }
    }
    internal class BertTokenizer
    {
        Dictionary<string, int> vocab;
        Dictionary<int, string> ids_to_tokens;
        bool do_lower_case;
        bool do_basic_tokenize;
        List<string> never_split;
        BasicTokenizer basic_tokenizer;
        List<string> all_special_tokens;
        public string cls_token;
        public string pad_token;
        public string sep_token;
        public string mask_token;
        public string unk_token;
        public int vocab_size {
            get {
                return vocab.Count;
            }
        }
        public static Dictionary<string, int> load_vocab(string vocab_file) {
            // Loads a vocabulary file into a dictionary.
            Dictionary<string, int> vocab = new Dictionary<string, int>();
            StreamReader sr = new StreamReader(vocab_file);
            int index = 0;
            while (!sr.EndOfStream) {
                string token = sr.ReadLine();
                token.TrimEnd(new char[] { '\n'});
                vocab.Add(token, index++);
            }
            return vocab;
        }

        public static List<string> whitespace_tokenize(string text) {
            //Runs basic whitespace cleaning and splitting on a piece of text.
            text = text.Trim();
            var token = text.Split(' ');
            return new List<string>(token);
        }

        public BertTokenizer(
            string vocab_file,
            bool do_lower_case = true,
            bool do_basic_tokenize = true,
            List<string> never_split = null,
            string unk_token = "[UNK]",
            string sep_token = "[SEP]",
            string pad_token = "[PAD]",
            string cls_token = "[CLS]",
            string mask_token = "[MASK]"
            )
        {
            this.vocab = load_vocab(vocab_file);
            this.ids_to_tokens = new Dictionary<int, string>();
            foreach (KeyValuePair<string, int> entry in this.vocab) {
                this.ids_to_tokens.Add(entry.Value, entry.Key);
            }
            this.do_lower_case = do_lower_case;
            this.do_basic_tokenize = do_basic_tokenize;
            this.never_split = never_split == null ? new List<string>() : never_split;
            if (do_basic_tokenize) this.basic_tokenizer = new BasicTokenizer(do_lower_case, never_split:never_split);
            this.all_special_tokens = new List<string>(new string[] { unk_token, sep_token, pad_token, cls_token, mask_token });
            this.unk_token = unk_token;
            this.cls_token = cls_token;
            this.mask_token = mask_token;
            this.pad_token = pad_token;
            this.sep_token = sep_token;
        }

        protected int _convert_token_to_id(string token) {
            //Converts a token (str) in an id using the vocab.
            return this.vocab[token];
        }
        protected string _convert_id_to_token(int id) {
            //Converts an index (integer) in a token (str) using the vocab.
            return this.ids_to_tokens[id];
        }

        public List<int> convert_tokens_to_ids(List<string> tokens) {
            List<int> ids = new List<int>();

            foreach (var token in tokens) {
                if (this.vocab.ContainsKey(token)) ids.Add(this.vocab[token]);
                else ids.Add(this.vocab[this.unk_token]);
            }
            return ids;
        }
        public List<int> build_inputs_with_special_tokens(List<int> token_ids_0, List<int> token_ids_1=null) {
            /*
            Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
            adding special tokens. A BERT sequence has the following format:

            - single sequence: `[CLS] X [SEP]`
            - pair of sequences: `[CLS] A [SEP] B [SEP]`

            Args:
                token_ids_0 (`List[int]`):
                    List of IDs to which the special tokens will be added.
                token_ids_1 (`List[int]`, *optional*):
                    Optional second list of IDs for sequence pairs.

            Returns:
                `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
            */
            var token_ids = new List<int>(token_ids_0.ToArray());
            //Console.WriteLine(cls_token);
            token_ids.Insert(0, vocab[cls_token]);
            token_ids.Add(vocab[sep_token]);
            if (token_ids_1 == null) return token_ids;

            token_ids.extend(token_ids_1);
            token_ids.Add(vocab[sep_token]);
            return token_ids;

        }

        public List<int> create_token_type_ids_from_sequences(List<int> token_ids_0, List<int> token_ids_1 = null) {
            /*Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
          */
            var token_ids_0_array = new int[token_ids_0.Count + 2];
            Array.Fill(token_ids_0_array, 0);
            var token_ids_0_list = new List<int>(token_ids_0_array);

            if (token_ids_1 == null) return token_ids_0_list;

            var token_ids_1_array = new int[token_ids_1.Count + 1];
            Array.Fill(token_ids_1_array, 1);
            var token_ids_1_list = new List<int>(token_ids_1_array);

            token_ids_0_list.extend(token_ids_1_list);
            return token_ids_0_list;
        }

        public List<string> _tokenize(string text) {
            List<string> split_tokens = new List<string>();
            if (do_basic_tokenize)
            {
                foreach (var token in basic_tokenizer.tokenize(text, all_special_tokens))
                {
                    if (this.basic_tokenizer.never_split.Contains(token))
                    {
                        split_tokens.Add(token);
                    }
                    else split_tokens.extend(wordpiece_tokenize(token));
                }
            }
            else split_tokens = wordpiece_tokenize(text);
            return split_tokens;
        }

        public List<string> wordpiece_tokenize(string text, int max_input_chars_per_word = 100) {
            /*Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.*/
            List<string> output_tokens = new List<string>();
            foreach (var token in whitespace_tokenize(text))
            {
                List<char> chars = new List<char>(token.ToCharArray());
                if (chars.Count > max_input_chars_per_word) {
                    output_tokens.Add(unk_token);
                    continue;
                }
                bool is_bad = false;
                int start = 0;
                List<string> sub_tokens = new List<string>();
                while (start < chars.Count) {
                    int end = chars.Count;
                    string cur_substr = "";
                    while (start < end) {
                        string substr = string.Join("", chars.GetRange(start, end - start));
                        if (start > 0) substr = "##" + substr;
                        if (vocab.ContainsKey(substr)) {
                            cur_substr = substr;
                            break;
                        }
                        end -= 1;
                    }
                    if (cur_substr.Length == 0) { is_bad = true; break; }
                    sub_tokens.Add(cur_substr);
                    start = end;
                }
                if (is_bad) output_tokens.Add(unk_token);
                else output_tokens.extend(sub_tokens);
            }
            return output_tokens;
        }



    }
}
