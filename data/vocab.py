import os
import javalang
import pickle
from gensim.models.word2vec import Word2Vec
import numpy as np
import re
import string
import random
import sys
import argparse

def solve_camel_and_underline(token):
    if token.isdigit():
        return [token]
    else:
        p = re.compile(r'([a-z]|\d)([A-Z])')
        sub = re.sub(p, r'\1_\2', token).lower()
        sub_tokens = sub.split("_")
        tokens = re.sub(" +", " ", " ".join(sub_tokens)).strip()
        final_token = []
        for factor in tokens.split(" "):
            final_token.append(factor.rstrip(string.digits))
        return final_token


def cut_data(token_seq, token_length_for_reserve):
    if len(token_seq) <= token_length_for_reserve:
        return token_seq
    else:
        start_index = token_seq.index("rank2fixstart")
        end_index = token_seq.index("rank2fixend")
        assert end_index > start_index
        length_of_annotated_statement = end_index - start_index + 1
        if length_of_annotated_statement <= token_length_for_reserve:
            padding_length = token_length_for_reserve - length_of_annotated_statement
            # give 2/3 padding space to content before annotated statement
            pre_padding_length = padding_length // 3 * 2
            # give 1/3 padding space to content after annotated statement
            post_padding_length = padding_length - pre_padding_length
            if start_index >= pre_padding_length and len(token_seq) - end_index - 1 >= post_padding_length:
                return token_seq[start_index - pre_padding_length: end_index + 1 + post_padding_length]
            elif start_index < pre_padding_length:
                return token_seq[:token_length_for_reserve]
            elif len(token_seq) - end_index - 1 < post_padding_length:
                return token_seq[len(token_seq) - token_length_for_reserve:]
        else:
            return token_seq[start_index: start_index + token_length_for_reserve]


if __name__ == "__main__":
    
    fl_dataset_path = "./raw_data/src_code.pkl"
    output_data_dir = "./cooked/w2v/"
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    with open(fl_dataset_path, "rb") as file:
        dataset_fl = pickle.load(file)
    
    shuffle_seed=888
    vector_size = 64
    token_length_for_reserve = 400
    extend_size = 2
    max_vocab_size = 50000

    # split train/val/test parts and generate training corpus for word2vec pre-training
    
    w2v_training_corpus = []

    for tag in dataset_fl:
        for method in dataset_fl[tag]:
            method = method.strip()
            tokens = javalang.tokenizer.tokenize(method)
            token_seq = []
            for token in tokens:
                if isinstance(token, javalang.tokenizer.String):
                    tmp_token = ["stringliteral"]
                else:
                    tmp_token = solve_camel_and_underline(token.value)
                token_seq += tmp_token
            token_seq = cut_data(token_seq, token_length_for_reserve)
            w2v_training_corpus.append(token_seq)
    
    # Token vectors pre-training and saving
    print("Token vectors pre-training and saving")
    random.seed(shuffle_seed)
    random.shuffle(w2v_training_corpus)
    w2v = Word2Vec(w2v_training_corpus, size=vector_size, workers=16, sg=1, min_count=2, max_vocab_size=50000)
    w2v.save(os.path.join(output_data_dir, "w2v_{}".format(vector_size)))
    print("vocab_size: {}".format(len(w2v.wv.vocab)))
    vectors = w2v.wv.syn0
    extend_vectors = np.zeros([extend_size, vector_size], dtype="float32")  # extend_size = 2
    vectors = np.vstack([vectors, extend_vectors])
    vocab_list = list(w2v.wv.vocab.keys())
    vocab_dict = {}
    for token in vocab_list:
        vocab_dict[token] = w2v.wv.vocab[token].index
    del vocab_list
    
    # Initiating "OOV" and "PADDING" keyword
    print("Initiating \"OOV\" and \"PADDING\" keyword")
    if not ("OOV" in vocab_dict or "PADDING" in vocab_dict):
        vectors[len(vocab_dict)] = np.random.random(vector_size).astype("float32") * 2 - 1  # range (-1, 1)
        vocab_dict["OOV"] = len(vocab_dict)
        vectors[len(vocab_dict)] = np.array([0] * vector_size, dtype="float32")
        vocab_dict["PADDING"] = len(vocab_dict)
    else:
        print("The 2 keywords are not expected to exist in the current vocab.")
        exit(-1)
    
    # Saving vocab and vectors
    print("Saving vocab and vectors")
    with open(os.path.join(output_data_dir, "vocab_{}.pkl".format(str(vector_size))), "wb") as file:
        pickle.dump(vocab_dict, file)
    with open(os.path.join(output_data_dir, "vectors_{}.pkl".format(str(vector_size))), "wb") as file:
        pickle.dump(vectors, file)