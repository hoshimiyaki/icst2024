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
import copy

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

# 每行按段分割
def cut_data_full(data,line_len,word_len,vocab_d):
    matrix=data[1]

    #分词
    code=[]
    for line in data[0]:
        method = line.strip()
        tokens = javalang.tokenizer.tokenize(method)
        token_seq = []
        for token in tokens:
            if isinstance(token, javalang.tokenizer.String):
                tmp_token = ["stringliteral"]
            else:
                tmp_token = solve_camel_and_underline(token.value)
            token_seq += tmp_token
        code.append(token_seq)
    # print(code)
    # raise Exception

    if len(code)>line_len:
        f_line=-1
        for line in code:
            if "rank2fixstart" in line:
                f_line=code.index(line)
                break
        assert f_line!=-1

        pre_len=line_len//3*2
        pre_start=0 if f_line-pre_len<0 else f_line-pre_len
        code,matrix=code[pre_start:pre_start+line_len],matrix[pre_start:pre_start+line_len]
    for i in range(len(code)):
        if "rank2fixstart" not in code[i]:
            code[i]=code[i][:word_len]
        else:
            code[i]=code[i][:word_len-1]+[code[i][-1]] #add rank2fixend
        if len(code[i])<word_len:
            code[i]+=["PADDING"]*(word_len-len(code[i]))
    if len(code)<line_len:
        for i in range(line_len-len(code)):
            code.append(["PADDING"]*word_len)
            matrix.append([0.005]*7)
    try:
        assert len(code)==line_len
    except:
        print(len(code))
        raise Exception
    for line in code:
        assert len(line)==word_len
    raw_code=None #copy.deepcopy(code)
    for i in range(len(code)):
        for j in range(len(code[0])):
            code[i][j]=vocab_d[code[i][j]] if code[i][j] in vocab_d else vocab_d["OOV"]
    return [code,matrix],raw_code

def deal(l_s,l_t):
    with open('cooked/w2v/vocab_64.pkl','rb') as f:
        vocab=pickle.load(f)
    with open('cooked/w2v/vectors_64.pkl','rb') as f:
        vector=pickle.load(f)

    with open('cooked/squeeze_5e-3_all_{}_{}.pkl'.format(l_s,l_t),'rb') as f:
        data=pickle.load(f)

    n_data={}
    single=[17, 59, 15, 69, 44, 47, 64]
    for k in data:
        v=data[k]
        n_p=[]
        for item in v:
            code=item[0]
            metric=item[1]
            idx=0x3f3f3f
            for row in code:
                if 27 in row or row[:7]==single:
                    idx=code.index(row)
            if idx==0x3f3f3f:
                print(code)
                print(metric)
                raise Exception
            one_code=code[idx]
            one_metrix=metric[idx]
            code=[one_code]+code+[one_code]
            metric=[one_metrix]+metric+[one_metrix]
            n_p.append([code,metric])
        n_data[k]=n_p

    with open('./cooked/squeeze_{}_{}_new.pkl'.format(l_s,l_t),"wb") as f:
        pickle.dump(n_data,f)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo of argparse')
    # 2. 添加命令行参数
    parser.add_argument('--data', type=str,required=True)
    parser.add_argument('--method', type=str,default="all")
    parser.add_argument('--line-len', type=int, default=40)
    parser.add_argument('--word-len', type=int, default=20)
    parser.add_argument('--vector-len', type=int, default=64)
    # 3. 从命令行中结构化解析参数
    args = parser.parse_args()
    print(args)
    method=args.method
    origin_data=args.data
    line_len=args.line_len
    word_len=args.word_len
    
    fl_dataset_path = "./raw_data/code_score_120_{}.pkl".format(origin_data)
    vector_path="./cooked/w2v/vocab_{}.pkl".format(str(args.vector_len))

    with open(fl_dataset_path, "rb") as file:
        dataset_fl = pickle.load(file)
    with open(vector_path,"rb") as f:
        vocab_dict=pickle.load(f)
    
    dealed_dataset={}
    for p in dataset_fl:
        for i in range(len(dataset_fl[p])):
            data=dataset_fl[p][i]
            deal_data,_=cut_data_full(data,line_len,word_len,vocab_dict)
            if i==0:
                dealed_dataset[p]=[deal_data]
            else:
                dealed_dataset[p].append(deal_data)

    with open("./cooked/{}_{}_{}_{}.pkl".format(origin_data,method,line_len,word_len),"wb") as f:
        pickle.dump(dealed_dataset,f)

    deal(line_len,word_len)
    print("Done")
