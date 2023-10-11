from torch import nn
import torch
import torch.nn.functional as F
import math
from math import ceil

def d_attention(query,keys,values, dropout=None):
        d_k = keys.size(-1)
        scores = torch.matmul(query, keys.transpose(-2, -1)) /math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        fea  = torch.matmul(p_attn, values)
        return fea, p_attn
		
# stmt-lstm: cnn; block-lstm:mean
class ConprehenLSTM(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size,num_layers,cnn_channel,l_b,l_s, pretrained_weight=None):
		super(ConprehenLSTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.label_size = label_size
		self.activation = torch.tanh
		self.num_layers = num_layers

		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		#if pretrained_weight is not None:
		#	self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
		self.embedding.weight.requires_grad = True
		self.row_encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=self.num_layers, bidirectional=True, batch_first=True)
		self.cnn=nn.Conv2d(1,cnn_channel,[1,hidden_dim*2])
		self.block_encoder = nn.LSTM(input_size=l_s*cnn_channel, hidden_size=hidden_dim, num_layers=self.num_layers, bidirectional=True, batch_first=True)
		self.decoder = nn.Linear(hidden_dim * 2, self.label_size)

	def forward(self, code,matrix,_place):

		embeddings = self.embedding(code) #[batch,line,word,embedding_dim]
		origin_shape=embeddings.shape
		embeddings=torch.reshape(embeddings,[origin_shape[0]*origin_shape[1],-1,origin_shape[-1]]) #[batch*line,word,embedding]
		lstm_out1, _ = self.row_encoder(embeddings)  # [batch*line,word,hidden*2]
		lstm_out1=lstm_out1.unsqueeze(1)
		cnn_out1 = self.cnn(lstm_out1) # [batch*line,word]
		cnn_shape=cnn_out1.shape
		cnn_out1=torch.reshape(cnn_out1,[cnn_shape[0],-1])
		input2=torch.reshape(cnn_out1,[origin_shape[0],origin_shape[1],-1]) #[batch,line,word]
		attn_output,_=d_attention(matrix,matrix,input2)
		lstm_out2, _ = self.block_encoder(attn_output)  # [batch,line,hidden*2]
		o1 = lstm_out2[:,0]
		o2=lstm_out2[:,-1]
		o_all=torch.stack((o1,o2),dim=1)
		out2 = torch.mean(o_all,dim=1)
		out = self.decoder(out2)
		return out
	