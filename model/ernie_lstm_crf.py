# coding=utf-8
import torch.nn as nn
from torch.autograd import Variable
import torch
from transformers import AutoModel
from torchcrf import CRF

"""
使用的crf是pytorch-crf. 安装方式： pip install pytorch-crf. pip install transformers pip install fire pip install seqeval
PyTorch 1.10
pip3 install torch torchvision torchaudio -i https://pypi.mirrors.ustc.edu.cn/simple/
"""

class ERNIE_LSTM_CRF(nn.Module):
    """
    ernie_lstm_crf model
    """
    def __init__(self, ernie_config, tagset_size, embedding_dim, hidden_dim, rnn_layers, dropout_ratio, dropout1, use_cuda=False):
        super(ERNIE_LSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        //加载ERNIE
        self.word_embeds = AutoModel.from_pretrained(ernie_config)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=rnn_layers, bidirectional=True,
                            dropout=dropout_ratio, batch_first=True)
        self.rnn_layers = rnn_layers
        self.dropout1 = nn.Dropout(p=dropout1)
        self.crf = CRF(num_tags=tagset_size, batch_first=True)
        self.liner = nn.Linear(hidden_dim*2, tagset_size)
        self.tagset_size = tagset_size

    def rand_init_hidden(self, batch_size):
        """
        random initialize hidden variable
        """
        return Variable(torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)), \
               Variable(torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim))


    def forward(self, sentence, attention_mask=None):
        '''
        args:
            sentence (batch_size, word_seq_len) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf input (batch_size, word_seq_len, tag_size), hidden
        '''
        batch_size = sentence.size(0)
        seq_length = sentence.size(1)
        embeds = self.word_embeds(sentence, attention_mask=attention_mask)
        hidden = self.rand_init_hidden(batch_size)
        if embeds[0].is_cuda:
            hidden = tuple(i.cuda() for i in hidden)
        lstm_out, hidden = self.lstm(embeds[0], hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim*2)
        d_lstm_out = self.dropout1(lstm_out)
        l_out = self.liner(d_lstm_out)
        lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)
        return lstm_feats

    def loss(self, feats, mask, tags):
        """
        feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        :return:
        """
        loss_value = -self.crf(feats, tags, mask) # 计算损失
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value
