# coding=utf-8

# 输入一段话，识别对应的实体

from config import Config
from utils import load_vocab, load_model, InputFeatures
from model.ernie_lstm_crf import ERNIE_LSTM_CRF
import torch
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def load_ner_model(config, tagset_size):
    model = ERNIE_LSTM_CRF(config.ernie_path, tagset_size, config.bert_embedding, config.rnn_hidden, config.rnn_layer,
                          dropout_ratio=config.dropout_ratio, dropout1=config.dropout1, use_cuda=config.use_cuda)
    model = load_model(model)
    if config.use_cuda:
        model.cuda()
    return model

config = Config()
label_dic = load_vocab(config.label_file)  # {tag: index}
vocab = load_vocab(config.vocab)
model = load_ner_model(config, len(label_dic))

def encoder_corpus(sentences, max_length, vocab):
    if isinstance(sentences, str):
        sentences = [sentences]
    result = []
    for line in sentences:
        text = line.strip()
        tokens = list(text)
        if len(tokens) > max_length - 2:
            tokens = tokens[0:(max_length - 2)]
        tokens_f = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)
        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=None)
        result.append(feature)
    return result

def predict(sentences):
    print(sentences)
    data = encoder_corpus(sentences, config.max_length, vocab)
    input_ids = torch.LongTensor([temp.input_id for temp in data]) # [[token1_index, token_2_index], []...]
    input_masks = torch.LongTensor([temp.input_mask for temp in data])
    model.eval()
    with torch.no_grad():
        feats = model(input_ids, input_masks)
        best_path = model.crf.decode(feats, input_masks.byte())
        # print(best_path, sep='\n')
        return best_path

sentences = '海钓比赛地点在厦门与金门之间的海域'
# sentences = '朱祁镇重用王振，导致土木堡一战大败，断送了几乎明军精锐力量。要不是于谦挺身而出，打赢了北京保卫战，明朝或许就结束了'#
sentences = sentences.replace(' ', '')
labels = predict(sentences)[0]
id2tag = {label_dic[tag]: tag for tag in label_dic.keys()}
labels = [id2tag[index] for index in labels]
sentences = ['[CLS]'] + list(sentences) + ['[SEP]']
assert len(sentences) == len(labels)

print(sentences, labels, sep='\n')