# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from config import Config
from model import ERNIE_LSTM_CRF
import torch.optim as optim
from utils import load_vocab, read_corpus, load_model, save_model, build_dataset, get_time_diff
from torch.utils.data import DataLoader
import fire
import warnings
import time
import datetime
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

warnings.filterwarnings('ignore', category=UserWarning)

def train(**kwargs):
    config = Config()
    config.update(**kwargs)
    print('当前设置为:\n', config)
    if config.use_cuda:
        torch.cuda.set_device(config.gpu)
    print('loading corpus')
    vocab = load_vocab(config.vocab) # {token: index}
    label_dic = load_vocab(config.label_file) # {tag: index}
    id2tag = {label_dic[tag]: tag for tag in label_dic.keys()}
    tagset_size = len(label_dic)
    train_data = read_corpus(config.train_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
    dev_data = read_corpus(config.dev_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
    test_data = read_corpus(config.test_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)


    train_dataset = build_dataset(train_data)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)

    dev_dataset = build_dataset(dev_data)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=config.batch_size)

    test_dataset = build_dataset(test_data)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size)

    model = ERNIE_LSTM_CRF(config.ernie_path, tagset_size, config.ernie_embedding, config.rnn_hidden, config.rnn_layer, dropout_ratio=config.dropout_ratio, dropout1=config.dropout1, use_cuda=config.use_cuda)
    if config.load_model:
        assert config.load_path is not None
        model = load_model(model, name=config.load_path)
    if config.use_cuda:
        model.cuda()
    optimizer = getattr(optim, config.optim)
    optimizer = optimizer(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    '''model.train() # 设置成训练模式
    step = 0
    eval_loss = float('inf')
    last_improved = 0 # 记录上一次更新的step值
    flag = False
    for epoch in range(config.base_epoch):
        for i, batch in enumerate(train_loader):
            step += 1
            model.zero_grad()
            inputs, masks, tags = batch
            inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
            if config.use_cuda:
                inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()
            feats = model(inputs, masks)
            loss = model.loss(feats, masks.byte(), tags)
            loss.backward()
            optimizer.step()
            if step % 5 == 0:
                print('step: {} |  epoch: {}|  loss: {}'.format(step, epoch, loss.item()))
            if step % 50 == 0:
                f1, dev_loss = dev(model, dev_loader, config, id2tag, test=False) # 保存模型
                if dev_loss < eval_loss:
                    eval_loss = dev_loss
                    save_model(model, epoch)
                    last_improved = step
                    improve = '*'
                else:
                    improve = ''
                print('eval  epoch: {}|  f1_score: {}|  loss: {}|   {}'.format(epoch, f1, dev_loss, improve))
            if step - last_improved > config.require_improvement: # early stop
                print('No optimization for a long time, auto-stopping...')
                flag = True
                break
        if flag:
            break'''
    test(model, test_loader, config, id2tag)


def dev(model, dev_loader, config, id2tag, test=False):
    model.eval()
    eval_loss = 0
    true = []
    pred = []
    with torch.no_grad():
        for i, batch in enumerate(dev_loader):
            if test: # 测试时间过长，打印信心可以看到测试进度
                print('处理测试集数据第' + str(i * config.batch_size) + '到第' + str((i+1) * config.batch_size) + '条...')
            inputs, masks, tags = batch
            inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
            if config.use_cuda:
                inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()
            feats = model(inputs, masks)
            # 此处使用维特比算法解码
            best_path  = model.crf.decode(feats, masks.byte())
            loss = model.loss(feats, masks.byte(), tags)
            eval_loss += loss.item()
            pred.extend([t for t in best_path])
            true.extend([[x for x in t.tolist() if x != 0] for t in tags])
    true = [[id2tag[y] for y in x] for x in true]
    pred = [[id2tag[y] for y in x] for x in pred]
    f1 = f1_score(true, pred)
    if test:
        accuracy = accuracy_score(true, pred)
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        report = classification_report(true, pred,4)
        return accuracy, precision, recall, f1, eval_loss / len(dev_loader), report
    model.train()
    return f1, eval_loss / len(dev_loader)

def test(model, test_loader, config, id2tag):
    model.eval()
    accuracy, precision, recall, f1, loss, report = dev(model=model, dev_loader=test_loader,
                                                        config=config, id2tag=id2tag, test=True)
    print("ERNIE-BiLSTM-CRF on PeopleDaliy dataset is done")
    # print("ERNIE-BiLSTM-CRF on MASA dataset is done")
    # print("ERNIE-BiLSTM-CRF on Boson dataset is done")
    # print("ERNIE-BiLSTM-CRF on Weibo dataset is done")

    msg1 = 'Test Loss:{0:5.2}, Test Acc:{1:6.2%}'
    print(msg1.format(loss, accuracy))
    msg2 = 'Test Precision:{}, Test Recall:{}, Test F1_Socre:{}'
    print(msg2.format(precision, recall, f1))
    print('Cllassification Report:')
    print(report)

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    fire.Fire()
    end_time= datetime.datetime.now()
    elapsed = end_time - start_time
    print("start_time:" + start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    print("end_time:" + end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    print(" spend times ：" + str(int(elapsed.total_seconds()*1000)) + "ms") 
# python main.py train --use_cuda=True --batch_size=50 PeopleDaily MASA