# coding=utf-8


class Config(object):
    def __init__(self):
        #人民日报语料库         
        self.label_file = './data/tag/PeopleDaliy_tag.txt'
        self.train_file = './data/People_daliy/People_Daily_train.txt'
        self.dev_file = './data/People_daliy/People_Daily_dev.txt'
        self.test_file = './data/People_daliy/People_Daily_test.txt' 
        #MASA语料库
        # self.label_file = './data/tag/MASA_tag.txt'
        # self.train_file = './data/MASA/MASA_train.txt' 
        # self.dev_file   = './data/MASA/MASA_dev.txt'
        # self.test_file  = './data/MASA/MASA_test.txt' 
        # Boson语料库
        # self.label_file = './data/tag/Boson_tag.txt'
        # self.train_file = './data/Boson/boson_train.txt' 
        # self.dev_file   = './data/Boson/boson_dev.txt'
        # self.test_file  = './data/Boson/boson_test.txt'
        #微博语料库
        # self.label_file = './data/tag/Weibo_tag.txt'
        # self.train_file = './data/Weibo/weiboNER_train.txt' 
        # self.dev_file   = './data/Weibo/weiboNER_dev.txt'
        # self.test_file  = './data/Weibo/weiboNER_test.txt'
        
        self.vocab = './data/ernie-base-chinese/vocab.txt'
        self.max_length = 100
        self.use_cuda = False
        self.gpu = 0
        self.batch_size = 50
        self.ernie_path = './data/ernie-base-chinese'
        self.rnn_hidden = 500
        self.ernie_embedding = 768
        self.dropout1 = 0.5
        self.dropout_ratio = 0.5
        self.rnn_layer = 1
        self.lr = 5e-5
        self.lr_decay = 0.00001
        self.weight_decay = 0.00005
        self.checkpoint = 'result/'
        self.optim = 'Adam'
        self.load_model = True
        self.load_path = 'PeopleDaily-9718'
        # self.load_path = 'MASA-98386'
        # self.load_path = 'Boson-9059'
        # self.load_path = 'Weibo-8205'
        self.base_epoch = 10
        self.require_improvement = 1000

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):

        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])

