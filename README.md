# ERNIE-BiLSTM-CRF-Pytorch-Chinese_NER
基于ERNIE的中文NER
# ERNIE-BiLSTM-CRF
## 模型架构
![ERNIE_BiLSTM_CRF.svg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4ab6ab8b749246caaf1a8fda08c5f7ba~tplv-k3u1fbpfcp-watermark.image?)
### ERNIE层
采用预训练语言模型ERNIE对输入的文本数据进行向量化表示
### BiLSTM
通过双向循环神经网络(BiLSTM)进行特征提取提取编码得到一个得分矩阵
### CRF
通过条件随机场(CRF)进行解码，再用维特比算法得到概率最大的一组标签作为算法的输出
# BERT-BiLSTM-CRF
## 模型架构
![Bert_bilstm-crf.svg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4b0366c0644e4db3a8c853fabcaf6a6c~tplv-k3u1fbpfcp-watermark.image?)
除了向量化采用BERT，其他与前者相同

# 实验
## 环境
使用此平台[openbayes](https://openbayes.com/)的容器可以省去搭环境的麻烦，数据会保存，每次需要安装依赖。

![算力容器.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4b1ccb4332084ff4b9a224df44e1ac8a~tplv-k3u1fbpfcp-watermark.image?)
> pip3 install pytorch-crf -i https://pypi.mirrors.ustc.edu.cn/simple/  
> pip3 install transformers -i https://pypi.mirrors.ustc.edu.cn/simple/  
> pip3 install fire -i https://pypi.mirrors.ustc.edu.cn/simple/  
> pip3 install seqeval -i https://pypi.mirrors.ustc.edu.cn/simple/

## 结果

```js
PEOPLE
step: 2600 |  epoch: 6|  loss: 0.03466796875
eval  epoch: 6|  f1_score: 0.9739056729430284|  loss: 1.22955077121708|   
No optimization for a long time, auto-stopping...
Bert-BiLSTM-CRF on PeopleDaliy dataset is done
Test Loss:  1.7, Test Acc:99.20%
Test Precision:0.9659206207102358, Test Recall:0.9723039951937519, Test F1_Socre:0.9691017964071856
Cllassification Report:
              precision    recall  f1-score   support
         LOC     0.9445    0.9431    0.9438      3430
         ORG     0.8693    0.8957    0.8823      2147
         PER     0.9814    0.9694    0.9754      1796
        eos>     1.0000    1.0000    1.0000      4636
      start>     1.0000    1.0000    1.0000      4636
   micro avg     0.9693    0.9715    0.9704     16645
   macro avg     0.9591    0.9616    0.9603     16645
weighted avg     0.9697    0.9715    0.9706     16645

start_time:2022-04-13 03:12:32.875
end_time:2022-04-13 03:44:08.855
BERT-BiLSTM-CRF on PeopleDaliy spend times ：1895980ms

eval  epoch: 5|  f1_score: 0.9800169553106455|  loss: 1.0503978200573871|   
No optimization for a long time, auto-stopping...
处理测试集数据第4600到第4650条...
ERNIE-BiLSTM-CRF on PeopleDaliy dataset is done
Test Loss:  1.4, Test Acc:99.35%
Test Precision:0.9692895978968752, Test Recall:0.9746470411534995, Test F1_Socre:0.9719609370319334
Cllassification Report:
              precision    recall  f1-score   support
         LOC     0.9453    0.9417    0.9435      3430
         ORG     0.8786    0.9166    0.8972      2147
         PER     0.9685    0.9766    0.9726      1796
        eos>     1.0000    0.9998    0.9999      4636
      start>     0.9998    1.0000    0.9999      4636
   micro avg     0.9691    0.9746    0.9719     16645
   macro avg     0.9584    0.9669    0.9626     16645
weighted avg     0.9696    0.9746    0.9721     16645

start_time:2022-04-12 10:55:04.588
end_time:2022-04-12 11:22:19.073
ERNIE-BiLSTM-CRF on PeopleDaliy spend times ：1634484

MASA
step: 2950 |  epoch: 3|  loss: 0.54973965883255
eval  epoch: 3|  f1_score: 0.9797556513014225|  loss: 0.8674196546586851|   
No optimization for a long time, auto-stopping...处理测试集数据第2350到第2400条...
Bert-BiLSTM-CRF on MASA dataset is done
Test Loss: 0.87, Test Acc:99.45%
Test Precision:0.9798500468603561, Test Recall:0.9796205200281096, Test F1_Socre:0.9797352700011713
Cllassification Report:
              precision    recall  f1-score   support
        _LOC     0.9644    0.9649    0.9646      1851
        _ORG     0.9365    0.9409    0.9387      1082
        _PER     0.9793    0.9769    0.9781       823
        eos>     0.9992    0.9992    0.9992      2391
      start>     1.0000    1.0000    1.0000      2391
   micro avg     0.9820    0.9824    0.9822      8538
   macro avg     0.9759    0.9764    0.9761      8538
weighted avg     0.9820    0.9824    0.9822      8538


start_time:2022-04-13 04:10:30.619
end_time:2022-04-13 04:46:29.236
ERNIE-BiLSTM-CRF on MASA spend times ：2158616ms

step: 3650 |  epoch: 4|  loss: 0.1283474713563919
eval  epoch: 4|  f1_score: 0.9846754629903556|  loss: 0.7818139177785876|   
No optimization for a long time, auto-stopping...
理测试集数据第2350到第2400条...
ERNIE-BiLSTM-CRF on MASA dataset is done
Test Loss: 0.79, Test Acc:99.60%
Test Precision:0.9851288056206089, Test Recall:0.985359568985711, Test F1_Socre:0.9852441737908422
Cllassification Report:
              precision    recall  f1-score   support
        _LOC     0.9695    0.9789    0.9742      1851
        _ORG     0.9342    0.9187    0.9264      1082
        _PER     0.9843    0.9891    0.9867       823
        eos>     1.0000    1.0000    1.0000      2391
      start>     1.0000    1.0000    1.0000      2391
   micro avg     0.9836    0.9841    0.9838      8538
   macro avg     0.9776    0.9773    0.9774      8538
weighted avg     0.9835    0.9841    0.9838      8538

start_time:2022-04-12 11:48:15.936
end_time:2022-04-12 12:32:24.776
ERNIE-BiLSTM-CRF on MASA spend times ：2648840

BOSON
step: 1495 |  epoch: 8|  loss: 1.1240458488464355
step: 1500 |  epoch: 8|  loss: 1.8254204988479614
eval  epoch: 8|  f1_score: 0.9000338104361546|  loss: 6.462216214700178|   
No optimization for a long time, auto-stopping...
处理测试集数据第0到第50条...
处理测试集数据第1050到第1100条...
Bert-BiLSTM-CRF on Boson dataset is done
Test Loss:  7.0, Test Acc:96.82%
Test Precision:0.8852934612809641, Test Recall:0.9170134073046694, Test F1_Socre:0.9008743045304871
Cllassification Report:
              precision    recall  f1-score   support
        _COM     0.7324    0.7156    0.7239       218
        _LOC     0.7721    0.8210    0.7958       458
        _ORG     0.7410    0.7875    0.7635       287
        _PER     0.9500    0.9438    0.9469       463
        _PRO     0.6190    0.7647    0.6842       323
        _TIM     0.7835    0.8649    0.8222       385
        eos>     1.0000    1.0000    1.0000      1096
      start>     1.0000    1.0000    1.0000      1096
   micro avg     0.8853    0.9170    0.9009      4326
   macro avg     0.8248    0.8622    0.8421      4326
weighted avg     0.8921    0.9170    0.9037      4326

start_time:2022-04-29 12:45:16.060
end_time:2022-04-29 13:01:20.169
BERT-BiLSTM-CRF on Boson spend times ：964109ms

step: 1700 |  epoch: 9|  loss: 0.7510769367218018
eval  epoch: 9|  f1_score: 0.902540937323546|  loss: 6.62023774060336|   
No optimization for a long time, auto-stopping...
处理测试集数据第0到第50条...
处理测试集数据第1050到第1100条...
ERNIE-BiLSTM-CRF on Boson dataset is done
Test Loss:  7.0, Test Acc:97.00%
Test Precision:0.8999322340185227, Test Recall:0.9209431345353676, Test F1_Socre:0.9103164629269964
Cllassification Report:
              precision    recall  f1-score   support
        _COM     0.8308    0.7661    0.7971       218
        _LOC     0.7894    0.8755    0.8302       458
        _ORG     0.7458    0.7666    0.7560       287
        _PER     0.9403    0.9525    0.9464       463
        _PRO     0.6040    0.7461    0.6676       323
        _TIM     0.7908    0.8442    0.8166       385
        eos>     1.0000    1.0000    1.0000      1096
      start>     1.0000    1.0000    1.0000      1096
   micro avg     0.8909    0.9216    0.9060      4326
   macro avg     0.8376    0.8689    0.8517      4326
weighted avg     0.8977    0.9216    0.9087      4326

start_time:2022-04-29 13:44:39.430
end_time:2022-04-29 14:02:25.155
test ERNIE-BiLSTM-CRF on Boson spend times ：1065725ms

WEIBO
eval  epoch: 9|  f1_score: 0.8382978723404255|  loss: 7.672243356704712|   
step: 255 |  epoch: 9|  loss: 1.0741403102874756
step: 260 |  epoch: 9|  loss: 1.288889765739441
step: 265 |  epoch: 9|  loss: 1.1143728494644165
step: 270 |  epoch: 9|  loss: 0.9143328666687012
处理测试集数据第0到第50条...
处理测试集数据第50到第100条...
处理测试集数据第100到第150条...
处理测试集数据第150到第200条...
处理测试集数据第200到第250条...
处理测试集数据第250到第300条...
Bert-BiLSTM-CRF on Weibo dataset is done
Test Loss:  9.0, Test Acc:96.52%
Test Precision:0.8148914167528438, Test Recall:0.8427807486631016, Test F1_Socre:0.8286014721345951
Cllassification Report:
              precision    recall  f1-score   support
     GPE.NAM     0.7021    0.7500    0.7253        44
     GPE.NOM     0.0000    0.0000    0.0000         2
     LOC.NAM     0.3200    0.4211    0.3636        19
     LOC.NOM     0.1538    0.2500    0.1905         8
     ORG.NAM     0.3878    0.4872    0.4318        39
     ORG.NOM     0.3478    0.5000    0.4103        16
     PER.NAM     0.7320    0.6698    0.6995       106
     PER.NOM     0.6185    0.6646    0.6407       161
        eos>     1.0000    1.0000    1.0000       270
      start>     1.0000    1.0000    1.0000       270
   micro avg     0.8149    0.8428    0.8286       935
   macro avg     0.5262    0.5743    0.5462       935
weighted avg     0.8300    0.8428    0.8354       935

start_time:2022-04-30 02:34:45.820
end_time:2022-04-30 02:37:25.067
Spend times ：159247ms
eval  epoch: 9|  f1_score: 0.8486842105263157|  loss: 6.780091285705566|   *
step: 255 |  epoch: 9|  loss: 3.4152257442474365
step: 260 |  epoch: 9|  loss: 3.636118173599243
step: 265 |  epoch: 9|  loss: 4.719958305358887
step: 270 |  epoch: 9|  loss: 3.588787078857422
处理测试集数据第0到第50条...
处理测试集数据第50到第100条...
处理测试集数据第100到第150条...
处理测试集数据第150到第200条...
处理测试集数据第200到第250条...
处理测试集数据第250到第300条...
ERNIE-BiLSTM-CRF on Weibo dataset is done
Test Loss:  8.3, Test Acc:96.37%
Test Precision:0.8186528497409327, Test Recall:0.8449197860962567, Test F1_Socre:0.8315789473684211
Cllassification Report:
              precision    recall  f1-score   support

     GPE.NAM     0.6452    0.9091    0.7547        44
     GPE.NOM     0.0000    0.0000    0.0000         2
     LOC.NAM     0.0000    0.0000    0.0000        19
     LOC.NOM     0.0000    0.0000    0.0000         8
     ORG.NAM     0.2027    0.3846    0.2655        39
     ORG.NOM     0.0000    0.0000    0.0000        16
     PER.NAM     0.6697    0.6887    0.6791       106
     PER.NOM     0.6778    0.7578    0.7155       161
        eos>     1.0000    1.0000    1.0000       270
      start>     1.0000    1.0000    1.0000       270
   micro avg     0.8187    0.8449    0.8316       935
   macro avg     0.4195    0.4740    0.4415       935
weighted avg     0.8090    0.8449    0.8243       935

start_time:2022-04-30 01:49:20.517
end_time:2022-04-30 01:51:52.726
 spend times ：152208ms
```
总体结果三个维度(Precision、Recall、F1)ERNIE相对BERT均有提升

| 数据集 |   | P(%) |R(%) |F1(%) |
| :----: | :----:  | :----: |:----: |:----: |
| 人民日报数据集 | BERT-BiLSTM-CRF  |96.592 |97.230 | 96.910 |
| 人民日报数据集 | ERNIE-BiLSTM-CRF |96.928 |97.464 | 97.196 |
| MASA数据集    | BERT-BiLSTM-CRF  |97.985 |97.962 | 97.973 |
| MASA数据集    | ERNIE-BiLSTM-CRF |98.512 |98.525 | 98.524 |
| Boson数据集   | BERT-BiLSTM-CRF  |88.529 |91.701 | 90.087 |
| Boson数据集   | ERNIE-BiLSTM-CRF |89.993 |92.094 | 91.031 |
| Weibo数据集   | BERT-BiLSTM-CRF  |81.489 |84.248 | 82.860 |
| Weibo数据集   | ERNIE-BiLSTM-CRF |81.865 |84.491 | 83.157 |

感谢网友StevenRogers在Gitee分享的源码，虽与其素昧平生，基准模型[BERT-BiLSTM-CRF](https://gitee.com/StevenRogers/bert-bilstm-crf-pytorch.git)<br />
预训练模型[BERT](https://huggingface.co/bert-base-chinese/tree/main) [ERNIE1.0](https://huggingface.co/nghuyong/ernie-1.0/tree/main)<br />
数据集 [人民日报](https://github.com/OYE93/Chinese-NLP-Corpus) [MASA](https://github.com/caoyuji1986/ner_corpus) [Boson](https://github.com/HuHsinpang/BosonNER-Pretreatment) [Weibo](https://github.com/OYE93/Chinese-NLP-Corpus)
