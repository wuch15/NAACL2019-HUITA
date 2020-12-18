#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from json import *

with open('reviews_Movies_and_TV_5.json','r')as f:
    data=f.readlines()
rawdata=[JSONDecoder().decode(x) for x in data]

for i in range(len(rawdata)):
    rawdata[i]['text']=[word_tokenize(x) for x in sent_tokenize(rawdata[i]['reviewText'].lower())]


# In[3]:


word_dict={'PADDING':[0,999999]}
for i in rawdata:
    for k in i['text']:
        for j in k:
            if j in word_dict:
                word_dict[j][1]+=1
            else:
                word_dict[j]=[len(word_dict),1]


# In[4]:


word_dict_freq={}
for x in  word_dict:
    if word_dict[x][1]>=10:
        word_dict_freq[x]=[len(word_dict_freq),word_dict[x][1]]
print(len(word_dict_freq),len(word_dict))


# In[ ]:


import numpy as np
embdict={}
plo=0
import pickle
with open('/data/wuch/glove.840B.300d.txt','rb')as f:
    linenb=0
    while True:
        j=f.readline()
        if len(j)==0:
            break
        k = j.split()
        #print(len(k))
        word=k[0].decode()
        linenb+=1
        if len(word) != 0:
            tp=[float(x) for x in k[1:]]
            #tp= np.fromstring(' '.join(k[1:]), dtype='float32')
            #print(word)
            if word in word_dict_freq:
                embdict[word]=tp
                if plo%100==0:
                    print(plo,linenb,word)
                plo+=1


# In[ ]:


from numpy.linalg import cholesky
word_dict1=word_dict_freq
print(len(embdict),len(word_dict1))
print(len(word_dict1))
wordemb=[0]*len(word_dict1)
xp=np.zeros(300,dtype='float32')

findemb=[]
for i in embdict.keys():
    wordemb[word_dict1[i][0]]=np.array(embdict[i],dtype='float32')
    findemb.append(wordemb[word_dict1[i][0]])
findemb=np.array(findemb,dtype='float32')

mu=np.mean(findemb, axis=0)
Sigma=np.cov(findemb.T)

norm=np.random.multivariate_normal(mu, Sigma, 1)
print(mu.shape,Sigma.shape,norm.shape)

for i in range(len(wordemb)):
    if type(wordemb[i])==int:
        wordemb[i]=np.reshape(norm, 300)
wordemb[0]=np.zeros(300,dtype='float32')
wordemb=np.array(wordemb,dtype='float32')
print(wordemb.shape)



uir_triples=[]
for i in rawdata:
    temp={}
    doc=[]
    for y in i['text']:
        doc.append([word_dict_freq[x][0] for x in y if x in word_dict_freq])
        
    temp['text']=doc
    temp['item']=i['asin']
    temp['user']=i['reviewerID']
    uir_triples.append(temp)


# In[7]:


for i in range(len(uir_triples)):
    uir_triples[i]['id']=i


# In[8]:


for i in range(len(rawdata)):
    uir_triples[i]['label']=rawdata[i]['overall']


# In[9]:

indices = np.arange(len(uir_triples))
np.random.shuffle(indices)
trainindex=indices[:int(0.8*len(uir_triples))]
otherindex=indices[int(0.8*len(uir_triples)):]

ir_id={}
ur_id={}

for i in trainindex:
    if uir_triples[i]['item'] in ir_id:
        
        ir_id[uir_triples[i]['item']].append(uir_triples[i]['id'])
    else:
        ir_id[uir_triples[i]['item']]=[uir_triples[i]['id']]
    if uir_triples[i]['user'] in ur_id:
        
        ur_id[uir_triples[i]['user']].append(uir_triples[i]['id'])
    else:
        ur_id[uir_triples[i]['user']]=[uir_triples[i]['id']]        


for i in otherindex:
    if uir_triples[i]['item'] not in ir_id:
        ir_id[uir_triples[i]['item']]=[]
    if uir_triples[i]['user'] not in ur_id:
        ur_id[uir_triples[i]['user']]=[]
        
# In[ ]:



MAX_SENT_LENGTH = 40
MAX_SENTS = 15
MAX_REVIEW_USER = 25
MAX_REVIEW_ITEM = 50


# In[15]:


import random
for i in ir_id:
    random.shuffle(ir_id[i])


# In[16]:


import random
for i in ur_id:
    random.shuffle(ur_id[i])


# In[18]:


all_u=[]
for i in ur_id:
    tp=[]
    for j in ur_id[i][:MAX_REVIEW_USER]:
        
        atp=[x[:MAX_SENT_LENGTH] for x in uir_triples[j]['text'][:MAX_SENTS]]
        btp=[x+(MAX_SENT_LENGTH-len(x))*[0] for x in atp]
        tp.append(btp+[[0]*MAX_SENT_LENGTH]*(MAX_SENTS-len(btp)))
    all_u.append(tp+[[[0]*MAX_SENT_LENGTH]*MAX_SENTS]*(MAX_REVIEW_USER-len(tp)))


# In[ ]:


all_i=[]
for i in ir_id:
    tp=[]
    for j in ir_id[i][:MAX_REVIEW_ITEM]:
        
        atp=[x[:MAX_SENT_LENGTH] for x in uir_triples[j]['text'][:MAX_SENTS]]
        btp=[x+(MAX_SENT_LENGTH-len(x))*[0] for x in atp]
        tp.append(btp+[[0]*MAX_SENT_LENGTH]*(MAX_SENTS-len(btp)))
    all_i.append(tp+[[[0]*MAX_SENT_LENGTH]*MAX_SENTS]*(MAX_REVIEW_ITEM-len(tp)))



# In[ ]:


import numpy as np
all_u=np.array(all_u,dtype='int32')


# In[ ]:


import numpy as np
all_i=np.array(all_i,dtype='int32')




id_dict={}
ider=0
for i in ir_id:
    id_dict[i]=ider
    ider+=1


# In[ ]:


ud_dict={}
ider=0
for i in ur_id:
    ud_dict[i]=ider
    ider+=1


# In[ ]:


itid=[]
usid=[]
label=[]
for i in uir_triples:
    itid.append(id_dict[i['item']])
    usid.append(ud_dict[i['user']])
    label.append(i['label'])


# In[ ]:


label=np.array(label,dtype='float32')
itid=np.array(itid,dtype='int32')
usid=np.array(usid,dtype='int32')


train_label=label[indices[:int(0.8*len(label))]]
train_itid=itid[indices[:int(0.8*len(label))]]
train_usid=usid[indices[:int(0.8*len(label))]]

val_label=label[indices[int(0.8*len(label)):int(0.9*len(label))]]
val_itid=itid[indices[int(0.8*len(label)):int(0.9*len(label))]]
val_usid=usid[indices[int(0.8*len(label)):int(0.9*len(label))]]

test_label=label[indices[int(0.9*len(label)):]]
test_itid=itid[indices[int(0.9*len(label)):]]
test_usid=usid[indices[int(0.9*len(label)):]]




def generate_batch_data_random(item,user,itid,usid, y, batch_size):
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]

    while (True):
        for i in batches:
            itx = item[itid[i]]
            usx=user[usid[i]]
            iid=np.expand_dims(itid[i],axis=1)
            uid=np.expand_dims(usid[i],axis=1)
            yy=y[i]
            yield ([itx, usx,iid,uid], yy)



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import Model, load_model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import *
from keras.optimizers import *



class AttLayer(Layer):
    def __init__(self, **kwargs):
        # self.init = initializations.get('normal')#keras1.2.2
        self.init = initializers.get('normal')

        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], 1)))
        # self.W = self.init((input_shape[-1],))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        # eij = K.tanh(K.dot(x, self.W))
        print(x.shape)
        print(self.W.shape)
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        print(ai.shape)
        # weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')
        weights = ai / K.expand_dims(K.sum(ai, axis=1), 1)
        print('weights', weights.shape)
        # weighted_input = x * weights.dimshuffle(0, 1, 'x')
        weighted_input = x * weights

        # return weighted_input.sum(axis=1)
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')


embedding_layer = Embedding(len(word_dict_freq), 300, weights=[wordemb],trainable=True)

embedded_sequences = embedding_layer(sentence_input)
d_emb=Dropout(0.2)(embedded_sequences)

word_cnn = Convolution1D(nb_filter=200, filter_length=3,  padding='same', activation='relu', strides=1)(d_emb)
word_cnn_d=Dropout(0.2)(word_cnn)

word_att = AttLayer()(word_cnn_d)
sentEncoder = Model(sentence_input, word_att)

review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder, name='sentEncoder')(review_input)

sent_cnn = Convolution1D(nb_filter=100, filter_length=3, padding='same', activation='relu', strides=1)(review_encoder)
sent_cnn_d=Dropout(0.2)(sent_cnn)

sent_att = AttLayer()(sent_cnn_d)

reviewEncoder = Model(review_input, sent_att)

reviews_input = Input(shape=(MAX_REVIEW_ITEM,MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
reviews_encoder = TimeDistributed(reviewEncoder, name='reviewEncoder')(reviews_input)

item_att= AttLayer()(reviews_encoder)

sentence_input2 = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')

embedded_sequences2 = embedding_layer(sentence_input2)
d_emb2=Dropout(0.2)(embedded_sequences2)

word_cnn2 = Convolution1D(nb_filter=200, filter_length=3,  padding='same', activation='relu', strides=1)(d_emb2)
word_cnn_d2=Dropout(0.2)(word_cnn2)

word_att2 = AttLayer()(word_cnn_d2)
sentEncoder2 = Model(sentence_input2, word_att2)

review_input2 = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder2 = TimeDistributed(sentEncoder2, name='sentEncoder2')(review_input2)

sent_cnn2 = Convolution1D(nb_filter=100, filter_length=3, padding='same', activation='relu', strides=1)(review_encoder2)
sent_cnn_d2=Dropout(0.2)(sent_cnn2)
sent_att2 = AttLayer()(sent_cnn_d2)

reviewEncoder2 = Model(review_input2, sent_att2)
reviews_input2 = Input(shape=(MAX_REVIEW_USER,MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
reviews_encoder2 = TimeDistributed(reviewEncoder2, name='reviewEncoder2')(reviews_input2)
user_att = AttLayer()(reviews_encoder2)


user_id = Input(shape=(1,), dtype='int32')
item_id = Input(shape=(1,), dtype='int32')


user_embedding= Embedding(len(ur_id),100,trainable=True)(user_id)
user_embedding= Flatten()(user_embedding)
item_embedding = Embedding(len(ir_id), 100,trainable=True)(item_id)
item_embedding= Flatten()(item_embedding)
factor=concatenate([user_att,user_embedding])
factor2=concatenate([item_att,item_embedding])
fa=multiply([factor,factor2])
preds = Dense(1,activation='relu')(fa)
model = Model([reviews_input,reviews_input2,item_id,user_id], preds)

model.compile(loss='mse', optimizer=Adam(lr=0.0001,amsgrad=True), metrics=['mse'])


from sklearn.metrics import mean_squared_error
for ep in range(3):
    traingen=generate_batch_data_random(all_i,all_u,train_itid,train_usid,train_label,20)
    valgen=generate_batch_data_random(all_i,all_u,test_itid,test_usid,test_label,20)
    model.fit_generator(traingen, epochs=1,steps_per_epoch=len(train_itid)//20)
    result = model.evaluate_generator(valgen, steps=len(test_itid)//20)
    print(result)

