
# coding: utf-8

# # PPDAI Text Mining

# In[ ]:

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
from ppdaiutil import *


# In[ ]:

config = {
    'TRAIN_PATH':'data/train.csv',
    'TEST_PATH':'data/test.csv',
    'QUESTION_PATH' : 'data/question.csv',   
}


# ** read data **

# In[ ]:

print('Load files...')
data={
    'qes' : pd.read_csv(config['QUESTION_PATH']),
    'tr' : pd.read_csv(config['TRAIN_PATH']),
    'te' : pd.read_csv(config['TEST_PATH']),
    #'co' : questions['words'],
}
data['co']=data['qes']['words']


# In[ ]:

if False:
    display(data['qes'].head())
    display(data['tr'].head())
    display(data['te'].head())


# **1. ID轉成詞語序列or單字序列**

# In[ ]:

def get_ids(qids):
    ids = []
    for t_ in qids:
        ids.append(int(t_[1:]))
    return np.asarray(ids)

def get_textschars(d):
    all_words = data['qes']['words']
    all_chars = data['qes']['chars']
    q1id, q2id = d['q1'], d['q2']
    id1s, id2s = get_ids(q1id), get_ids(q2id)
    q1_texts = []
    q2_texts = []
    for t_ in zip(id1s, id2s):
        q1_texts.append(all_words[t_[0]])
        q2_texts.append(all_words[t_[1]])
    d['q1_texts'] = q1_texts
    d['q2_texts'] = q2_texts
    
    q1_chars = []
    q2_chars = []
    for t_ in zip(id1s, id2s):
        q1_chars.append(all_chars[t_[0]])
        q2_chars.append(all_chars[t_[1]])
    d['q1_chars'] = q1_chars
    d['q2_chars'] = q2_chars
    

print('Get texts/chars...')
get_textschars(data['tr'])
get_textschars(data['te'])


# In[ ]:

data['tr'].head()


# **2. 序列化**
# - tokenizer

# In[ ]:


with open('data/word_embed.txt') as f:
    MAX_NB_WORDS = (len(list(f)))

trq1_text=data['tr']['q1_texts'].values
trq2_text=data['tr']['q2_texts'].values
teq1_text=data['te']['q1_texts'].values
teq2_text=data['te']['q2_texts'].values
alltext=np.concatenate([trq1_text, trq2_text, teq1_text, teq2_text])
MAX_SEQUENCE_LENGTH = max(list(map(lambda x: len(x), alltext))) 

tokenizer = Tokenizer(num_words=MAX_NB_WORDS) 
tokenizer.fit_on_texts(alltext) 


# In[ ]:

data['tr']['q1_sequences'] = tokenizer.texts_to_sequences(trq1_text) 
data['tr']['q2_sequences'] = tokenizer.texts_to_sequences(trq2_text) 
data['te']['q1_sequences'] = tokenizer.texts_to_sequences(teq1_text) 
data['te']['q2_sequences'] = tokenizer.texts_to_sequences(teq2_text) 
#data['tr'].head()


# In[ ]:

word_index = tokenizer.word_index 
print('Found %s unique tokens' % len(word_index)) 


# ** 3. pad_sequences **

# In[ ]:

data['trq1_padseq'] = pad_sequences(data['tr']['q1_sequences'], maxlen=MAX_SEQUENCE_LENGTH) 
data['trq2_padseq'] = pad_sequences(data['tr']['q2_sequences'], maxlen=MAX_SEQUENCE_LENGTH) 
data['teq1_padseq'] = pad_sequences(data['te']['q1_sequences'], maxlen=MAX_SEQUENCE_LENGTH) 
data['teq2_padseq'] = pad_sequences(data['te']['q2_sequences'], maxlen=MAX_SEQUENCE_LENGTH) 


# ** 4. prepare embeddings**

# In[ ]:


EMBEDDING_FILE='data/word_embed.txt'
EMBEDDING_DIM = 300

embeddings_index = {} 
f = open(EMBEDDING_FILE,"rb") 
for line in f: 
    values = line.split() 
    word = values[0].decode(encoding='utf-8')
    coefs = np.asarray(values[1:], dtype='float32') 
    embeddings_index[word] = coefs 
f.close() 

nb_words = len(word_index)+1
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM)) 
for word, i in word_index.items(): 
    word = word.upper()
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector 

    


# ** prepare training data**

# In[ ]:

## sample train/validation data

VALIDATION_SPLIT = 0.1
trlen = len(data['trq1_padseq'])
perm = np.random.permutation(trlen)
idx_train = perm[:int(trlen*(1-VALIDATION_SPLIT))] 
idx_val = perm[int(trlen*(1-VALIDATION_SPLIT)):] 

data_trainq1=data['trq1_padseq'][idx_train] 
data_trainq2=data['trq2_padseq'][idx_train] 
data_valq1=data['trq1_padseq'][idx_val] 
data_valq2=data['trq2_padseq'][idx_val] 

labels_train = data['tr']['label'][idx_train] 
labels_val = data['tr']['label'][idx_val] 


date_testq1 = data['teq1_padseq']
date_testq2 = data['teq2_padseq']


# In[ ]:

embedding_layer = Embedding(input_dim=nb_words, output_dim=300, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False) 
num_lstm = 300 
num_dense = 256 
rate_drop_lstm = 0.25 
rate_drop_dense = 0.25 
act = 'relu' 


q1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='q1_input') 
q2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='q2_input') 
q1_embseq= embedding_layer(q1_input) 
q2_embseq= embedding_layer(q2_input) 

lstm_layerq1 = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm,return_sequences=True, name='q1_lstm') 
q1_lstm = lstm_layerq1(q1_embseq) 
q1_drop = Dropout(rate_drop_dense, name='q1_drop')(q1_lstm) 
q1_att = Attention(MAX_SEQUENCE_LENGTH)(q1_drop)

lstm_layerq2 = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm,return_sequences=True, name='q2_lstm') 
q2_lstm = lstm_layerq2(q2_embseq) 
q2_drop = Dropout(rate_drop_dense, name='q2_drop')(q2_lstm) 
q2_att = Attention(MAX_SEQUENCE_LENGTH)(q2_drop)

q1q2_concat = Concatenate(axis=-1,name='q1q2concat')([q1_att,q2_att])
q1q2_concat = Dense(num_dense, activation=act, name='Q_dense')(q1q2_concat) 
q1q2_concat = Dropout(rate_drop_dense, name='Q_drop')(q1q2_concat) 
q1q2_concat = BatchNormalization(name='Q_batchnorm')(q1q2_concat) 
preds = Dense(1, activation='sigmoid', name='Q_output')(q1q2_concat)

model = Model(inputs=[q1_input, q2_input],  outputs=preds) 
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy']) 
print(model.summary()) 
plot_model(model, to_file='model.png')


# ** training **

# In[ ]:


STAMP = 'model/simple_lstm_glove_vectors_%.2f_%.2f'%(rate_drop_lstm,rate_drop_dense) 
print('STAMP',STAMP)
bst_model_path = STAMP + '.h5' 
print('bst_model_path',bst_model_path) 

early_stopping =EarlyStopping(monitor='val_loss', patience=5) 
#model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True) 

hist = model.fit([data_trainq1, data_trainq2], labels_train, validation_data=([data_valq1,data_valq2], labels_val), epochs=50, batch_size=256, shuffle=True, callbacks=[early_stopping]) 

#model.load_weights(bst_model_path) 
bst_val_score = min(hist.history['val_loss']) 

y_test = model.predict([date_testq1, date_testq2], batch_size=1024, verbose=1) 

#data['sam'][list_classes] = y_test 
#data['sam'].to_csv('%.4f_'%(bst_val_score) + STAMP + '.csv', index=False)

    
    
    


# In[ ]:


def make_submission(predict_prob):
    with open('submission.csv', 'w') as file:
        file.write(str('y_pre') + '\n')
        for line in predict_prob:
            file.write(str(line) + '\n')
    file.close()
    
testpred = model.predict([date_testq1, date_testq2], batch_size=1024, verbose=1) 
make_submission(testpred[:, 0])


# In[ ]:




# In[ ]:




# In[ ]:

## TOTDO: 
1. 是否成改成logloss
2. 確認test data的logloss

