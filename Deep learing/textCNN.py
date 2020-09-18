#%%
import jieba
from gensim.models import word2vec
import pandas as pd
import numpy as np

df = pd.read_csv('train_data.csv')
df = df.fillna('')
df1 = pd.read_csv('test_data.csv')
df1 = df1.fillna('')
gg = pd.read_excel('e04.xlsx')
gg = list(gg.e04.values)
y_train = df['label']
df = df.drop(['ID','label_name','label'],axis = 1)
df1 = df1.drop(['id'],axis = 1)
data = pd.concat([df,df1],axis = 0)
data['title_keyword'] = data['title'].str.cat(data['keyword'], sep=',')
data['title_keyword'] = data['title_keyword'].astype('str')
data = data.reset_index(drop=True)

def get_custom_stopwords(stop_words_file):
    with open(stop_words_file,encoding="utf-8") as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list
stop_words_file = "hit_stopwords.txt"
stopwords = get_custom_stopwords(stop_words_file)
stopwords+=gg

def chinese_word_cut(mytext):
    a = jieba.lcut(mytext)
    a1 = [i for i in a if i not in stopwords]
    return a1
X = data.title_keyword.apply(chinese_word_cut)
cutword = [" ".join(X[i]) for i in range(len(X))]
data['cutword'] = cutword
model = word2vec.Word2Vec(X, size=700, iter=10, sg=1)

vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
word2idx = {"_PAD": 0}  # 初始化 `[word : token]` 字典，tokenize語料庫就是用該词典。
embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
print('Found %s word vectors.' % len(model.wv.vocab.items()))
for i in range(len(vocab_list)):
	word = vocab_list[i][0]
	word2idx[word] = i + 1
	embeddings_matrix[i + 1] = vocab_list[i][1]
#%%
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding,Input,Conv1D,MaxPooling1D,concatenate,Flatten,Dropout
from keras.models import Model
from keras.layers import Input
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import Adam
import keras
from keras.utils import np_utils

X_train = data.iloc[:239629]
X_test = data.iloc[239629:]
max_len = 50
texts = X_train['cutword'].values
tokenizer = Tokenizer(num_words=embeddings_matrix.shape[0]-1)
tokenizer.fit_on_texts(texts)
X2 = tokenizer.texts_to_sequences(texts)
x_train_padded_seqs=pad_sequences(X2,maxlen=max_len)
texts1 = X_test['cutword'].values
X3 = tokenizer.texts_to_sequences(texts1)
x_test_padded_seqs=pad_sequences(X3,maxlen=max_len)

kernel_size = 3
model = Sequential()
main_input = Input(shape=(max_len,), dtype='float64')
embedder = Embedding(embeddings_matrix.shape[0], 300, input_length=max_len, weights=[embeddings_matrix], trainable=False)
embed = embedder(main_input)
cnn1 = Conv1D(256, 3, padding='valid', strides=1, activation='relu')(embed)
cnn1 = MaxPooling1D(max_len-3+1)(cnn1)
cnn2 = Conv1D(256, 4, padding='valid', strides=1, activation='relu')(embed)
cnn2 = MaxPooling1D(max_len-4+1)(cnn2)
cnn3 = Conv1D(256, 5, padding='valid', strides=1, activation='relu')(embed)
cnn3 = MaxPooling1D(max_len-5+1)(cnn3)
# 合併三個模型的输出向量
cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
flat = Flatten()(cnn)
hidden = Dense(256, activation='relu')(flat)
drop = Dropout(0.2)(hidden)
main_output = Dense(10, activation='softmax')(drop)
model = Model(inputs=main_input, outputs=main_output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
ytrain = y_train.values
one_hot_labels = np_utils.to_categorical(ytrain, num_classes=10)
model.fit(x_train_padded_seqs, one_hot_labels, batch_size=128, epochs=20)
result = model.predict(x_test_padded_seqs)
result_labels = np.argmax(result, axis=1)
#%%
test_label = pd.DataFrame({'id':list(range(len(result_labels))),'label':result_labels})
test_label.to_csv('result1.csv',index = False)

