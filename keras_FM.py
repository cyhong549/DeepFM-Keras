import numpy as np
np.random.seed(42)
import random as rn
rn.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from sklearn.base import BaseEstimator

from keras.layers import Input, Embedding, Dense,Flatten,\
    Concatenate,dot,Activation,Reshape,BatchNormalization,concatenate,Dropout,add,\
    RepeatVector,merge,multiply,Lambda
from keras.models import Model
from keras.regularizers import l2 as l2_reg
#from keras import initializations
import itertools
from keras import backend  as KK
from keras.engine.topology import Layer
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam


def dis_sparse(concat_train,concat_test,one_hot=True):
    from sklearn.preprocessing import OneHotEncoder,LabelEncoder     
    lbl = LabelEncoder()
    l = np.vstack((concat_train,concat_test))
    l = np.unique(l)    
    lbl.fit(l.reshape(-1,1))
    concat_train= lbl.transform(concat_train.reshape(-1,1))
    concat_test = lbl.transform(concat_test.reshape(-1,1))
    if(one_hot==True):
        one_clf = OneHotEncoder()
        l = np.vstack((concat_train.reshape(-1,1),concat_test.reshape(-1,1)))
        l = np.unique(l)    
        one_clf.fit(l.reshape(-1,1))
        sparse_training_matrix= one_clf.transform(concat_train.reshape(-1,1))
        sparse_testing_matrix = one_clf.transform(concat_test.reshape(-1,1))
        return sparse_training_matrix,sparse_testing_matrix
    else:
        return concat_train,concat_test

class MyLayer(Layer):
    def __init__(self, output_dim= 1, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, x):
        return KK.dot(x, self.kernel)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    def get_config(self):
        config = super().get_config()
        config['output_dim'] =  self.output_dim# say self. _localization_net  if you store the argument in __init__
        return config



  
def binary_crossentropy_with_ranking(y_true, y_pred):
    """ Trying to combine ranking loss with numeric precision"""
    # first get the log loss like normal
    logloss = KK.mean(KK.binary_crossentropy( y_true,y_pred), axis=-1)
    # next, build a rank loss
    # clip the probabilities to keep stability
    y_pred_clipped = KK.clip(y_pred, KK.epsilon(), 1-KK.epsilon())
    # translate into the raw scores before the logit
    y_pred_score = KK.log(y_pred_clipped / (1 - y_pred_clipped))
    # determine what the maximum score for a zero outcome is
    y_pred_score_zerooutcome_max = KK.max(tf.boolean_mask(y_pred_score ,(y_true < 1)))
    # determine how much each score is above or below it
    rankloss = y_pred_score - y_pred_score_zerooutcome_max
    # only keep losses for positive outcomes
    rankloss = tf.boolean_mask(rankloss,tf.equal(y_true,1))
    # only keep losses where the score is below the max
    rankloss = KK.square(KK.clip(rankloss, -100, 0))
    # average the loss for just the positive outcomes
    #tf.reduce_sum(tf.cast(myOtherTensor, tf.float32))
    rankloss = KK.sum(rankloss, axis=-1) / (KK.sum(KK.cast(y_true > 0,tf.float32) + 1))
    return (rankloss + 1)* logloss #- an alternative to try
    #return logloss

# PFA, prob false alert for binary classifier  
def binary_PFA(y_true, y_pred, threshold=KK.variable(value=0.5)):  
    y_pred = KK.cast(y_pred >= threshold, 'float32')  
    # N = total number of negative labels  
    N = KK.sum(1 - y_true)  
    # FP = total number of false alerts, alerts from the negative class labels  
    FP = KK.sum(y_pred - y_pred * y_true)  
    return FP/N 


# P_TA prob true alerts for binary classifier  
def binary_PTA(y_true, y_pred, threshold=KK.variable(value=0.5)):  
    y_pred = KK.cast(y_pred >= threshold, 'float32')  
    # P = total number of positive labels  
    P = KK.sum(y_true)  
    # TP = total number of correct alerts, alerts from the positive class labels  
    TP = KK.sum(y_pred * y_true)  
    return TP/P

def auc(y_true, y_pred):  
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)  
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)  
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)  
    binSizes = -(pfas[1:]-pfas[:-1])  
    s = ptas*binSizes  
    return KK.sum(s, axis=0)  


def log_loss(y_true, y_pred):
    """ Trying to combine ranking loss with numeric precision"""
    # first get the log loss like normal
    logloss = KK.sum(KK.binary_crossentropy(y_true,y_pred), axis=-1)
    return logloss
    

def build_model(max_features,continue_cols,K=8,solver='adam',l2=0.0,l2_fm = 0.0,is_self=False):
    np.random.seed(2018)
    inputs = []
    flatten_layers=[]
    columns = range(len(max_features))
    ###------second order term-------###
    for c in columns:
        #print (c,max_features[c])
        inputs_c = Input(shape=(1,), dtype='int32',name = 'input_%s'%(c))
        num_c = max_features[c]
        inputs.append(inputs_c)
        #print (num_c,K,c)
        embed_c = Embedding(
                        num_c,
                        K,
                        input_length=1,
                        name = 'embed_%s'%(c),
                        W_regularizer=l2_reg(l2_fm)
                        )(inputs_c)
        
        #print (embed_c.get_shape(),'---')
        #flatten_c = Flatten()(embed_c)
        flatten_c = Reshape((K,))(embed_c)
        flatten_layers.append(flatten_c)
    inputs_dict = []
    continue_cols_columns=range(len(continue_cols))
    for col in continue_cols_columns:
        #print (col,continue_cols[col])
        inputs_c = Input(shape=(1,), dtype='float',name = 'input_sec_%s'%(col))
        inputs.append(inputs_c)
        inputs_c = BatchNormalization(name='BN_%s'%(col))(inputs_c)
        inputs_dict.append(inputs_c)
        inputs_cK = MyLayer(output_dim = K)(inputs_c)
        flatten_layers.append(inputs_cK)                  #### F * None * K
    summed_features_emb = add(flatten_layers)            ####  None * K
    summed_features_emb_square = multiply([summed_features_emb,summed_features_emb]) ##### None * K
    squared_features_emb = []
    for layer in flatten_layers:
         squared_features_emb.append(multiply([layer,layer]))
    squared_sum_features_emb = add(squared_features_emb)                             ###### None * K
    subtract_layer = Lambda(lambda inputs: inputs[0] - inputs[1],output_shape=lambda shapes: shapes[0])
    y_second_order = subtract_layer([summed_features_emb_square, squared_sum_features_emb])
    y_second_order  = Lambda(lambda x: x * 0.5)(y_second_order)
    y_second_order = Dropout(0.9,seed=2018)(y_second_order)
    ###----first order------######
    fm_layers = []
    for c in columns:
        num_c = max_features[c]
        embed_c = Embedding(
                        num_c,
                        1,
                        input_length=1,
                        name = 'linear_%s'%(c),
                        W_regularizer=l2_reg(l2)
                        )(inputs[c])
        flatten_c = Flatten()(embed_c)
        fm_layers.append(flatten_c)
    for col in continue_cols_columns:
        inputs_c = MyLayer(output_dim = 1)(inputs_dict[col])
        #layer.build(inputs_c.get_shape().as_list())
        #inputs_c = RepeatVector(K)(inputs_c)
        #inputs_c = layer.call(inputs_c)
        fm_layers.append(inputs_c)                #####---- None * 1
    y_first_order = add(fm_layers) 
    y_first_order = BatchNormalization()(y_first_order)
    y_first_order = Dropout(0.8,seed=2018)(y_first_order)
    ##deep 
    y_deep  = concatenate(flatten_layers)         #####    None * (F*K)
    y_deep=Dense(32)(y_deep)
    y_deep = Activation('relu',name='output_1')(y_deep)
    y_deep = Dropout(rate=0.5,seed=2012)(y_deep)
    y_deep=Dense(32)(y_deep)
    y_deep = Activation('relu',name='output_2')(y_deep)
    y_deep = Dropout(rate=0.5,seed=2012)(y_deep)
    concat_input = concatenate([y_first_order,y_second_order,y_deep],axis=1)
#    concat_input=Dense(16)(concat_input)
#    concat_input = Activation('relu',name='concat')(concat_input)
#    #y_deep = Dropout(rate=0.5,seed=2012)(y_deep)
#    concat_input = Dropout(rate=0.5,seed=2012)(concat_input)
    outputs = Dense(1,activation='sigmoid', name='main_output')(concat_input)
    model = Model(inputs=inputs, outputs=outputs,name='model')
    solver = Adam(lr=0.01,decay=0.1)
    if(is_self==True):
        model.compile(optimizer=solver,
                    loss= binary_crossentropy_with_ranking,metrics=[auc,log_loss])
    else:
        model.compile(optimizer=solver,
                    loss= 'binary_crossentropy',metrics=[auc,log_loss])
    #model.fit(X,y,batch_size=batch_size,validation_data=(vali_X,vali_y),epochs=epochs)
    return model



import pandas as pd    
import gc



#dfTest = pd.read_csv("tmp_huanghm_yxjd_dataset_20171208_addleafnodes.csv.gz",usecols=t,dtype={'uid':np.str_},nrows=100000)

dfTrain = pd.read_csv("data/train.csv")
dfTrain = dfTrain.iloc[0:int(0.7*dfTrain.shape[0]),:]
dfTest = dfTrain.iloc[int(0.7*dfTrain.shape[0]):,:]


#dfTrain = pd.read_csv("tmp_huanghm_yxjd_dataset_20171205_20171207_addleafnodes.csv.gz",nrows=1000)

global_columns  = dfTrain.columns.tolist()
ID_columns  = ["ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",]

qid_columns = ['id']
target_columns = ['target']
###----global remove other columns---##
continue_cols = global_columns[:]
[continue_cols.remove(x) for x in target_columns]
[continue_cols.remove(x) for x in ID_columns]
[continue_cols.remove(x) for x in qid_columns]

####continue plus id columns----###
all_feature = ID_columns[:]
all_feature.extend(continue_cols)

col_index = []
for col in  all_feature:
    col_index.append(global_columns.index(col))

target_col =  global_columns.index(target_columns[0])


all_data = dfTrain.append(dfTest)


batch_size = 204800
epochs = 25
###----------make sure the ids max columns number---###
max_features = {}
for i in range(len(ID_columns)):
    max_features[ID_columns[i]]=(all_data[ID_columns[i]].unique().shape[0])        
del all_data
gc.collect()   
max_features_df = pd.DataFrame(data = np.array([list(max_features.keys()),list(max_features.values())]).T,columns=['ids','max_features'],index=range(len(max_features)))
max_features = pd.merge(pd.DataFrame(ID_columns,columns=['ids']),max_features_df,on=['ids'])
max_features.max_features = max_features.max_features.astype(int)
max_features = max_features.max_features.tolist()



####----dump ids to 0-numbers-1
for i in ID_columns:
    dfTrain[i],dfTest[i] = dis_sparse(dfTrain[i].reshape(-1,1),\
           dfTest[i].values.reshape(-1,1),one_hot=False)


###_-----transofrom all the features--
train_x,train_y = dfTrain[all_feature],dfTrain[target_columns]
test_x,test_y = dfTest[all_feature],dfTest[target_columns]
del dfTest
del dfTrain
gc.collect()
#his= clf.fit(train_x.T.values,train_y.values,batch_size=batch_size,\
#                  epochs=epochs,validation_data=(test_x.T.values,test_y.values))
X = train_x.T.values
y = train_y.values
X = [np.array(X[i,:]) for i in range(X.shape[0])]
validation_data=(test_x.T.values,test_y.values)
vali_X,vali_y = validation_data 
vali_X = [np.array(vali_X[i,:]) for i in range(vali_X.shape[0])]

del train_x
del validation_data
gc.collect()



#model = build_model(max_features,continue_cols,K=8,solver='adam',l2=0.0,l2_fm = 0.1,is_self=True)

model= build_model(max_features,continue_cols,K=8,solver='adam',l2=0.1,l2_fm = 0.15)
#from keras.utils import plot_model
#plot_model(model, to_file='DeepFM.png',rankdir='LR')
#####26 
his = model.fit(X,y,batch_size=batch_size,validation_data=(vali_X,vali_y),epochs=epochs)

model.save('point_loss.npy')
model.save_weights('point_loss.weight')
import time
pd.DataFrame(his.history).to_csv("%s_his.csv"%(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))


y_pred_d = model.predict(vali_X)  

from sklearn.metrics import roc_auc_score,log_loss
print (roc_auc_score(vali_y,y_pred_d))
print (log_loss(vali_y,y_pred_d))

