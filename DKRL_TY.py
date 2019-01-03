#coding:utf-8
import tensorflow as tf
import numpy as np
import os
import argparse
import math
import os.path
import timeit
from sklearn import preprocessing
from multiprocessing import JoinableQueue,Queue,Process
from tensorflow.core.framework import summary_pb2

class DKRL:
    @property
    def n_entity(self):
        return self.__n_entity
    @property
    def n_relation(self):
        return self.__n_relation
    @property
    def n_train(self):
        return self.__train_triple.shape[0]
    @property
    def trainable_variables(self):
        return self.__trainable
    @property
    def hr_t(self):
        return self.__hr_t
    @property
    def tr_h(self):
        return self.__tr_h
    @property
    def ht_r(self):
        return self.__ht_r
    @property
    def train_hr_t(self):
        return self.__train_hr_t
    @property
    def train_tr_h(self):
        return self.__train_tr_h
    @property
    def train_ht_r(self):
        return self.__train_ht_r
    @property
    def left_num(self):
        return self.__left_num
    @property
    def right_num(self):
        return self.__right_num
    @property
    def ent_embedding(self):
        return self.__ent_embedding
    @property
    def word_embedding(self):
        return self.__word_embeddings
    @property
    def des_embedding(self):
        return self.__des_embedding
    @property
    def ent_cnn_embedding(self):
        return self.__ent_cnn_embedding
    @property
    def rel_embedding(self):
        return self.__rel_embedding
    
    def get_ent_cnn_embedding(self):
        entity_cnn=self.get_entity_cnn_embedding()
        op=tf.assign(self.__ent_cnn_embedding,entity_cnn)
        return op

    def set_word_embedding(self):
        op=tf.assign(self.__word_embeddings,self.__des_embedding)
        return op

    def raw_training_data(self,batch_size=100):
        n_triple=len(self.__train_triple)
        rand_idx=np.random.permutation(n_triple)
        start=0
        while start<n_triple:
            end=min(start+batch_size,n_triple)
            yield self.__train_triple[rand_idx[start:end]]
            start=end
    
    def testing_data(self,batch_size=100):
        n_triple=len(self.__test_triple)
        start=0
        while start < n_triple:
            end=min(start+batch_size,n_triple)
            yield self.__test_triple[start:end,:]
            start=end

    def validation_data(self, batch_size=100):
        n_triple = len(self.__valid_triple)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__valid_triple[start:end, :]
            start = end
    def E_D_data(self, batch_size=100):
        n_triple = len(self.__E_D_triple)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__E_D_triple[start:end, :]
            start = end

    def D_E_data(self, batch_size=100):
        n_triple = len(self.__D_E_triple)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__D_E_triple[start:end, :]
            start = end

    def D_D_data(self, batch_size=100):
        n_triple = len(self.__D_D_triple)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__D_D_triple[start:end, :]
            start = end
    
    def load_triple(self,path):
        with open(path,'r',encoding='utf-8') as f_triple:
            #(h,t,r)
            return np.asarray([[self.__entity_id_map[triple.replace("\n","").split(" ")[0]],
                               self.__entity_id_map[triple.replace("\n","").split(" ")[1]],
                               self.__relation_id_map[triple.replace("\n","").split(" ")[2]]] for triple in f_triple.readlines()],
                               dtype=np.int32)
    def gen_relation_attr(self):
        
        left_entity={}
        right_entity={}
        left_num={}
        right_num={}
        hrt=self.__train_triple
        relation_data=self.__relation_id_map

        for (r,r_idx) in relation_data.items():
            left_entity[r_idx]=dict()
            right_entity[r_idx]=dict()

        for h,t,r in hrt:
            if t not in right_entity[r]:
                right_entity[r][t]=0
            if h not in left_entity[r]:
                left_entity[r][h]=0

            right_entity[r][t] = right_entity[r][t] + 1
            left_entity[r][h] = left_entity[r][h] + 1
        
        for (r,r_idx) in relation_data.items():
            left_sum1=left_sum2=0
            right_sum1=right_sum2=0
            for (entity,count) in right_entity[r_idx].items():
                right_sum1=right_sum1+1
                right_sum2=right_sum2+count
            right_num[r_idx]=right_sum2/right_sum1

            for (entity,count) in left_entity[r_idx].items():
                left_sum1=left_sum1+1
                left_sum2=left_sum2+count
            left_num[r_idx]=left_sum2/left_sum1

            #print('%.3f %.3f %.3f'%(r_idx,right_num[r_idx],left_num[r_idx]))
       
        return left_num,right_num

    #map<int,map<int,set<int> > >
    def gen_hr_t(self,triple_data):
        hr_t=dict()
        for h,t,r in triple_data:
            if h not in hr_t:
                hr_t[h]=dict()
            if r not in hr_t[h]:
                hr_t[h][r]=set()
            hr_t[h][r].add(t)
        
        return hr_t
    
    def gen_tr_h(self,triple_data):
        tr_h=dict()
        for h,t,r in triple_data:
            if t not in tr_h:
                tr_h[t]=dict()
            if r not in tr_h[t]:
                tr_h[t][r]=set()
            tr_h[t][r].add(h)
    
        return tr_h
    
    def gen_ht_r(self,triple_data):
        ht_r=dict()
        for h,t,r in triple_data:
            if h not in ht_r:
                ht_r[h]=dict()
            if t not in ht_r[h]:
                ht_r[h][t]=set()
            ht_r[h][t].add(r)
    
        return ht_r

    def train(self,inputs,scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            if self.__initialized:
                scp.reuse_variables()
            
            pos_triple,neg_triple=inputs

            """
            FB15k_word_pos=tf.reshape(self.__FB15k_entityWords,[-1,1])
            #根据词的序号获得对应的词向量
            FB15k_des=tf.nn.embedding_lookup(self.__word_embeddings, FB15k_word_pos)
            #reshape 词向量矩阵
            FB15k_entity_des=tf.reshape(FB15k_des,[-1,self.__avg_des_len,self.n_w,1])
            """
            FB15k_entityWords=tf.slice(self.__entityWords,[0,0],[self.__FB15k_entity,self.n_w])

            FB15k_word_pos=tf.reshape(FB15k_entityWords,[-1,1])
            #根据词的序号获得对应的词向量
            FB15k_des=tf.nn.embedding_lookup(self.__word_embeddings,FB15k_word_pos)
            #reshape 词向量矩阵
            FB15k_entity_des=tf.reshape(FB15k_des,[-1,self.__avg_des_len,self.n_w,1])

            #conv1
            h_conv1=self.conv2d(FB15k_entity_des,self.W_conv1,self.n_w)+self.b_conv1

            #batch normalization
            
            axes=list(range(len(h_conv1.get_shape())-1))

            h_mean1,h_variance1=tf.nn.moments(h_conv1,axes)

            h_conv1_norm=tf.nn.batch_normalization(h_conv1,h_mean1,h_variance1,offset=self.__beta1,scale=self.__gamma1,variance_epsilon=0.001)
            
            h_conv1_act=tf.nn.relu(h_conv1_norm)

            h_pool1=self.max_pool(h_conv1_act,self.n_pooling)

            #h_pool1_drop=tf.nn.dropout(h_pool1,0.5)

            #conv2

            h_conv2=self.conv2d(h_pool1,self.W_conv2,self.n_1)+self.b_conv2

            #h_conv2_drop=tf.nn.dropout(h_conv2,0.5)

            #batch normalization
            
            h_mean2,h_variance2=tf.nn.moments(h_conv2,axes)

            h_conv2_norm=tf.nn.batch_normalization(h_conv2,h_mean2,h_variance2,offset=self.__beta2,scale=self.__gamma2,variance_epsilon=0.001)
            
            h_conv2_act=tf.nn.relu(h_conv2_norm)

            h_pool2=self.avg_pool(h_conv2_act,self.avg_pooling_len)

            entity_cnn_embedding=tf.reshape(h_pool2,[-1,self.n])
            
            #正样本
            pos_h_e = tf.nn.embedding_lookup(entity_cnn_embedding,pos_triple[:,0])
            pos_t_e = tf.nn.embedding_lookup(entity_cnn_embedding,pos_triple[:,1])
            pos_r_e = tf.nn.embedding_lookup(self.__rel_embedding,pos_triple[:,2])

            #负样本
            neg_h_e = tf.nn.embedding_lookup(entity_cnn_embedding,neg_triple[:,0])
            neg_t_e = tf.nn.embedding_lookup(entity_cnn_embedding,neg_triple[:,1])
            neg_r_e = tf.nn.embedding_lookup(self.__rel_embedding,neg_triple[:,2])

            if self.__L1_flag:
                pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e),1,keep_dims=True)
                neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e),1,keep_dims=True)
            else:
                pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e)**2,1,keep_dims=True)
                neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e)**2,1,keep_dims=True)

            loss = tf.reduce_sum(tf.maximum(pos - neg + self.__margin, 0))     
            return loss

    #inputs:shape (?,3)
    def test(self,_inputs,scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            scp.reuse_variables()

            #获取实体(h,t)的CNN向量以及关系向量
            print("Get entity's cnn embedding and relation's embedding")

            inputs=tf.reshape(_inputs,[-1,3,1])

            h=tf.nn.embedding_lookup(self.__ent_cnn_embedding,inputs[:,0])
            t=tf.nn.embedding_lookup(self.__ent_cnn_embedding,inputs[:,1])
            r=tf.nn.embedding_lookup(self.__rel_embedding,inputs[:,2])

            hrt_res=tf.reduce_sum(-abs(h+r-self.__ent_cnn_embedding),2)
            trh_res=tf.reduce_sum(-abs(r-t+self.__ent_cnn_embedding),2)
            htr_res=tf.reduce_sum(-abs(h-t+self.__rel_embedding),2)

            _,tail_ids=tf.nn.top_k(hrt_res,k=self.__n_entity)
            _,head_ids=tf.nn.top_k(trh_res,k=self.__n_entity)
            _,relation_ids=tf.nn.top_k(htr_res,k=self.__n_relation)
        
        return head_ids,tail_ids,relation_ids

    def normalize_embedding(self,scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            scp.reuse_variables()
            #normalize_entity_op = self.__ent_cnn_embedding.assign(tf.clip_by_norm(self.__ent_cnn_embedding, clip_norm=1, axes=1))
            normalize_relation_op = self.__rel_embedding.assign(tf.clip_by_norm(self.__rel_embedding, clip_norm=1, axes=1))
            normalize_word_embedding_op =self.__word_embeddings.assign(tf.clip_by_norm(self.__word_embeddings,clip_norm=1,axes=1))
            #return normalize_entity_op,normalize_relation_op
            return normalize_word_embedding_op,normalize_relation_op

    def weight_variable(self,shape,name,bound):
        initial = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(name=name,shape=shape,initializer=initial)

    def bias_variable(self,shape,name,bound):
        initial = tf.constant_initializer(value=0.0, dtype=tf.float32)
        return tf.get_variable(name=name,shape=shape,initializer=initial)

    #padding='SAME'
    #卷积核的步长为[1,1,n,1]
    def conv2d(self,x,W,n):
        return tf.nn.conv2d(x,W,strides=[1,1,n,1],padding='SAME')

    def max_pool(self,x,y):
        return tf.nn.max_pool(x,ksize=[1,y,1,1],strides=[1,y,1,1],padding='SAME')

    def avg_pool(self,x,y):
        return tf.nn.avg_pool(x,ksize=[1,y,1,1],strides=[1,y,1,1],padding='VALID')

    def get_entity_cnn_embedding(self,scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            scp.reuse_variables()

            All_word_pos=tf.reshape(self.__entityWords,[-1,1])
            #根据词的序号获得对应的词向量
            All_des=tf.nn.embedding_lookup(self.__word_embeddings,All_word_pos)
            #reshape 词向量矩阵
            All_entity_des=tf.reshape(All_des,[-1,self.__avg_des_len,self.n_w,1])
            #conv1
            h_conv1=self.conv2d(All_entity_des,self.W_conv1,self.n_w)+self.b_conv1

            #batch normalization
            
            axes=list(range(len(h_conv1.get_shape())-1))

            h_mean1,h_variance1=tf.nn.moments(h_conv1,axes)

            h_conv1_norm=tf.nn.batch_normalization(h_conv1,h_mean1,h_variance1,offset=self.__beta1,scale=self.__gamma1,variance_epsilon=0.001)

            h_conv1_act=tf.nn.relu(h_conv1_norm)

            h_pool1=self.max_pool(h_conv1_act,self.n_pooling)
     
            #conv2

            h_conv2=self.conv2d(h_pool1,self.W_conv2,self.n_1)+self.b_conv2

            #batch normalization
            
            h_mean2,h_variance2=tf.nn.moments(h_conv2,axes)

            h_conv2_norm=tf.nn.batch_normalization(h_conv2,h_mean2,h_variance2,offset=self.__beta2,scale=self.__gamma2,variance_epsilon=0.001)
            
            h_conv2_act=tf.nn.relu(h_conv2_norm)

            h_pool2=self.avg_pool(h_conv2_act,self.avg_pooling_len)

            entity_cnn=tf.reshape(h_pool2,[-1,self.n])

            op=tf.assign(self.__ent_cnn_embedding,entity_cnn)
	
            return op

    def __init__(self,data_dir,train_batch,eval_batch,L1_flag,margin):
        
        self.__L1_flag = L1_flag
        self.__margin = margin
        self.__train_batch = train_batch
        self.__eval_batch = eval_batch
        self.__initialized = False
        self.__trainable = list()
        self.window_1 = 2
        self.window_2 = 2 
        self.n_w = 100 # input embedding size
        self.n_1 = 100 # second embedding size
        self.n = 100 # output embedding size
        self.n_pooling=4

        with open(os.path.join(data_dir,'entity2id.txt'),'r',encoding='utf-8') as f:
            self.__FB15k_entity_id_map={x.strip().split(' ')[0]: int(x.strip().split(' ')[1]) for x in f.readlines()}

        self.__FB15k_entity=len(self.__FB15k_entity_id_map)

        with open(os.path.join(data_dir,'total_entity2id.txt'),'r',encoding='utf-8') as f:
            self.__entity_id_map={x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
            self.__id_entity_map={v: k for k,v in self.__entity_id_map.items()}
        
        self.__n_entity=len(self.__entity_id_map)

        print("FB15K ENTITY: %d" % self.__FB15k_entity)

        print("TOTAL ENTITY: %d" % self.__n_entity)

        with open(os.path.join(data_dir, 'relation2id.txt'), 'r', encoding='utf-8') as f:
            self.__relation_id_map = {x.strip().split(' ')[0]: int(x.strip().split(' ')[1]) for x in f.readlines()}
            self.__id_relation_map = {v: k for k, v in self.__relation_id_map.items()}
        
        self.__n_relation = len(self.__relation_id_map)

        print("RELATION: %d" % self.__n_relation)

        with open(os.path.join(data_dir,'total_word2id.txt'),'r',encoding='utf-8') as f:
            self.__word_id_map = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
            self.__id_word_map = {v: k for k,v in self.__word_id_map.items()}

        self.__n_word=len(self.__word_id_map)

        #add zero-padding (word,num)
        self.__word_id_map['zeropadding'] = self.__n_word
        self.__id_word_map[self.__n_word] = 'zeropadding'
        self.__n_word = self.__n_word + 1

        print("WORD: %d" % self.__n_word)

        print('STATR TO GET ENTITY_WORD')
 
        with open(os.path.join(data_dir,'total_entityWords.txt'),'r',encoding='utf-8') as f:
           self.__max_des_len=max([int(x.replace("\t"," ").split(" ")[1]) for x in f.readlines()])

        with open(os.path.join(data_dir,'total_entityWords.txt'),'r',encoding='utf-8') as f:
           self.__min_des_len=min([int(x.replace("\t"," ").split(" ")[1]) for x in f.readlines()])
        
        print("MAX_DES_LEN: %d" % self.__max_des_len)

        print("MIN_DES_LEN: %d" % self.__min_des_len)

        self.__avg_des_len = 150

        """
        self.__FB15k_entityWords=np.empty((self.__FB15k_entity,self.__avg_des_len),dtype=np.int32)

        with open(os.path.join(data_dir,'FB15k_entityWords.txt'),'r',encoding='utf-8') as f:
            entity_word_line=[entity_word.replace("\t"," ").replace("\n","").split(" ") for entity_word in f.readlines()]
            #(entity_name,word_num,word1...wordn)
            for entity_word in entity_word_line:
                entity_name=entity_word[0]
                word_num=int(entity_word[1])
                if word_num>self.__avg_des_len:
                    word_num=self.__avg_des_len
                for x in range(word_num):
                    self.__FB15k_entityWords[self.__FB15k_entity_id_map[entity_name]][x]=self.__word_id_map[entity_word[x+2]]
                if word_num < self.__avg_des_len:
                    for x in range(self.__avg_des_len-word_num):
                        self.__FB15k_entityWords[self.__FB15k_entity_id_map[entity_name]][x+word_num]=self.__n_word-1 #add zero-padding's num
        """
        self.__entityWords=np.empty((self.__n_entity,self.__avg_des_len),dtype=np.int32)

        with open(os.path.join(data_dir,'total_entityWords.txt'),'r',encoding='utf-8') as f:
            entity_word_line=[entity_word.replace("\t"," ").replace("\n","").split(" ") for entity_word in f.readlines()]
            #(entity_name,word_num,word1...wordn)
            for entity_word in entity_word_line:
                entity_name=entity_word[0]
                word_num=int(entity_word[1])
                if word_num>self.__avg_des_len:
                    word_num=self.__avg_des_len
                for x in range(word_num):
                    self.__entityWords[self.__entity_id_map[entity_name]][x]=self.__word_id_map[entity_word[x+2]]
                if word_num < self.__avg_des_len:
                    for x in range(self.__avg_des_len-word_num):
                        self.__entityWords[self.__entity_id_map[entity_name]][x+word_num]=self.__n_word-1 #add zero-padding's num

        print('ALL ENTITY_WORD IS OBTAINED')

        print(self.__FB15k_entityWords)

        print(self.__entityWords)

        #pretrained embedding by word2vec

        vecdir="vec_"+str(self.n_w)+".txt"

        print('START TO GET WORD EMBEDDING')

        self.__des_embedding=np.random.randn(self.__n_word,self.n_w).astype(np.float32)

        with open(os.path.join(data_dir,vecdir),'r',encoding='utf-8') as f:

            word_embedding_line=[word_embedding.replace("\n","").split(" ") for word_embedding in f.readlines()]

            for embedding in word_embedding_line:
                word=embedding[0]
                for x in range(self.n_w):
                    #if self.__word_id_map.has_key(word)==False:
                    if word not in self.__word_id_map:
                        continue
                    else:
                        self.__des_embedding[self.__word_id_map[word]][x]=float(embedding[x+1])
           
            # add zero-padding: all number is zero
            for x in range(self.n_w):
                self.__des_embedding[self.__word_id_map['zeropadding']][x]=0.0

        self.__entityWords=tf.convert_to_tensor(self.__entityWords,dtype=tf.int32)

        #self.__FB15k_entityWords=tf.convert_to_tensor(self.__FB15k_entityWords,dtype=tf.int32)

        print('ALL WORD_EMBEDDING IS OBTAINED')

        self.__train_triple=self.load_triple(os.path.join(data_dir,"train.txt"))
        print("TRAIN_TRIPLES: %d" % self.__train_triple.shape[0])

        self.__valid_triple=self.load_triple(os.path.join(data_dir,"valid.txt"))
        print("VALID_TRIPLES: %d" % self.__valid_triple.shape[0])

        self.__test_triple=self.load_triple(os.path.join(data_dir,"test.txt"))
        print("TEST_TRIPLES: %d" % self.__test_triple.shape[0])

        self.__E_D_triple=self.load_triple(os.path.join(data_dir,"e-d.txt"))
        print("E-D_TRIPLES: %d" % self.__E_D_triple.shape[0])

        self.__D_E_triple=self.load_triple(os.path.join(data_dir,"d-e.txt"))
        print("D-E_TRIPLES: %d" % self.__D_E_triple.shape[0])

        self.__D_D_triple=self.load_triple(os.path.join(data_dir,"d-d.txt"))
        print("D-D_TRIPLES: %d" % self.__D_D_triple.shape[0])

        self.__train_hr_t=self.gen_hr_t(self.__train_triple)
        self.__train_tr_h=self.gen_tr_h(self.__train_triple)
        self.__train_ht_r=self.gen_ht_r(self.__train_triple)

        """
        self.__hr_t=self.gen_hr_t(np.concatenate([self.__train_triple,self.__valid_triple,self.__test_triple],axis=0))
        self.__tr_h=self.gen_tr_h(np.concatenate([self.__train_triple,self.__valid_triple,self.__test_triple],axis=0))
        self.__ht_r=self.gen_ht_r(np.concatenate([self.__train_triple,self.__valid_triple,self.__test_triple],axis=0))
        """

        self.__hr_t=self.gen_hr_t(np.concatenate([self.__train_triple,self.__valid_triple,self.__test_triple,self.__E_D_triple,self.__D_E_triple,self.__D_D_triple],axis=0))
        self.__tr_h=self.gen_tr_h(np.concatenate([self.__train_triple,self.__valid_triple,self.__test_triple,self.__E_D_triple,self.__D_E_triple,self.__D_D_triple],axis=0))
        self.__ht_r=self.gen_ht_r(np.concatenate([self.__train_triple,self.__valid_triple,self.__test_triple,self.__E_D_triple,self.__D_E_triple,self.__D_D_triple],axis=0))

        self.__left_num,self.__right_num=self.gen_relation_attr()
      
        bound=6.0/math.sqrt(self.n)

        with tf.device('/gpu'):

             self.avg_pooling_len=math.ceil(self.__avg_des_len/self.n_pooling)
             
             #n_l 个 window_1 x n_w 的卷积核+偏置项
             self.W_conv1=self.weight_variable(shape=[self.window_1,self.n_w,1,self.n_1],name="conv1_w",bound=bound)

             self.b_conv1=self.bias_variable(shape=[self.n_1],name="conv1_b",bound=bound)

             #n 个 window_2 x n_l 的卷积核+偏置项
             self.W_conv2=self.weight_variable(shape=[self.window_2,1,self.n_1,self.n],name="conv2_w",bound=bound)

             self.b_conv2=self.bias_variable(shape=[self.n],name="conv2_b",bound=bound)

             tf.summary.histogram('layer1/conv_w',self.W_conv1)
             tf.summary.histogram('layer2/conv_w',self.W_conv2)
             tf.summary.histogram('layer1/conv_b',self.b_conv1)
             tf.summary.histogram('layer2/conv_b',self.b_conv2)

             self.__ent_embedding = tf.get_variable(name = "ent_embedding", shape = [self.__n_entity, self.n],
                                     initializer = tf.random_uniform_initializer(minval=-bound,maxval=bound,seed=344))

             self.__ent_cnn_embedding = tf.get_variable(name = "ent_cnn_embedding", shape = [self.__n_entity, self.n],
                                     initializer = tf.random_uniform_initializer(minval=-bound,maxval=bound,seed=345),trainable=False)

             self.__rel_embedding = tf.get_variable(name = "rel_embedding", shape = [self.__n_relation, self.n],
                                     initializer = tf.random_uniform_initializer(minval=-bound,maxval=bound,seed=346))

             self.__word_embeddings = tf.get_variable(name = "word_embeddings", shape = [self.__n_word,self.n_w],
                                     initializer = tf.random_uniform_initializer(minval=-bound,maxval=bound,seed=347))

             self.__beta1 = tf.Variable(tf.constant(0.0, shape=[self.n_1]),name='beta1')
             self.__gamma1 = tf.Variable(tf.constant(1.0, shape=[self.n_1]),name='gamma1')

             self.__beta2 = tf.Variable(tf.constant(0.0, shape=[self.n]),name='beta2')
             self.__gamma2 = tf.Variable(tf.constant(1.0, shape=[self.n]),name='gamma2')

             #self.__word_norm_embedding=preprocessing.normalize(self.des_embedding,norm='l2')

             self.__trainable.append(self.W_conv1)
             self.__trainable.append(self.W_conv2)
             self.__trainable.append(self.__rel_embedding)
             self.__trainable.append(self.__word_embeddings)

             self.__trainable.append(self.__beta1)
             self.__trainable.append(self.__gamma1)
             self.__trainable.append(self.__beta2)
             self.__trainable.append(self.__gamma2)

             #self.__trainable.append(self.b_conv1)
             #self.__trainable.append(self.b_conv2)
             #self.__trainable.append(self.__ent_embedding)
             #self.__trainable.append(self.__word_norm_embedding)
             

def train_ops(model: DKRL,learning_rate=0.01,optimizer_str='adma'):
    with tf.device('/gpu'):

        pos_triple=tf.placeholder(tf.int32,[None,3])
        neg_triple=tf.placeholder(tf.int32,[None,3])

        train_loss=model.train([pos_triple,neg_triple])

        init_word_embedding_op = model.set_word_embedding()
 
        if optimizer_str == 'gradient':
            optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer_str == 'adam':
            optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer_str == 'rms':
            optimizer=tf.train.RMSPropOptmizer(learning_rate=learning_rate)
        else:
            raise NotImplementedError("Don's support %s optmizer" % optimizer_str)
        
        grads=optimizer.compute_gradients(train_loss,model.trainable_variables)

        op_train=optimizer.apply_gradients(grads)

        return pos_triple,neg_triple,train_loss,op_train,init_word_embedding_op

def test_ops(model: DKRL):
    with tf.device('/cpu'):
        test_input=tf.placeholder(tf.int32,[None,3])
        head_ids,tail_ids,relation_ids=model.test(test_input) 
    return test_input,head_ids,tail_ids,relation_ids

def normalize_ops(model: DKRL):
    with tf.device('/gpu'):
        return model.normalize_embedding()

def cal_ent_cnn_ops(model: DKRL):
    with tf.device('/gpu'):
        set_entity_cnn_op=model.get_entity_cnn_embedding()
        return set_entity_cnn_op
    
def worker_func(in_queue: JoinableQueue, out_queue: Queue,tr_h,hr_t,ht_r):
    while True:
        dat=in_queue.get()
        if dat is None:
            in_queue.task_done()
            continue
        testing_data,head_pred,tail_pred,relation_pred=dat
        out_queue.put(test_evaluation(testing_data,head_pred,tail_pred,relation_pred,tr_h,hr_t,ht_r))
        in_queue.task_done()

def data_generator_func(in_queue: JoinableQueue,out_queue: Queue,right_num,left_num,tr_h,hr_t,ht_r,n_entity,n_relation):
    while True:
        dat = in_queue.get()
        if dat is None:
            break
        
        pos_triple_batch=dat.copy()
        neg_triple_batch=dat.copy()
        htr=dat.copy()

        #construct negative-triple
        for idx in range(htr.shape[0]):
            h=htr[idx,0]
            t=htr[idx,1]
            r=htr[idx,2]
            if np.random.uniform(0,1) < 0.5:
                # h t predict r
                tmp_r=np.random.randint(0,n_relation-1)
                while tmp_r in ht_r[h][t]:
                    tmp_r=np.random.randint(0,n_relation-1)
                neg_triple_batch[idx,2]=tmp_r
            else:
                prob=left_num[r]/(left_num[r]+right_num[r])
                #prob = 0.5
                # t r predict h
                if np.random.uniform(0,1) < prob: 
                    tmp_h=np.random.randint(0,n_entity-1)
                    while tmp_h in tr_h[t][r]:
                        tmp_h=np.random.randint(0,n_entity-1)
                    neg_triple_batch[idx,0]=tmp_h
            # h r predict t    
                else: 
                    tmp_t=np.random.randint(0,n_entity-1)
                    while tmp_t in hr_t[h][r]:
                        tmp_t=np.random.randint(0,n_entity-1)
                    neg_triple_batch[idx,1]=tmp_t
            """
            prob=left_num[r]/(left_num[r]+right_num[r])
            #prob = 0.5
            # t r predict h
            if np.random.uniform(0,1) < prob: 
                tmp_h=np.random.randint(0,n_entity-1)
                while tmp_h in tr_h[t][r]:
                    tmp_h=np.random.randint(0,n_entity-1)
                neg_triple_batch[idx,0]=tmp_h
            # h r predict t    
            else: 
                tmp_t=np.random.randint(0,n_entity-1)
                while tmp_t in hr_t[h][r]:
                    tmp_t=np.random.randint(0,n_entity-1)
                neg_triple_batch[idx,1]=tmp_t
            """
        out_queue.put((pos_triple_batch,neg_triple_batch))
    
def test_evaluation(testing_data,head_pred,tail_pred,relation_pred,tr_h,hr_t,ht_r):
    assert len(testing_data)==len(head_pred)
    assert len(testing_data)==len(tail_pred)
    assert len(testing_data)==len(relation_pred)

    mean_rank_h=list()
    mean_rank_t=list()
    mean_rank_r=list()

    filtered_mean_rank_h=list()
    filtered_mean_rank_t=list()
    filtered_mean_rank_r=list()

    testing_len=len(testing_data)    
    
    for i in range(testing_len):
        h=testing_data[i,0]
        t=testing_data[i,1]
        r=testing_data[i,2]
	
	# mean rank - predict head entity
        mr=0
        for val in head_pred[i]:
            if val == h:
                mean_rank_h.append(mr)
                break
            mr+=1

        # mean rank - predict tail entity
        mr = 0
        for val in tail_pred[i]:
            if val == t:
                mean_rank_t.append(mr)
                break
            mr+=1

        # mean rank - predict relation 
        mr = 0
        for val in relation_pred[i]:
            if val == r:
                mean_rank_r.append(mr)
                break
            mr+=1

        #filtered mean rank - predict head entity
        fmr=0
        for val in head_pred[i]:
            if val == h:
                filtered_mean_rank_h.append(fmr)
                break
            if t in tr_h and r in tr_h[t] and val in tr_h[t][r]:
                continue
            else:
                fmr += 1

     	#filtered mean rank - predict tail entity
        fmr=0
        for val in tail_pred[i]:
            if val == t:
                filtered_mean_rank_t.append(fmr)
                break
            if h in hr_t and r in hr_t[h] and val in hr_t[h][r]:
                continue
            else:
                fmr += 1
        
        #filtered mean rank - predict relation
        fmr=0
        for val in relation_pred[i]:
            if val == r:
                filtered_mean_rank_r.append(fmr)
                break
            if h in ht_r and t in ht_r[h] and val in ht_r[h][t]:
                continue
            else:
                fmr += 1

    return (mean_rank_h,filtered_mean_rank_h),(mean_rank_t,filtered_mean_rank_t),(mean_rank_r,filtered_mean_rank_r) 

def main(_):

    parser = argparse.ArgumentParser(description='DKRL.')
    parser.add_argument('--lr', dest='lr', type=float, help="Learning rate", default=0.001)
    parser.add_argument('--L1_flag', dest='L1_flag', type=int, help="norm method", default=1)
    parser.add_argument('--margin', dest='margin', type=int, help="margin", default=1)
    parser.add_argument('--data', dest='data_dir', type=str, help="Data folder", default='./data/DKRL/')
    parser.add_argument("--dim", dest='dim', type=int, help="Embedding dimension", default=100)
    parser.add_argument("--worker", dest='n_worker', type=int, help="Evaluation worker", default=3)
    parser.add_argument("--load_model", dest='load_model', type=str, help="Model file:xxx.meta", default="")
    parser.add_argument("--max_iter", dest='max_iter', type=int, help="Max iteration", default=501)
    parser.add_argument("--train_batch", dest="train_batch", type=int, help="Training batch size", default=5000) 
    parser.add_argument("--eval_batch", dest="eval_batch", type=int, help="Evaluation batch size", default=500)
    parser.add_argument("--optimizer", dest='optimizer', type=str, help="Optimizer", default='adam')

    parser.add_argument("--generator", dest='n_generator', type=int, help="Data generator", default=10)
    parser.add_argument("--save_dir", dest='save_dir', type=str, help="Model path", default='./')
    parser.add_argument("--save_per", dest='save_per', type=int, help="Save per x iteration", default=50)
    parser.add_argument("--eval_per", dest='eval_per', type=int, help="Evaluate every x iteration", default=50)
    parser.add_argument("--summary_dir", dest='summary_dir', type=str, help="summary directory",
                        default='./ProjE_summary/')
    parser.add_argument("--keep", dest='drop_out', type=float, help="Keep prob (1.0 keep all, 0. drop all)",
                        default=0.5)
    parser.add_argument("--prefix", dest='prefix', type=str, help="model_prefix", default='DEFAULT')

    args=parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    print("init DKRL model")

    model=DKRL(data_dir=args.data_dir,train_batch=args.train_batch,eval_batch=args.eval_batch,
                L1_flag=args.L1_flag,margin=args.margin)

    print("construct train ops")

    pos_triple,neg_triple,train_loss,train_op,init_word_embedding_op = train_ops(model,learning_rate=args.lr,optimizer_str=args.optimizer)

    print("construct test ops")

    test_input,test_head,test_tail,test_relation = test_ops(model)

    normalize_word_embedding_op,normalize_relation_op = normalize_ops(model)

    cal_all_ent_cnn = cal_ent_cnn_ops(model)

    init = tf.global_variables_initializer()

    config = tf.ConfigProto() 
    
    config.gpu_options.allow_growth = True

    config.allow_soft_placement = True

    saver=tf.train.Saver()

    with tf.Session(config=config) as sess:
            #print(tf.get_default_graph().as_graph_def()) #查看graph中的constant变量
            conv1_w=tf.get_default_graph().get_tensor_by_name("conv1_w:0")
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('./DKRL_graphs/neg_both',sess.graph)
            sess.run(init)
            iter_offset = 0
            if args.load_model is not None and os.path.exists(args.load_model):                
                print("start to load model")
                #saver=tf.train.import_meta_graph(args.load_model)
                saver.restore(sess,tf.train.latest_checkpoint('./'))
                print(sess.run(conv1_w))
                iter_offset = int(args.load_model.split('.')[-3].split('_')[-1]) + 1
                print("Load model from %s,iteration %d restored."%(args.load_model,iter_offset))
            else:
                #saver = tf.train.Saver()
                #start to train
                sess.run([init_word_embedding_op])

            total_inst=model.n_train

            #generate training data
            raw_training_data_queue = Queue()
            training_data_queue = Queue()
            data_generators = list()

            for i in range(args.n_generator):
                data_generators.append(Process(target=data_generator_func,args=(
                                raw_training_data_queue,training_data_queue,model.right_num,model.left_num,model.train_tr_h,model.train_hr_t,model.train_ht_r,model.n_entity,model.n_relation)))
                data_generators[-1].start()

            evaluation_queue=JoinableQueue()
            result_queue=Queue()

            for i in range(args.n_worker):
                worker=Process(target=worker_func,args=(evaluation_queue,result_queue,
                               model.tr_h,model.hr_t,model.ht_r))
                worker.start()
                print("work %d start!"% i)

            for n_iter in range(iter_offset,args.max_iter):

                start_time = timeit.default_timer()
                total_loss =0.0
                ninst = 0

                print('initializing raw training data...')
                nbatches_count = 0
                for dat in model.raw_training_data(batch_size=args.train_batch):
                    raw_training_data_queue.put(dat)
                    nbatches_count += 1
                print("raw training data initialized.")

                while nbatches_count > 0:

                    nbatches_count -= 1

                    pos_triple_batch,neg_triple_batch = training_data_queue.get()

                    summary,loss, _= sess.run([merged,train_loss,train_op], feed_dict={pos_triple:pos_triple_batch,
                                                                        neg_triple:neg_triple_batch})
                    sess.run([normalize_relation_op])           
                                                                     
                    total_loss += loss

                    ninst += pos_triple_batch.shape[0]

                    if ninst % (10000) is not None:
                       
                        print(
                        '[%d sec](%d/%d) : %.2f -- loss : %.5f ' % (
                            timeit.default_timer() - start_time, ninst, total_inst, float(ninst) / total_inst,
                            loss / (pos_triple_batch.shape[0])),end='\r')
                    
                print("")
                print("iter %d avg loss %.5f, time %.3f" % (n_iter, total_loss / ninst, timeit.default_timer() - start_time))

                value = summary_pb2.Summary.Value(tag="average_loss", simple_value=total_loss/ninst)
                loss_summary = summary_pb2.Summary(value=[value])
                #total_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="total_loss",simple_value=total_loss)])

                writer.add_summary(loss_summary,n_iter)

                #writer.add_summary(total_loss_summary,n_iter)

                writer.add_summary(summary,n_iter)                                                            

                if n_iter % args.save_per == 0 or n_iter == args.max_iter - 1:
                    save_path=saver.save(sess,os.path.join(
                        args.save_dir,"DKRL_"+str(args.prefix)+"_"+str(n_iter)+".ckpt"))
                    print("DKRL Model saved at %s" % save_path)
                

                if n_iter!=0 and (n_iter % args.eval_per == 0 or n_iter == args.max_iter - 1):

                    test_start_time=timeit.default_timer()

                    sess.run([cal_all_ent_cnn])
                   
                    for data_func,test_type in zip([model.testing_data, model.E_D_data, model.D_E_data, model.D_D_data],['TEST','E-D','D-E','D-D']):
                    #for data_func,test_type in zip([model.validation_data],['VALID']):
                        
                        accu_mean_rank_h = list()
                        accu_mean_rank_t = list()
                        accu_mean_rank_r = list()
                        accu_filtered_mean_rank_h = list()
                        accu_filtered_mean_rank_t = list()
                        accu_filtered_mean_rank_r = list()
                 
                        evaluation_count = 0

                        for testing_data in data_func(batch_size=args.eval_batch):
                            head_pred,tail_pred,relation_pred=sess.run([test_head,test_tail,test_relation],
                                                    {test_input: testing_data})
                     
                            evaluation_queue.put((testing_data,head_pred,tail_pred,relation_pred))
                            evaluation_count += 1
                 
                        for i in range(args.n_worker):
                            evaluation_queue.put(None)

                        print("waiting for worker finishes their work")
                        evaluation_queue.join()
                        print("all worker stopped.")

                        while evaluation_count > 0:
                             evaluation_count -= 1

                             (mrh,fmrh),(mrt,fmrt),(mrr,fmrr) = result_queue.get()
                             accu_mean_rank_h += mrh
                             accu_mean_rank_t += mrt
                             accu_mean_rank_r += mrr

                             accu_filtered_mean_rank_h += fmrh
                             accu_filtered_mean_rank_t += fmrt
                             accu_filtered_mean_rank_r += fmrr
                        

                        head_result = np.mean(np.asarray(accu_mean_rank_h,dtype=np.int32)<10)
                        head_filtered_result = np.mean(np.asarray(accu_filtered_mean_rank_h,dtype=np.int32)<10)
                        tail_result = np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 10)
                        tail_filtered_result = np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 10)
                        average_filtered_result = (head_filtered_result+tail_filtered_result)/2.0
                        
                        """
                        if test_type=='E-D':
                            E_D_result = tf.Summary(value=[tf.Summary.Value(tag="E_D",simple_value=average_filtered_result)])
                            writer.add_summary(E_D_result,n_iter)
                        elif test_type=='D_E':
                            D_E_result = tf.Summary(value=[tf.Summary.Value(tag="D_E",simple_value=average_filtered_result)])
                            writer.add_summary(D_E_result,n_iter)
                        elif test_type=='D_D':
                            D_D_result = tf.Summary(value=[tf.Summary.Value(tag="D_D",simple_value=average_filtered_result)])
                            writer.add_summary(D_D_result,n_iter)
                        else:
                            test_result = tf.Summary(value=[tf.Summary.Value(tag="test",simple_value=average_filtered_result)])
                            writer.add_summary(test_result,n_iter)
                        """
                        test_result = tf.Summary(value=[tf.Summary.Value(tag=test_type,simple_value=average_filtered_result)])
                        writer.add_summary(test_result,n_iter)

                        print('cost time:[%.3f sec]'%(timeit.default_timer()-test_start_time))

                        print(
                            "[%s] INITIALIZATION [HEAD PREDICTION] MEAN RANK: %.1f FILTERED MEAN RANK %.1f HIT@10 %.3f FILTERED HIT@10 %.3f" %
                            (test_type,np.mean(accu_mean_rank_h),
                            np.mean(accu_filtered_mean_rank_h),
                            head_result,head_filtered_result))

                        print(
                            "[%s] INITIALIZATION [TAIL PREDICTION] MEAN RANK: %.1f FILTERED MEAN RANK %.1f HIT@10 %.3f FILTERED HIT@10 %.3f" %
                            (test_type, np.mean(accu_mean_rank_t), 
                            np.mean(accu_filtered_mean_rank_t),
                            tail_result,tail_filtered_result))
                
                        print(
                            "[%s] INITIALIZATION [RELATION PREDICTION] MEAN RANK: %.1f FILTERED MEAN RANK %.1f HIT@1 %.3f FILTERED HIT@1 %.3f" %
                            (test_type, np.mean(accu_mean_rank_r), 
                            np.mean(accu_filtered_mean_rank_r),
                            np.mean(np.asarray(accu_mean_rank_r, dtype=np.int32) < 1),
                            np.mean(np.asarray(accu_filtered_mean_rank_r, dtype=np.int32) < 1)))

if __name__ == '__main__':
    tf.app.run()
