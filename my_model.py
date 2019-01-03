import tensorflow as tf
import numpy as np 
import pickle as pkl 
import tensorflow.contrib.layers as layers 

class my_model(object):
	def __init__(self,config, is_training=True):

		self.embedding_size = config.embedding_size
		self.batch_size = config.batch_size

		self.batch_seq_size = self.batch_size*(config.neg_ent + config.neg_rel + 1)
		self.feature_map_size = config.feature_map_size

		self.entTotal = config.entTotal
		self.relTotal = config.relTotal
		self.margin = config.margin
		self.content_len = content_len

		self.is_training = is_training

		self.first=True
		with tf.name_scope("input") as inp:

			## positive
			self.pos_h = tf.placeholder(name="pos_h",shape=[None,],dtype=tf.int32)
			self.pos_r = tf.placeholder(name="pos_r",shape=[None,],dtype=tf.int32)
			self.pos_t = tf.placeholder(name="pos_t",shape=[None,],dtype = tf.int32)

			##negtive

			self.neg_h = tf.placeholder(name="neg_h",shape=[None,],dtype=tf.int32)
			self.neg_r = tf.placeholder(name="neg_r",shape=[None,],dtype=tf.int32)
			self.neg_t = tf.placeholder(name="neg_t",shape=[None,],dtype = tf.int32)

		with tf.name_scope("embedding") as embedding:

			self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [self.entTotal, self.embedding_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [self.relTotal, self.embedding_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			#读取glove预先训练好的向量
			# if self.is_training:
			# 	self.init_word_embedding()
				
		with tf.name_scope("loss") as loss:

			##positive instance
			p_h = tf.nn.embedding_lookup(self.ent_embeddings,self.pos_h)
			p_r = tf.nn.embedding_lookup(self.rel_embeddings,self.pos_r)
			p_t = tf.nn.embedding_lookup(self.ent_embeddings,self.pos_t)



			##neagtive instance
			n_h = tf.nn.embedding_lookup(self.ent_embeddings,self.neg_h)
			n_r = tf.nn.embedding_lookup(self.rel_embeddings,self.neg_r)
			n_t = tf.nn.embedding_lookup(self.ent_embeddings,self.neg_t)
	
			_p_score = tf.reduce_sum(p_score, 1, keep_dims=True)
			_n_score = tf.reduce_sum(n_score,1, keep_dims=True)

			self.loss = tf.reduce_sum(tf.maximum(_p_score - _n_score + self.margin, 0))

		with tf.name_scope("predict") as predict:


			self.get_predict_def()
			self.predict()
	def get_ent_cnn_embedding(self):

		all_ent_desc_vec = tf.nn.embedding_lookup(self.word_embedding,self.desciption_data)
		self.leng = tf.constant(self.lengths,dtype=tf.int32) 
		pre_compute_conv_res = self.conv_layers(all_ent_desc_vec, self.leng)

		pre_compute_conv_op = tf.assign(self.ent_cnn_embeddings,pre_compute_conv_res)
		print("pre-compute convolution result")
		return pre_compute_conv_op

	def conv_function(self, ent_words_vec, content_len):
	
			
			return conv_outputs

	def lstm_layers(self):

		if self.first:
			with tf.variable_scope(tf.get_variable_scope()) as conv:

				conv_outputs = self.conv_function(ent_words_vec, content_len)

			self.first=False
		else:
			with tf.variable_scope(tf.get_variable_scope(),reuse=True) as conv:

				conv_outputs = self.conv_function(ent_words_vec, content_len)				

		return conv_outputs

	def get_predict_def(self):

		self.test_h= tf.placeholder(name="test_h",shape=[None,],dtype=tf.int32)
		self.test_t = tf.placeholder(name="test_t",shape=[None,],dtype=tf.int32)
		self.test_r = tf.placeholder(name="test_r",shape=[None,],dtype=tf.int32)

	def predict(self):

		## dkrl
		p_h = tf.nn.embedding_lookup(self.ent_embeddings,self.test_h)
		p_r = tf.nn.embedding_lookup(self.rel_embeddings,self.test_r)
		p_t = tf.nn.embedding_lookup(self.ent_embeddings,self.test_t)
		h_conv_res = tf.nn.embedding_lookup(self.ent_cnn_embeddings,self.test_h)
		t_conv_res = tf.nn.embedding_lookup(self.ent_cnn_embeddings,self.test_t)

		self.predict = tf.reduce_sum(self.calc_loss_dkrl(p_h, p_r, p_t, h_conv_res, t_conv_res), 1 , keep_dims=True)

		# self.predict = tf.reduce_sum(self.calc_loss_tre(p_h, p_r, p_t), 1 , keep_dims=True)


		
		

