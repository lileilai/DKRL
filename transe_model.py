import tensorflow as tf
import numpy as np 
import pickle as pkl 
import tensorflow.contrib.layers as layers 


class TransE(object):
	def __init__(self,config):

		self.embedding_size = config.embedding_size
		self.batch_size = config.batch_size

		self.vocab_size = config.vocab_size
		self.pretrain_word_embedding_path = config.pretrain_word_embedding_path
		self.batch_seq_size = self.batch_size*(config.neg_ent + config.neg_rel + 1)

		self.entTotal = config.entTotal
		self.relTotal = config.relTotal
		self.margin = config.margin
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
			self.word_embedding = tf.get_variable(name='word_embeddings',
											   dtype=tf.float32,
											   shape=[self.vocab_size, self.embedding_size],
											   initializer=tf.contrib.layers.xavier_initializer(uniform = False),
											   trainable=True)

		with tf.name_scope("loss") as loss:

			p_h = tf.nn.embedding_lookup(self.ent_embeddings,self.pos_h)
			p_r = tf.nn.embedding_lookup(self.rel_embeddings,self.pos_r)
			p_t = tf.nn.embedding_lookup(self.ent_embeddings,self.pos_t)

			n_h = tf.nn.embedding_lookup(self.ent_embeddings,self.neg_h)
			n_r = tf.nn.embedding_lookup(self.rel_embeddings,self.neg_r)
			n_t = tf.nn.embedding_lookup(self.ent_embeddings,self.neg_t)

			p_score = self.calc_loss(p_h,p_r,p_t)
			n_score = self.calc_loss(n_h,n_r,n_t)

			_p_score = tf.reduce_sum(p_score, 1, keep_dims=True)
			_n_score = tf.reduce_sum(n_score,1,keep_dims=True)

			self.loss = tf.reduce_sum(tf.maximum(_p_score - _n_score + self.margin, 0))

		with tf.name_scope("predict") as predict:


			self.get_predict_def()
			self.predict()

	def get_predict_def(self):

		self.test_h= tf.placeholder(name="test_h",shape=[None,],dtype=tf.int32)
		self.test_t = tf.placeholder(name="test_t",shape=[None,],dtype=tf.int32)
		self.test_r = tf.placeholder(name="test_r",shape=[None,],dtype=tf.int32)
		self.test_ent_words = tf.placeholder(name="test_ent_words",shape=[None,None],dtype=tf.int32)

	def predict(self):

		##transE
		p_h = tf.nn.embedding_lookup(self.ent_embeddings,self.test_h)
		p_r = tf.nn.embedding_lookup(self.rel_embeddings,self.test_r)
		p_t = tf.nn.embedding_lookup(self.ent_embeddings,self.test_t)
		self.predict = tf.reduce_sum(self.calc_loss(p_h,p_r,p_t), 1 , keep_dims=True)

	def calc_loss(self, h, r, t):

		return abs(h + r - t)

	def init_word_embedding(self):
		pass
