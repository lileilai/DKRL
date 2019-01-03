import tensorflow as tf
import numpy as np 
import pickle as pkl 
import tensorflow.contrib.layers as layers 

class SSP(object):
	def __init__(self,config, is_training=True):

		self.embedding_size = config.embedding_size
		self.batch_size = config.batch_size

		self.batch_seq_size = self.batch_size*(config.neg_ent + config.neg_rel + 1)
		self.entTotal = config.entTotal
		self.relTotal = config.relTotal
		self.margin = config.margin
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
			self.semantic_embeddings = tf.get_variable(name = "semantic_embeddings", 
												shape = [self.entTotal, self.embedding_size], 
												initializer = tf.contrib.layers.xavier_initializer(uniform = False),
												trainable=False)
				
		with tf.name_scope("loss") as loss:

			##positive instance
			p_h = tf.nn.embedding_lookup(self.ent_embeddings,self.pos_h)
			p_r = tf.nn.embedding_lookup(self.rel_embeddings,self.pos_r)
			p_t = tf.nn.embedding_lookup(self.ent_embeddings,self.pos_t)

			p_semantic_h = tf.nn.embedding_lookup(self.semantic_embeddings,self.pos_h)
			p_semantic_t = tf.nn.embedding_lookup(self.semantic_embeddings,self.pos_t)

			##neagtive instance
			n_h = tf.nn.embedding_lookup(self.ent_embeddings,self.neg_h)
			n_r = tf.nn.embedding_lookup(self.rel_embeddings,self.neg_r)
			n_t = tf.nn.embedding_lookup(self.ent_embeddings,self.neg_t)

			n_semantic_h = tf.nn.embedding_lookup(self.semantic_embeddings,self.neg_h)
			n_semantic_t = tf.nn.embedding_lookup(self.semantic_embeddings,self.neg_t)
			p_score = self.calc_loss_ssp(p_h, p_r, p_t, p_semantic_h, p_semantic_t)
			n_score = self.calc_loss_ssp(n_h, n_r, n_t, n_semantic_h, n_semantic_t)

			_p_score = tf.reduce_sum(p_score, 1, keep_dims=True)
			_n_score = tf.reduce_sum(n_score,1, keep_dims=True)

			self.loss = tf.reduce_sum(tf.maximum(_p_score - _n_score + self.margin, 0))

		with tf.name_scope("predict") as predict:


			self.get_predict_def()
			self.predict()

	def semantic_composition(self,embedding_h,embedding_t,axis=1):

		return (embedding_h + embedding_t) / tf.maximum(1e-5,tf.reduce_sum(abs(embedding_h + embedding_t),axis=axis,keep_dims=True))

	def get_predict_def(self):

		self.test_h=  tf.placeholder(name="test_h",shape=[None,],dtype=tf.int32)
		self.test_t = tf.placeholder(name="test_t",shape=[None,],dtype=tf.int32)
		self.test_r = tf.placeholder(name="test_r",shape=[None,],dtype=tf.int32)

	def predict(self):

		## dkrl
		p_h = tf.nn.embedding_lookup(self.ent_embeddings,self.test_h)
		p_r = tf.nn.embedding_lookup(self.rel_embeddings,self.test_r)
		p_t = tf.nn.embedding_lookup(self.ent_embeddings,self.test_t)

		semantic_h = tf.nn.embedding_lookup(self.semantic_embeddings,self.test_h)
		semantic_t = tf.nn.embedding_lookup(self.semantic_embeddings,self.test_t)

		self.predict = tf.reduce_sum(self.calc_loss_ssp(p_h, p_r, p_t, semantic_h,semantic_t), 1 , keep_dims=True)

		# self.predict = tf.reduce_sum(self.calc_loss_tre(p_h, p_r, p_t), 1 , keep_dims=True)

	def calc_loss_ssp(self, h, r, t, semantic_h=None, semantic_t=None):

		error =  self.calc_loss_tre(h, r, t)
		semantic_compos = self.semantic_composition(semantic_h,semantic_t,axis=1)
		return 0.2*abs(error - tf.reduce_sum(error*semantic_compos, axis=1, keep_dims=True)*\
				semantic_compos) + abs(error)

	def calc_loss_tre(self, h, r, t):

		return abs(h + r - t)

	def init_semantic_embedding(self):
		semantic_file = "./data/neighbor/lsi_entity_neighbor_delete.txt"
		semantic_embed = []
		with open(semantic_file, 'r') as f:
			for line in f.readlines()[0:14896]:
				semantic_embed.append(list(map(lambda x: float(x), line.split())))

		semantic_embed = np.array(semantic_embed)
		print("-----success load pre-training neighbor semantic")
		init_semantic_embedding_op = tf.assign(self.semantic_embeddings,semantic_embed)

		return init_semantic_embedding_op

		
		

