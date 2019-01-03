import tensorflow as tf
import numpy as np 
import pickle as pkl 
import tensorflow.contrib.layers as layers 

class SSP_JOINT_TXT(object):
	def __init__(self,config, content_len, vocab2id, is_training=True,desciption_data=None,lengths=None):

		self.embedding_size = config.embedding_size
		self.batch_size = config.batch_size

		self.vocab_size = config.vocab_size
		self.pretrain_word_embedding_path = config.pretrain_word_embedding_path
		self.batch_seq_size = self.batch_size*(config.neg_ent + config.neg_rel + 1)
		self.feature_map_size = config.feature_map_size

		self.entTotal = config.entTotal
		self.relTotal = config.relTotal
		self.margin = config.margin
		self.content_len = content_len

		self.is_training = is_training

		self.vocab2id = vocab2id

		self.first=True
		self.desciption_data = desciption_data
		self.lengths = lengths

		with tf.name_scope("input") as inp:

			## positive
			self.pos_h = tf.placeholder(name="pos_h",shape=[None,],dtype=tf.int32)
			self.pos_r = tf.placeholder(name="pos_r",shape=[None,],dtype=tf.int32)
			self.pos_t = tf.placeholder(name="pos_t",shape=[None,],dtype = tf.int32)
			self.pos_h_words = tf.placeholder(name="pos_h_words",shape=[None, None], dtype = tf.int32)
			self.pos_t_words = tf.placeholder(name="pos_t_words",shape=[None, None], dtype = tf.int32)
			self.pos_h_content_len = tf.placeholder(name="pos_h_content_len", shape=[None, ],dtype=tf.int32)
			self.pos_t_content_len = tf.placeholder(name="pos_t_content_len", shape=[None, ],dtype=tf.int32)
			##negtive

			self.neg_h = tf.placeholder(name="neg_h",shape=[None,],dtype=tf.int32)
			self.neg_r = tf.placeholder(name="neg_r",shape=[None,],dtype=tf.int32)
			self.neg_t = tf.placeholder(name="neg_t",shape=[None,],dtype = tf.int32)
			self.neg_h_words = tf.placeholder(name="neg_h_words",shape=[None, None], dtype = tf.int32)
			self.neg_t_words = tf.placeholder(name="neg_t_words",shape=[None, None], dtype = tf.int32)
			self.neg_h_content_len = tf.placeholder(name="neg_h_content_len", shape=[None, ] ,dtype=tf.int32)
			self.neg_t_content_len = tf.placeholder(name="neg_t_content_len", shape=[None, ],dtype=tf.int32)
		with tf.name_scope("batch_norm") as BN:

			self.__beta1 = tf.Variable(tf.constant(0.0, shape=[self.embedding_size]),name='beta1')
			self.__gamma1 = tf.Variable(tf.constant(1.0, shape=[self.embedding_size]),name='gamma1')
			self.__beta2 = tf.Variable(tf.constant(0.0, shape=[self.embedding_size]),name='beta2')
			self.__gamma2 = tf.Variable(tf.constant(1.0, shape=[self.embedding_size]),name='gamma2')

		with tf.name_scope("embedding") as embedding:

			self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [self.entTotal, self.embedding_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [self.relTotal, self.embedding_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			self.word_embedding = tf.get_variable(name='word_embeddings',
											   dtype=tf.float32,
											   shape=[len(self.vocab2id), self.embedding_size],
											   initializer=tf.contrib.layers.xavier_initializer(uniform = False),
											   trainable=True)
			self.ent_cnn_embeddings = tf.get_variable(name = "ent_cnn_embeddings", 
												shape = [self.entTotal, self.embedding_size], 
												initializer = tf.contrib.layers.xavier_initializer(uniform = False),trainable=False)
			self.semantic_embeddings = tf.get_variable(name = "semantic_embeddings", 
												shape = [self.entTotal, self.embedding_size], 
												initializer = tf.contrib.layers.xavier_initializer(uniform = False),trainable=False)
			#读取glove预先训练好的向量
			# if self.is_training:
			# 	self.init_word_embedding()
				
		with tf.name_scope("loss") as loss:

			##positive instance
			p_h = tf.nn.embedding_lookup(self.ent_embeddings,self.pos_h)
			p_r = tf.nn.embedding_lookup(self.rel_embeddings,self.pos_r)
			p_t = tf.nn.embedding_lookup(self.ent_embeddings,self.pos_t)
			p_h_words_vec = tf.nn.embedding_lookup(self.word_embedding,self.pos_h_words)
			p_h_conv_res = self.conv_layers(p_h_words_vec, self.pos_h_content_len)
			# p_h_conv_res = self.cbow(p_h_words_vec, self.pos_h_content_len)

			p_t_words_vec = tf.nn.embedding_lookup(self.word_embedding,self.pos_t_words)
			p_t_conv_res = self.conv_layers(p_t_words_vec, self.pos_t_content_len)
			# p_t_conv_res = self.cbow(p_t_words_vec, self.pos_t_content_len)

			p_semantic_h = tf.nn.embedding_lookup(self.semantic_embeddings,self.pos_h)
			p_semantic_t = tf.nn.embedding_lookup(self.semantic_embeddings,self.pos_t)



			##neagtive instance
			n_h = tf.nn.embedding_lookup(self.ent_embeddings,self.neg_h)
			n_r = tf.nn.embedding_lookup(self.rel_embeddings,self.neg_r)
			n_t = tf.nn.embedding_lookup(self.ent_embeddings,self.neg_t)
			n_h_words_vec = tf.nn.embedding_lookup(self.word_embedding,self.neg_h_words)
			n_h_conv_res = self.conv_layers(n_h_words_vec, self.neg_h_content_len)
			# n_h_conv_res = self.cbow(n_h_words_vec, self.neg_h_content_len)

			n_t_words_vec = tf.nn.embedding_lookup(self.word_embedding,self.neg_t_words)
			n_t_conv_res = self.conv_layers(n_t_words_vec, self.neg_t_content_len)

			n_semantic_h = tf.nn.embedding_lookup(self.semantic_embeddings,self.neg_h)
			n_semantic_t = tf.nn.embedding_lookup(self.semantic_embeddings,self.neg_t)

			# n_t_conv_res = self.cbow(n_t_words_vec, self.neg_t_content_len)

			p_score = self.calc_loss_dkrl(p_h, p_r, p_t, p_h_conv_res, p_t_conv_res, semantic_h=p_semantic_h, semantic_t=p_semantic_t)
			n_score = self.calc_loss_dkrl(n_h, n_r, n_t, n_h_conv_res, n_t_conv_res, semantic_h=n_semantic_h, semantic_t=n_semantic_t)

			# p_score = self.calc_loss_tre(p_h_conv_res, p_r, p_t_conv_res)
			# n_score = self.calc_loss_tre(n_h_conv_res, n_r, n_t_conv_res)

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
	
			conv1_res = tf.layers.conv1d(ent_words_vec,
										 filters=self.feature_map_size,
										 kernel_size=2,
										 padding='valid',
										 activation=tf.nn.relu,
										 kernel_initializer=tf.contrib.layers.xavier_initializer(),
										 name='conv1')
			# conv1_bn_res = tf.contrib.layers.batch_norm(conv1_res,
			# 											center=True,
			# 											scale=True,
			# 											is_training=self.is_training,
			# 											trainable=True,
			# 											decay=0.9)
			# conv1_maxpool_res = tf.nn.max_pool(conv1_res,ksize=[1,4,1,1],strides=[1,4,1,1],padding='SAME',name="pool_1")
			conv1_maxpool_res = tf.layers.max_pooling1d(conv1_res,pool_size=4,strides=4,
														name='conv1_maxpool',padding="SAME")
			conv2_res = tf.layers.conv1d(conv1_maxpool_res,
										 filters=self.feature_map_size,
										 kernel_size=1,
										 padding='valid',
										 activation=tf.nn.relu,
										 kernel_initializer=tf.contrib.layers.xavier_initializer(),
										 name='conv2')
			# conv2_bn_res = tf.contrib.layers.batch_norm(conv2_res,
			# 											center=True,
			# 											scale=True,
			# 											is_training=self.is_training,
			# 											trainable=True,
			# 											decay=0.9)

			content_len = tf.reshape(content_len, [-1,1])
			conv_outputs = tf.truediv(tf.reduce_sum(conv2_res, 1),tf.cast(content_len,tf.float32) , name="outputs")
			# conv_outputs = tf.truediv(tf.reduce_sum(conv2_bn_res, 1),tf.cast(content_len,tf.float32) , name="outputs")
			return conv_outputs

	def semantic_composition(self,embedding_h,embedding_t,axis=1):

		return (embedding_h + embedding_t) / tf.maximum(1e-5,tf.reduce_sum(abs(embedding_h + embedding_t),axis=axis,keep_dims=True))

	def conv_layers(self, ent_words_vec, content_len):

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
		self.test_h_words = tf.placeholder(name="test_h_words",shape=[None, None],dtype=tf.int32)
		self.test_t_words = tf.placeholder(name="test_t_words",shape=[None, None],dtype=tf.int32)

		self.test_h_content_len = tf.placeholder(name="neg_h_content_len", shape=[None, ] ,dtype=tf.int32)
		self.test_t_content_len = tf.placeholder(name="neg_t_content_len", shape=[None, ],dtype=tf.int32)

	def predict(self):

		## dkrl
		p_h = tf.nn.embedding_lookup(self.ent_embeddings,self.test_h)
		p_r = tf.nn.embedding_lookup(self.rel_embeddings,self.test_r)
		p_t = tf.nn.embedding_lookup(self.ent_embeddings,self.test_t)
		h_conv_res = tf.nn.embedding_lookup(self.ent_cnn_embeddings,self.test_h)
		t_conv_res = tf.nn.embedding_lookup(self.ent_cnn_embeddings,self.test_t)

		semantic_h = tf.nn.embedding_lookup(self.semantic_embeddings,self.test_h)
		semantic_t = tf.nn.embedding_lookup(self.semantic_embeddings,self.test_t)

		self.predict = tf.reduce_sum(self.calc_loss_dkrl(p_h, p_r, p_t, h_conv_res, t_conv_res, semantic_h,semantic_t), 1 , keep_dims=True)

		# self.predict = tf.reduce_sum(self.calc_loss_tre(p_h, p_r, p_t), 1 , keep_dims=True)

	def calc_loss_dkrl(self, h, r, t, d_h, d_t, semantic_h=None, semantic_t=None):

		error =  abs( (0.9*d_h + 0.1*h) + r   - (0.9*d_t + 0.1*t))
		semantic_compos = self.semantic_composition(semantic_h,semantic_t,axis=1)
		return 0.2*abs(error - tf.reduce_sum(error*semantic_compos, axis=1, keep_dims=True)*\
				semantic_compos) + abs(error)

	def calc_loss_tre(self, h, r, t):

		return abs(h + r - t)

	def init_word_embedding(self):

		current_embedding = np.random.uniform(-1, 1, [len(self.vocab2id), self.embedding_size])
		current_embedding[0] = np.array([0.0]*self.embedding_size)
		with open(self.pretrain_word_embedding_path, 'r') as f:
			for line in f.readlines():
				items = line.strip().split(' ')
				if self.vocab2id.__contains__(items[0]):
					current_embedding[self.vocab2id[items[0]]] = items[1:]
		print("-----success load pre-trainning word2vec")
		init_word_embedding_op = tf.assign(self.word_embedding,current_embedding)

		return init_word_embedding_op
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

		
		

