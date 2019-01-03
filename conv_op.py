# def conv_function(self, ent_words_vec, content_len):
	# 		filter_shape_1 = [2, self.embedding_size, 1, self.feature_map_size]
	# 		weights_1 = tf.get_variable("weights_1", filter_shape_1, initializer=tf.truncated_normal_initializer(stddev=0.1))
	# 		biases_1 = tf.get_variable("biases_1", [self.feature_map_size], initializer=tf.constant_initializer(0.0))

	# 		ent_words_vec = tf.expand_dims(ent_words_vec,-1)
	# 		# conv1_res = tf.layers.conv1d(ent_words_vec,
	# 		# 							 filters=self.feature_map_size,
	# 		# 							 kernel_size=2,
	# 		# 							 padding='valid',
	# 		# 							 activation=tf.nn.relu,
	# 		# 							 kernel_initializer=tf.contrib.layers.xavier_initializer(),
	# 		# 							 name='conv1')
	# 		conv1_res = tf.nn.conv2d(ent_words_vec,
 #                                    weights_1,
 #                                    strides=[1, 1, self.embedding_size, 1],
 #                                    padding='VALID',
 #                                    name='conv_1')
	# 		conv1_res = tf.nn.bias_add(conv1_res, biases_1)

	# 		axes=list(range(len(conv1_res.get_shape())-1))
	# 		mean_1 ,variance_1 =tf.nn.moments(conv1_res,axes)

	# 		bn_norm_1 = tf.nn.batch_normalization(conv1_res,mean_1,variance_1,offset=self.__beta1,scale=self.__gamma1,variance_epsilon=0.001)
	# 		conv1_res = tf.nn.relu(bn_norm_1, name='relu_1')

	# 		# conv1_bn_res = tf.contrib.layers.batch_norm(conv1_res,
	# 		# 											center=True,
	# 		# 											scale=True,
	# 		# 											is_training=self.is_training,
	# 		# 											trainable=True,
	# 		# 											decay=0.9)
	# 		conv1_maxpool_res = tf.nn.max_pool(conv1_res,ksize=[1,4,1,1],strides=[1,4,1,1],padding='SAME',name="pool_1")
	# 		# conv1_maxpool_res = tf.layers.max_pooling1d(conv1_res,pool_size=4,strides=1,
	# 		# 											name='conv1_maxpool')
	# 		# conv2_res = tf.layers.conv1d(conv1_maxpool_res,
	# 		# 							 filters=self.feature_map_size,
	# 		# 							 kernel_size=1,
	# 		# 							 padding='valid',
	# 		# 							 activation=tf.nn.relu,
	# 		# 							 kernel_initializer=tf.contrib.layers.xavier_initializer(),
	# 		# 							 name='conv2')
	# 		# print(conv1_maxpool_res.shape)
	# 		pool1_res = tf.reduce_sum(conv1_maxpool_res,2)
	# 		pool1_res = tf.expand_dims(pool1_res,-1)
	# 		## second layer
	# 		filter_shape_2 = [1, self.embedding_size, 1, self.feature_map_size]
	# 		weights_2 = tf.get_variable("weights_2", filter_shape_2, initializer=tf.truncated_normal_initializer(stddev=0.1))
	# 		biases_2 = tf.get_variable("biases_2", [self.feature_map_size], initializer=tf.constant_initializer(0.0))
            
	# 		conv2_res = tf.nn.conv2d(pool1_res,
 #                                    weights_2,
 #                                    strides=[1, 1, self.embedding_size, 1],
 #                                    padding='VALID',
 #                                    name='conv_2')
	# 		conv2_res = tf.nn.bias_add(conv2_res, biases_2)
	# 		axes=list(range(len(conv2_res.get_shape())-1))
	# 		mean_2 ,variance_2 =tf.nn.moments(conv2_res,axes)

	# 		bn_norm_2 = tf.nn.batch_normalization(conv2_res,mean_2, variance_1,offset=self.__beta2,scale=self.__gamma2,variance_epsilon=0.001)
	# 		conv2_res = tf.nn.relu(bn_norm_2, name='relu_2')
	# 		pool2_res = tf.nn.avg_pool(conv2_res,ksize=[1,self.embedding_size,1,1],strides=[1,self.embedding_size,1,1],name="pool_2",padding="SAME")
			
	# 		conv_outputs = tf.reshape(pool2_res,[-1,self.embedding_size])
	# 		# conv2_res = tf.nn.relu(tf.nn.bias_add(conv2_res, biases_2), name='relu_2')

	# 		# conv2_bn_res = tf.contrib.layers.batch_norm(conv2_res,
	# 		# 											center=True,
	# 		# 											scale=True,
	# 		# 											is_training=self.is_training,
	# 		# 											trainable=True,
	# 		# 											decay=0.9)
	# 		# content_len = tf.reshape(content_len, [-1,1])
	# 		# conv_outputs = tf.truediv(tf.reduce_sum(conv2_res, 1),tf.cast(content_len,tf.float32) , name="outputs")
	# 		# conv_outputs = tf.truediv(tf.reduce_sum(conv2_bn_res, 1),tf.cast(content_len,tf.float32) , name="outputs")
	# 		return conv_outputs