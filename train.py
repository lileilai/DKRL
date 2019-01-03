import tensorflow as tf
from dkrl_model import DKRL
from transe_model import TransE
from SSP_JOINT_TXT import SSP_JOINT_TXT 
from SSP import SSP 
import pickle as pkl


import numpy as np 
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

# Data parameters
tf.flags.DEFINE_string('train_file_path', "./data/fb15k2/train2id.txt", 'Data file path')
tf.flags.DEFINE_string('valid_file_path', "./data/fb15k2/valid2id.txt", 'Data file path')
tf.flags.DEFINE_string('test_file_path', "./data/fb15k2/test2id.txt", 'Data file path')
tf.flags.DEFINE_string('description_path', "./data/fb15k2/train_entity_words.txt", 'Data file path')

tf.flags.DEFINE_integer('neg_ent', 1, 'link prediction entity')
tf.flags.DEFINE_integer('neg_rel', 0, 'link prediction relation')


tf.flags.DEFINE_integer('entTotal', 0, 'number of entity')
tf.flags.DEFINE_integer('relTotal', 0, 'number of relation')
tf.flags.DEFINE_integer('margin', 1, 'number of relation')

# Model hyperparameters
tf.flags.DEFINE_integer('vocab_size', 0, 'Vocabulary size')
tf.flags.DEFINE_string('pretrain_word_embedding_path', './data/glove.6B.100d.txt', 'Learning rate')  # All
tf.flags.DEFINE_integer('embedding_size', 100, 'Word embedding size. For CNN, C-LSTM.')
tf.flags.DEFINE_integer('feature_map_size', 100, 'CNN filter sizes. For CNN, C-LSTM.')
tf.flags.DEFINE_integer('num_filters', 128, 'Number of filters per filter size. For CNN, C-LSTM.')
tf.flags.DEFINE_integer('hidden_size', 128, 'Number of hidden units in the LSTM cell. For LSTM, Bi-LSTM')
tf.flags.DEFINE_integer('num_layers', 1, 'Number of the LSTM cells. For LSTM, Bi-LSTM, C-LSTM')
tf.flags.DEFINE_float('keep_prob', 0.5, 'Dropout keep probability')  # All
tf.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')  # All

# Training parameters
tf.flags.DEFINE_integer('batch_size', 1000, 'Batch size')

tf.flags.DEFINE_integer('num_epochs', 500, 'Number of epochs')
tf.flags.DEFINE_float('decay_rate', 1, 'Learning rate decay rate. Range: (0, 1]')  # Learning rate decay
tf.flags.DEFINE_integer('decay_steps', 100000, 'Learning rate decay steps')  # Learning rate decay
tf.flags.DEFINE_integer('evaluate_every_steps', 2, 'Evaluate the model on validation set after this many steps')
tf.flags.DEFINE_integer('save_every_steps', 100, 'Save the model after this many steps')
tf.flags.DEFINE_integer('num_checkpoint', 10, 'Number of models to store')

tf.flags.DEFINE_string('model', "dkrl", 'flag of which model')

flags = tf.app.flags.FLAGS
print(flags.num_epochs)

def train():

	# data_loader = DataLoader(train_path = flags.train_file_path,valid_path=flags.valid_file_path,test_path=flags.test_file_path,
	# 	description_path = flags.description_path, batch_size=flags.batch_size)
	if flags.model=="transE":

		from DataLoader import DataLoader
		data_loader = DataLoader(train_path = flags.train_file_path,valid_path=flags.valid_file_path,test_path=flags.test_file_path,
			batch_size=flags.batch_size)
	elif flags.model=="dkrl" or flags.model=="ssp_joint_txt":
		from dkrl_data_loader import DataLoader
		data_loader = DataLoader(train_path = flags.train_file_path,valid_path=flags.valid_file_path,test_path=flags.test_file_path,
			description_path = flags.description_path, batch_size=flags.batch_size)
	elif flags.model=="ssp":
		from DataLoader import DataLoader
		data_loader = DataLoader(train_path = flags.train_file_path,valid_path=flags.valid_file_path,test_path=flags.test_file_path,
			batch_size=flags.batch_size)

	flags.entTotal = len(data_loader.ent_dict)
	flags.relTotal = len(data_loader.rel_dict)
	with tf.Graph().as_default():
		with tf.Session() as sess:

			if flags.model=="transE":
				model  = TransE(flags)
			elif flags.model=="dkrl":
				model = DKRL(flags, data_loader.lengths, data_loader.vocab2id,is_training=True)
			elif flags.model=="ssp":
				model = SSP(flags, is_training=True)
			elif flags.model=="ssp_joint_txt":
				model = SSP_JOINT_TXT(flags, data_loader.lengths, data_loader.vocab2id,is_training=True)

			saver = tf.train.Saver(max_to_keep=flags.num_checkpoint)
			global_step = tf.Variable(0,trainable=False,name="global_step")
			# optimizer = tf.train.AdamOptimizer(flags.learning_rate)
			optimizer = tf.train.GradientDescentOptimizer(flags.learning_rate)
			grads_and_vars = optimizer.compute_gradients(model.loss)
			train_op = optimizer.apply_gradients(grads_and_vars)
			sess.run(tf.global_variables_initializer())
			# saver.restore(sess,"./res-ssp-joint-txt/750-model.tf")
			if flags.model=="ssp_joint_txt":
				sess.run([model.init_semantic_embedding()])
			# saver.restore(sess,"./res-ssp/300-model.tf")
			if flags.model=="dkrl":
				sess.run([model.init_word_embedding()])
			elif flags.model=="ssp_joint_txt":
				sess.run([model.init_word_embedding(),model.init_semantic_embedding()])
			elif flags.model=="ssp":
				sess.run([model.init_semantic_embedding()])
			def train_step(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t, 
				p_h_desc, p_t_desc, n_h_desc, n_t_desc, content_len, is_training=True):
				feedDict={

						model.pos_h:pos_h,
						model.pos_r:pos_r,
						model.pos_t:pos_t,
						model.neg_h:neg_h,
						model.neg_r:neg_r,
						model.neg_t:neg_t,

						model.pos_h_words: p_h_desc,
						model.pos_t_words: p_t_desc,

						model.neg_h_words: n_h_desc,
						model.neg_t_words: n_t_desc,

						model.pos_h_content_len:content_len[0],
						model.pos_t_content_len:content_len[1],
						model.neg_h_content_len:content_len[2],
						model.neg_t_content_len:content_len[3]
					}
				_, loss = sess.run([ train_op, model.loss],feed_dict=feedDict)
				return loss
			def train_step_transe(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
				feedDict={

						model.pos_h:pos_h,
						model.pos_r:pos_r,
						model.pos_t:pos_t,
						model.neg_h:neg_h,
						model.neg_r:neg_r,
						model.neg_t:neg_t
					}
				_, loss = sess.run([ train_op, model.loss],feed_dict=feedDict)
				return loss

			for times in range(1,flags.num_epochs+1):
				res=0.0
				for batch_data in data_loader.next_batch(flag=1):

					if flags.model=="transE" or flags.model=="ssp":
						pos, neg = batch_data
						res += train_step_transe(pos[:,0],pos[:,2],pos[:,1],neg[:,0], neg[:,2],neg[:,1])
					elif flags.model=="dkrl" or flags.model=="ssp_joint_txt":
						pos , p_h_desc,p_t_desc, neg, n_h_desc, n_t_desc ,content_len = batch_data
						res += train_step(pos[:,0],pos[:,2],pos[:,1],neg[:,0], neg[:,2],neg[:,1],
							p_h_desc, p_t_desc, n_h_desc, n_t_desc,content_len)
					
				print("epoch: ",times)
				print("loss: ",res)
				if times%flags.save_every_steps==0:
					if flags.model=="transE":
						saver.save(sess,"./res-transe/{}-model.tf".format(times))
					elif flags.model=="dkrl":
						saver.save(sess,"./res-dkrl/{}-model.tf".format(times))
					elif flags.model=="ssp_joint_txt":
						saver.save(sess,"./res-ssp-joint-txt/{}-model.tf".format(times))
					elif flags.model=="ssp":
						saver.save(sess,"./res-ssp/{}-model.tf".format(times))

if __name__=="__main__":
	train()






