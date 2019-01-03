import tensorflow as tf
from dkrl_model import DKRL
from transe_model import TransE
from SSP import SSP
from SSP_JOINT_TXT import SSP_JOINT_TXT
import pickle as pkl 

import numpy as np 
import time 

import os


os.environ['CUDA_VISIBLE_DEVICES']='0'

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
tf.flags.DEFINE_integer('num_epochs', 100, 'Number of epochs')
tf.flags.DEFINE_float('decay_rate', 1, 'Learning rate decay rate. Range: (0, 1]')  # Learning rate decay
tf.flags.DEFINE_integer('decay_steps', 100000, 'Learning rate decay steps')  # Learning rate decay
tf.flags.DEFINE_integer('evaluate_every_steps', 2, 'Evaluate the model on validation set after this many steps')
tf.flags.DEFINE_integer('save_every_steps', 10, 'Save the model after this many steps')
tf.flags.DEFINE_integer('num_checkpoint', 10, 'Number of models to store')

tf.flags.DEFINE_string('model', "DKRL", 'flag of which model')


flags = tf.app.flags.FLAGS


if flags.model=="transE" or flags.model=="ssp":

	from DataLoader import DataLoader
	data_loader = DataLoader(train_path = flags.train_file_path,valid_path=flags.valid_file_path,test_path=flags.test_file_path,
		batch_size=flags.batch_size)
elif flags.model=="dkrl" or flags.model=="ssp_joint_txt":
	from dkrl_data_loader import DataLoader
	data_loader = DataLoader(train_path = flags.train_file_path,valid_path=flags.valid_file_path,test_path=flags.test_file_path,
		description_path = flags.description_path, batch_size=flags.batch_size)
class Test(object):

	def __init__(self, test_data_size):

		self.l_raw_mean_rank = float(0.0)
		self.l_filter_mean_rank = float(0.0)
		self.l_raw_hit_10 = float(0.0)
		self.l_filter_hit_10 = 0.0

		self.r_raw_mean_rank = 0.0
		self.r_filter_mean_rank = 0.0
		self.r_raw_hit_10 = 0.0
		self.r_filter_hit_10 = 0.0

		self.test_data_size = test_data_size
		self.all_data_dict = dict()

		for item in data_loader.train_data_list:

			self.all_data_dict[item] = len(self.all_data_dict)
		for item in data_loader.valid_data_list:

			self.all_data_dict[item] = len(self.all_data_dict)

		for item in data_loader.test_data_list:

			self.all_data_dict[item] = len(self.all_data_dict)

		self.index = 0

	def test_head(self, predict_h, temp_h, triple):


		r_rank = 0.0
		f_rank = 0.0
		cur_h = triple[0]
		for i in range(len(data_loader.ent_dict)):

			if predict_h[i]>=predict_h[cur_h]:
				continue

			r_rank += 1
			if tuple(temp_h[i]) not in self.all_data_dict:

				f_rank +=1
		self.l_raw_mean_rank += r_rank
		self.l_filter_mean_rank += f_rank

		if r_rank < 10:
			self.l_raw_hit_10 +=1
		if f_rank < 10:
			self.l_filter_hit_10 +=1

	def test_tail(self, predict_t, temp_t, triple):

		r_rank = 0.0
		f_rank = 0.0
		cur_t = triple[1]
		for i in range(len(data_loader.ent_dict)):

			if predict_t[i]>=predict_t[cur_t]:
				continue

			r_rank += 1
			if tuple(temp_t[i]) not in self.all_data_dict:

				f_rank +=1
		self.r_raw_mean_rank += r_rank
		self.r_filter_mean_rank += f_rank

		if r_rank < 10:
			self.r_raw_hit_10 +=1
		if f_rank < 10:
			self.r_filter_hit_10 +=1

	def Print(self):

		print("l_raw_mean_rank {:.3f} l_filter_mean_rank {:.3f} l_raw_hit_10 {:.5f} l_filter_hit_10 {:.5f}".format(self.l_raw_mean_rank/self.index, self.l_filter_mean_rank/self.index,
			self.l_raw_hit_10/self.index,self.l_filter_hit_10/self.index))

		print("r_raw_mean_rank {:.3f} r_filter_mean_rank {:.3f} r_raw_hit_10 {:.5f} r_filter_hit_10 {:.5f}".format(self.r_raw_mean_rank/self.index, self.r_filter_mean_rank/self.index,
			self.r_raw_hit_10/self.index,self.r_filter_hit_10/self.index))


def test_model():

	
	
	flags.entTotal = len(data_loader.ent_dict)
	flags.relTotal = len(data_loader.rel_dict)

	with tf.Graph().as_default():
		with tf.Session() as sess:

			if flags.model=="dkrl":
				model = DKRL(flags, data_loader.lengths, data_loader.vocab2id,is_training=False,
					desciption_data = data_loader.get_all_description(),lengths=data_loader.get_all_content_len())
			elif flags.model=="transE":
				model  = TransE(flags)
			elif flags.model=="ssp_joint_txt":
				model = SSP_JOINT_TXT(flags, data_loader.lengths, data_loader.vocab2id,is_training=False,
					desciption_data = data_loader.get_all_description(),lengths=data_loader.get_all_content_len())
			elif flags.model=="ssp":
				model = SSP(flags, is_training=False)

			test = Test(test_data_size=len(data_loader.test_data_list))

			saver = tf.train.Saver(max_to_keep=flags.num_checkpoint)
			# saver.restore(sess,"./res-ssp/500-model.tf")
			# saver.restore(sess,"./res-ssp-joint-txt/750-model.tf")
			saver.restore(sess,"./res-ssp/500-model.tf")

			# if flags.model=="dkrl":
			# 	sess.run([model.get_ent_cnn_embedding()])

			# elif flags.model=="ssp_joint_txt":
			# 	sess.run([model.get_ent_cnn_embedding(),model.init_semantic_embedding()])

			# elif flags.model=="ssp":
			# 	sess.run([model.init_semantic_embedding()])

			def test_step(h, r, t, test_words, content_len):
				feedDict={

						model.test_h:h,
						model.test_r:r,
						model.test_t:t,
						model.test_h_words:test_words[0],
						model.test_t_words:test_words[1],
						model.test_h_content_len: content_len[0],
						model.test_t_content_len: content_len[1]
					}
				res = sess.run([model.predict],feed_dict=feedDict)
				return res

			def test_step_transe(h, r, t):
				feedDict={

						model.test_h:h,
						model.test_r:r,
						model.test_t:t
					}
				res = sess.run([model.predict],feed_dict=feedDict)
				return res

			for data in data_loader.get_predict_instance():

				if test.index==len(data_loader.test_data_list):
					break

				if flags.model=="transE" or flags.model=="ssp":
					temp_h, temp_t, item = data
					predict_h = test_step_transe(temp_h[:,0], temp_h[:,2], temp_h[:,1])
					predict_t = test_step_transe(temp_t[:,0], temp_t[:,2], temp_t[:,1])
					test.test_head(predict_h[0], temp_h, item)
					test.test_tail(predict_t[0], temp_t, item)
					test.index += 1
					test.Print()
				elif flags.model=="dkrl" or flags.model=="ssp_joint_txt":
					temp_h , temp_t , item, head_words,tail_words, head_content_len,tail_content_len = data
					predict_h = test_step(temp_h[:,0], temp_h[:,2], temp_h[:,1], head_words, head_content_len)
					predict_t = test_step(temp_t[:,0], temp_t[:,2], temp_t[:,1], tail_words, tail_content_len)
					test.test_head(predict_h[0], temp_h, item)
					test.test_tail(predict_t[0], temp_t, item)
					test.index += 1
					test.Print()
			
if __name__=="__main__":
	test_model()
