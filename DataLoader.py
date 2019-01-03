import tensorflow as tf 
import numpy as np 
import pickle as pkl 
import pandas as pd
import os

class DataLoader(object):

	def __init__(self,train_path, valid_path, test_path, neg_ent=1, neg_rel=0, batch_size=1000):

		self.train_path = train_path
		self.valid_path = valid_path
		self.test_path = test_path

		self.neg_ent = neg_ent
		self.neg_rel = neg_rel

		self.ent_dict=dict()
		self.rel_dict=dict()

		self.batch_size=batch_size
		print("-----batch_size: ",batch_size)

		# is_exist = os.path.exists("./preprocess_pkl/1.pkl") and os.path.exists("./preprocess_pkl/2.pkl") and os.path.exists("./preprocess_pkl/3.pkl")
		##load train  valid test dataset
		self.train_data_list = self.load_data(self.train_path,flag=1)
		self.valid_data_list = self.load_data(self.valid_path,flag=2)
		self.test_data_list =  self.load_data(self.test_path,flag=3)

		self.train_batch_num = (len(self.train_data_list)-1) // self.batch_size +1
		self.valid_batch_num = (len(self.valid_data_list)-1) // self.batch_size +1
		self.test_batch_num =  (len(self.test_data_list)-1) // self.batch_size +1


		print("-----train dataset size: ",len(self.train_data_list))
		print("-----valid dataset size: ",len(self.valid_data_list))
		print("-----test dataset size: ",len(self.test_data_list))

	def load_data(self,file_path,flag):

		df = pd.read_table(file_path,"r",encoding="utf-8", header=None,delimiter=" ")
		data_list = list(zip(df[0],df[1],df[2]))
		if flag==1:
			##计算 rel_heads, rel_tails
			self.compute_rel_heads_tails(data_list)

		return data_list
	def compute_rel_heads_tails(self, data_list):

		rel_heads = dict()
		rel_tails = dict()
		for data in data_list:
			h, t, r = data
			if h not in self.ent_dict:
				self.ent_dict[h] = len(self.ent_dict)
			if t not in self.ent_dict:
				self.ent_dict[t] = len(self.ent_dict)
			if r not in self.rel_dict:
				self.rel_dict[r] = len(self.rel_dict)

			if r not in rel_heads:
				temp_dict = dict()
				temp_list = list()
				temp_list.append(t)
				temp_dict[h] = temp_list
				rel_heads[r] = temp_dict
			else:

				if h not in rel_heads[r].keys():
					li = list()
					li.append(t)
					rel_heads[r][h] = li
				else:
					rel_heads[r][h].append(t)

			if r not in rel_tails:
				temp_dict = dict()
				temp_list = list()
				temp_list.append(h)
				temp_dict[t] = temp_list
				rel_tails[r] = temp_dict
			else:
				
				if t not in rel_tails[r].keys():
					li = list()
					li.append(h)
					rel_tails[r][t] = li
				else:
					rel_tails[r][t].append(h)
		# print("-----",len(rel_heads))
		# print("-----",len(rel_tails))

		self.rel_hpt=list()
		self.rel_tph=list()
		for rel in self.rel_dict.keys():
			Sum=0.0
			Total=0.0
			for key in rel_heads[rel].keys():
				Sum += 1
				Total += len(rel_heads[rel][key])
			self.rel_hpt.append(Total/Sum)

		for rel in self.rel_dict.keys():
			Sum=0.0
			Total=0.0
			for key in rel_tails[rel].keys():
				Sum += 1
				Total += len(rel_tails[rel][key])
			self.rel_tph.append(Total/Sum)
	

	def sample_false_triple(self,pos):

		neg  = list()
		for triple in pos:

			h,t,r = triple
			temp = list(triple)
			prob = self.rel_hpt[r] / (self.rel_hpt[r] + self.rel_tph[r])

			if np.random.randint(0,1000)<1000*prob:
				temp[0] = np.random.randint(0,len(self.ent_dict))%len(self.ent_dict)
			else:
				temp[1] = np.random.randint(0,len(self.ent_dict))%len(self.ent_dict)
			neg.append(temp)

		return neg
	def replace_head_and_tail(self, triple):
		## replace head
		triple = list(triple)
		temp_h = np.array([triple] * len(self.ent_dict))
		temp_t = np.array([triple] * len(self.ent_dict))
		
		## replace head
		index=0
		for i, h in enumerate(self.ent_dict.keys()):

			temp_h[index][0] = h
			index +=1
		index=0
		for i, t in enumerate(self.ent_dict.keys()):
			temp_t[index][1] = t
			index+=1
		return temp_h,temp_t

	def next_batch(self,flag):
		if flag==1:
			rag = np.random.permutation(np.arange(self.train_batch_num))
			for index in rag:
				start_index = index*self.batch_size
				end_index = min(start_index+self.batch_size, len(self.train_data_list))
				pos = self.train_data_list[start_index:end_index]
				neg = self.sample_false_triple(pos)
				pos = np.array(pos)
				neg = np.array(neg)
				yield pos,neg
		elif flag==2:
			rag = np.random.permutation(np.arange(self.valid_batch_num))
			for index in rag:
				start_index = index*self.batch_size
				end_index = min(start_index+self.batch_size, len(self.train_data_list))
				pos = self.valid_data_list[start_index:end_index]
				neg = self.sample_false_triple(pos)
				yield pos,neg
	def get_predict_instance(self):
		index=0
		for item in self.test_data_list:
			print(index,item)
			index+=1
			temp_h, temp_t = self.replace_head_and_tail(item)
			temp_h = np.array(temp_h)
			temp_t = np.array(temp_t)
			yield temp_h, temp_t, item

# if __name__=="__main__":
# 	train_path = "./data/fb15k2/train2id.txt"
# 	valid_path = "./data/fb15k2/valid2id.txt"
# 	test_path = "./data/fb15k2/test2id.txt"
# 	model = DataLoader(train_path, valid_path, test_path)

