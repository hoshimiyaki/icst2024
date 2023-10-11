import os
import random
import pickle
import sys

pattern=str(sys.argv[1])

train_val_split = 0.9
step=40


with open(str("../data/cooked/"+pattern+".pkl"), "rb") as file:
	input_total = pickle.load(file)
with open(str("../data/cooked/y.pkl"), "rb") as file:
	labels = pickle.load(file)

# 10-fold cross-validation
step = 40
versions = list(input_total.keys())
random.seed(888)
random.shuffle(versions)
for group_index in range(10):
	train_pos_x = []
	train_neg_x = []
	train_x=[]
	train_y=[]

	data_path = "../data/cooked/{}/".format(group_index + 1)
	test_versions = versions[group_index * step: (group_index + 1) * step]
	train_versions = [version for version in versions if version not in test_versions]
	for version in train_versions:
		pos_indices = [index for (index, value) in enumerate(labels[version]) if value == 0]
		neg_indices = [index for (index, value) in enumerate(labels[version]) if value == 1]
		assert len(pos_indices) + len(neg_indices) == len(labels[version])
		for pos_index in pos_indices:
			random.seed(888)
			random.shuffle(neg_indices)
			# 一个bug配10个非bug，解决类别不匹配问题
			for neg_index in neg_indices[:10]:
				train_pos_x.append(input_total[version][pos_index])
				train_neg_x.append(input_total[version][neg_index])
				train_x.append(input_total[version][pos_index])
				train_y.append(0)
				train_x.append(input_total[version][neg_index])
				train_y.append(1)

	val_x=train_x[round(len(train_x)*train_val_split):]
	val_y=train_y[round(len(train_y)*train_val_split):]
	train_x=train_x[:round(len(train_x)*train_val_split)]
	train_y=train_y[:round(len(train_y)*train_val_split)]

	test_x = {}
	test_y = {}
	for version in test_versions:
		test_x[version] = input_total[version]
		test_y[version] = labels[version]
	
	seed=888
	random.seed(seed)
	random.shuffle(train_pos_x)
	random.seed(seed)
	random.shuffle(train_neg_x)
	

	if not os.path.exists(data_path):
		os.makedirs(data_path)

	train_dir = os.path.join(data_path, "train/")
	if not os.path.exists(train_dir):
		os.makedirs(train_dir)

	test_dir = os.path.join(data_path, "test/")
	if not os.path.exists(test_dir):
		os.makedirs(test_dir)

	val_dir = os.path.join(data_path, "val/")
	if not os.path.exists(val_dir):
		os.makedirs(val_dir)

	with open(os.path.join(train_dir, "x_pos_"+pattern+".pkl"), "wb") as file:
		pickle.dump(train_pos_x, file)
	with open(os.path.join(train_dir, "x_neg_"+pattern+".pkl"), "wb") as file:
		pickle.dump(train_neg_x, file)
	with open(os.path.join(train_dir, "x_"+pattern+".pkl"), "wb") as file:
		pickle.dump(train_x, file)
	with open(os.path.join(train_dir, "y.pkl"), "wb") as file:
		pickle.dump(train_y, file)

	with open(os.path.join(val_dir, "x_"+pattern+".pkl"), "wb") as file:
		pickle.dump(val_x, file)
	with open(os.path.join(val_dir, "y.pkl"), "wb") as file:
		pickle.dump(val_y, file)

	with open(os.path.join(test_dir, "x_"+pattern+".pkl"), "wb") as file:
		pickle.dump(test_x, file)
	with open(os.path.join(test_dir, "y.pkl"), "wb") as file:
		pickle.dump(test_y, file)
