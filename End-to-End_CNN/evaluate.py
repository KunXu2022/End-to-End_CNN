# encoding:gbk
import argparse
import os
import re
import sys
import csv
import numpy as np
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--filelist', help='Path to training set ground truth (.csv)', required=True)
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--model_path', default='E:\GCN\CNN\evaluate\ckpt\model.ckpt', help='model checkpoint file path')
parser.add_argument('--num_input_channels', type=int, default=17) #输入特征通道数量
parser.add_argument('--num_classes', type=int, default=2) #分类数量
parser.add_argument('--data_size', type=int, default=16) #数据尺寸
parser.add_argument('--output_filelist', default='E:\GCN\CNN\prediction\output.txt', help='TXT filename, filelist, each line is an output for pixel')
parser.add_argument('--log_dir', default='log')
FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu
FILELIST = FLAGS.filelist
BATCH_SIZE = FLAGS.batch_size
NUM_INPUT_CHANNELS = FLAGS.num_input_channels
NUM_CLASSES = FLAGS.num_classes
DATA_SIZE = FLAGS.data_size
MODEL_PATH = FLAGS.model_path
OUTPUT_FILELIST = FLAGS.output_filelist
LOG_DIR = FLAGS.log_dir

if not os.path.exists(LOG_DIR): 
	os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

#获取数据
EVALUATE_FILES = []
for line in open(os.path.join(FILELIST, 'train_all.txt')):
	line = re.sub(r"\n","",line) #去掉字符串中的转义字符正则化好用
	EVALUATE_FILES.append(line)
	
def evaluate():
	
	is_training_eval = False
	
	with tf.device('/gpu:0'):
		
		pointclouds_pl = tf.placeholder(tf.float32,
                               shape=(BATCH_SIZE, DATA_SIZE, DATA_SIZE, DATA_SIZE, NUM_INPUT_CHANNELS))
		is_training_pl = tf.placeholder(dtype=tf.bool)
		
		#定义卷积模型及损失函数
		pred = get_model(pointclouds_pl, is_training_pl)
		pred_softmax = tf.nn.softmax(pred)
		
	#配置tf.Session的运算方式
	config = tf.ConfigProto() #实例化对象
	config.gpu_options.allow_growth = True #动态申请显存
	config.allow_soft_placement = True #自动选择运行设备
	config.gpu_options.per_process_gpu_memory_fraction = 1 #GPU内存占用率设置
	config.log_device_placement = False #不在终端打印出各项操作是在哪个设备上运行
	sess = tf.Session(config=config)
	
	# Restore variables from disk.
	loader = tf.train.import_meta_graph(MODEL_PATH + '.meta')
	loader.restore(sess, MODEL_PATH)
	log_string(LOG_FOUT, "Model restored!")
	
	#获取参数
	graph = tf.get_default_graph()
	Placeholder = graph.get_tensor_by_name("Placeholder:0")
	is_training = graph.get_tensor_by_name("Placeholder_1:0")
	for op in graph.get_operations():
		print(op.name)
	init = tf.global_variables_initializer()
	sess.run(init)
	ops = {'pred': pred,
		   'pred_softmax': pred_softmax}
	
	fout_out_filelist = open(OUTPUT_FILELIST, 'w')
	
	for filename in EVALUATE_FILES:
		
		#tf.reset_default_graph()
		
		#读取数据
		data_temp = []
		csvfile = open(os.path.join(FILELIST, filename), 'r')
		reader = csv.reader(csvfile)
		for item in reader:
			if reader.line_num == 1: #第一行为属性名
				continue
			data_temp.append(item[3:-2]) #前三列为I,J,K; 后两列为标签
		data_temp = np.array(data_temp)
		data_temp = data_temp.flatten() #将多维数组转换成一维数组
		data_temp = data_temp.astype(np.float) #将字符串转化为float
		data_temp = data_temp.tolist() #数组转列表
		
		data_input = change_format(data_temp, BATCH_SIZE, DATA_SIZE, DATA_SIZE, DATA_SIZE, NUM_INPUT_CHANNELS, 'TRUE')

		feed_dict = {Placeholder: data_input,
		             is_training: is_training_eval}
		
		pred_val, pred_softmax = sess.run([ops['pred'], ops['pred_softmax']],
										  feed_dict=feed_dict)
		
		pred_val = np.array(pred_val)
		pred_softmax = np.array(pred_softmax)
		pred_val = pred_val.reshape(-1)
		pred_softmax = pred_softmax.reshape(-1)
		pred_val = pred_val.tolist()
		pred_softmax = pred_softmax.tolist()
		
		fout_out_filelist.write(filename + '\n')
		#log_string(LOG_FOUT, 'filename: %f' % (filename))
		'''
		for i in range(len(pred_val)):
			fout_out_filelist.write(str(pred_val[i]) + '\n')
			fout_out_filelist.write(str(pred_softmax[i]) + '\n')
			
			log_string(LOG_FOUT, 'pred_val: %f' % (pred_val[i]))
			log_string(LOG_FOUT, 'pred_softmax: %f' % (pred_softmax[i]))
	'''
	fout_out_filelist.close()
	

if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate()
    LOG_FOUT.close()
