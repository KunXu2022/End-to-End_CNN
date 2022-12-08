# encoding:gbk
from __future__ import division
import os
import re
import sys
import math
import argparse
import importlib
import numpy as np
import tensorflow as tf
from model import *

sess = tf.InteractiveSession() # Create a TensorFlow session

parser = argparse.ArgumentParser() # Parse arguments, for passing arguments on the command line

parser.add_argument('--filelist', help='Path to training set ground truth (.csv)', required=True)
parser.add_argument('--num_input_channels', type=int, default=11) # Number of input feature channels
parser.add_argument('--num_classes', type=int, default=2) # Number of categories
parser.add_argument('--data_size', type=int, default=16) # Data size
'''
one epoch = batch * batch_size
'''
parser.add_argument('--epochs', type=int, default=136) # Number of iterations
parser.add_argument('--batch_size', type=int, default=73) # size of each batch is 73
'''
Learning rate update (gradient descent strategy, aiming to speed up convergence).
decayed_learning_rate = learning_rate*decay_rate^(globle_step/decay_steps)
where learning_rate is the initial learning rate set empirically.
     decay_rate is the decay rate coefficient set empirically.
     globle_step is the current training round, epoch or batch.
     decay_steps defines the decay period, which can be used with the decaycase parameter to keep the learning rate constant during the decay_step training rounds.
     The learning rate can be kept constant within decay_step training rounds.
'''
parser.add_argument('--optimizer', default='adam') # optimizer (gradient descent)
parser.add_argument('--learning_rate', type=float, default=0.01) # initial learning rate
parser.add_argument('--decay_step', type=int, default=200000) # decay coefficient
parser.add_argument('--decay_rate', type=float, default=0.9) # decay coefficient
'''
Doing normalization before gradient descent can improve the accuracy
The parameter setting is the same as the learning rate.
'''
parser.add_argument('--bn_init_decay', type=float, default=0.5) # Initial normalization parameter
parser.add_argument('--bn_decay_rate', type=int, default=0.5) # decay coefficient
parser.add_argument('--bn_decay_clip', type=float, default=0.99) # decay coefficient
parser.add_argument('--results_path') # Results storage path
parser.add_argument('--log_dir', default='log') # Running information storage path

FLAGS = parser.parse_args() # instantiate object

# capitalize: global variables
FILELIST = FLAGS.filelist # Variable assignment
LEARNING_RATE = FLAGS.learning_rate
NUM_INPUT_CHANNELS = FLAGS.num_input_channels
NUM_CLASSES = FLAGS.num_classes
DATA_SIZE = FLAGS.data_size
EPOCHS = FLAGS.epochs
BATCH_SIZE = FLAGS.batch_size
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
BN_INIT_DECAY = FLAGS.bn_init_decay
BN_DECAY_RATE = FLAGS.bn_decay_rate
BN_DECAY_CLIP = FLAGS.bn_decay_clip
BN_DECAY_STEP = DECAY_STEP
RESULTS_PATH = FLAGS.results_path
LOG_DIR = FLAGS.log_dir

'''Create log folder'''
if not os.path.exists(LOG_DIR): 
	os.mkdir(LOG_DIR)
os.system('copy model.py %s' % (LOG_DIR)) # back up of model def
os.system('copy train.py %s' % (LOG_DIR)) # back up of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

'''Get the list of data,file name stored in txt'''
# Get the list of data
TRAIN_FILES = []
TEST_FILES = []
for line in open(os.path.join(FILELIST, 'train4.txt')):
	line = re.sub(r"\n","",line) # remove escaped 
	TRAIN_FILES.append(line)
for line in open(os.path.join(FILELIST, 'test4.txt')):
	line = re.sub(r"\n","",line)
	TEST_FILES.append(line)
# ten folds cross-validation of the divided dataset
'''
Data set information: use 11880 csv files
Tenfold cross-validation dataset information: training set 10692
                   test set 1188
batch_size:training set 36
           test_set 36
batch:training set 297
      test_set 33
'''

def train():
	with tf.device('/gpu:0'):
		#with tf.Graph().as_default():
		# placeholder to allocate space in advance for data to be read later 
		# (this method is used by the TensorFlow static framework)
		data, label = placeholder_inputs(BATCH_SIZE, DATA_SIZE, NUM_INPUT_CHANNELS)
		is_training = tf.placeholder(dtype=tf.bool)
		
		# initialize parameters
		# Notify the optimizer to add the "batch" parameter at each training session.
		batch = tf.Variable(0) # initialize first, will update later
		learning_rate = get_learning_rate(LEARNING_RATE, batch, BATCH_SIZE,
										DECAY_STEP, DECAY_RATE)
		bn_decay = get_bn_decay(BN_INIT_DECAY, batch, BATCH_SIZE,
								BN_DECAY_STEP, BN_DECAY_RATE,
								BN_DECAY_CLIP)
		tf.summary.scalar('learning_rate', learning_rate)
		tf.summary.scalar('bn_decay', bn_decay) # Merge chart information and manage summary automatically
		
		'''
		Usage of tf.summary.scalar().
		Function: Save all the summaries to disk for tensorboard to display.
		After running, a log file will be created in the tjn folder to save the results.
		cmd command, switch to the corresponding folder and start tensorboard.
		"tensorboard --logdir='tjn file path'"
		Then type "localhost:6006" on the page, (the address may be different on different hosts)
		'''
		
		# Define the convolution model and loss function
		pred = get_model(data, is_training, bn_decay = bn_decay)
		loss = get_loss(pred, label)
		tf.summary.scalar('loss', loss)
		
		# accuracy
		correct = tf.equal(tf.argmax(pred, -1), tf.to_int64(label)) # compare element by element
		correct = tf.cast(correct, tf.float32) # bool to float32
		accuracy = tf.reduce_sum(correct) / float(BATCH_SIZE)
		tf.summary.scalar('accuracy', accuracy)
		
		'''
		The learning rate is continuously corrected during the run.
		The learning adaption is based on its loss amount. The larger the loss amount, 
		the larger the learning rate and the larger the correction angle.
		The smaller the loss, the smaller the correction, and the smaller the learning rate, 
		but it will not exceed the learning rate set by itself.
		'''
		# Optimization method selection
		if OPTIMIZER == 'momentum':
				optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
		elif OPTIMIZER == 'adam':
			optimizer = tf.train.AdamOptimizer(learning_rate)
		# Correction of parameters
		train_op = optimizer.minimize(loss, global_step = batch)
		'''
		There are two operations inside minimize: 
			(1) calculate the gradient of each variable 
			(2) update the values of these variables with the gradient
		train_op is the optimizer
		'''
		
		# instantiate the object, save and extract the neural network parameters
		saver = tf.train.Saver()
		'''
		The parameters are stored in the checkpoint file, 
		which holds a list of all model files in a directory that
		
		Later, when applying the model, it is obtained directly with 
		"ckpt = tf.train.get_checkpoint_state(model_save_path)".
		'''
	
	# Configure the computing method of tf.Session (GPU or CPU)
	config = tf.ConfigProto() # instantiate object
	config.gpu_options.allow_growth = True # Dynamic request for video memory
	config.allow_soft_placement = True # Automatic selection of running device
	config.gpu_options.per_process_gpu_memory_fraction = 1 # GPU memory usage setting
	config.log_device_placement = False # Do not print out in the terminal which device each operation is running on
	sess = tf.Session(config=config)
	
	# Save all summaries to disk for tensorboard display
	merged = tf.summary.merge_all()
	# set the storage path
	train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),sess.graph)
	test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))
	
	# initialize uninitialized global variables
	init = tf.global_variables_initializer()
	# Pass values into the is_training placeholder
	sess.run(init, {is_training: True})
	
	# Dictionary, passed as an interface to the training and evaluation epoch loops
	ops = {'data': data,
		   'label': label,
		   'is_training':is_training,
		   'pred': pred,
           'loss': loss,
	       'train_op': train_op,
           'merged': merged,
	       'step': batch}
	
	# Perform epoch loop
	for epoch in range(EPOCHS):
		
		# refresh the output at the same location, for a more beautiful visualization
		log_string(LOG_FOUT, '**** EPOCH %03d ****' % (epoch))
		sys.stdout.flush()
		
		# No need to pass the parameters of the placeholder declaration, so it can be run directly
		train_mean_loss, train_accuracy = get_train(sess, ops, train_writer)
		test_mean_loss, test_accuracy, test_avg_class_acc = get_test(sess, ops, test_writer)
		
		# Save models, every 10 saves
		if epoch % 10 == 0:
			save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
			log_string(LOG_FOUT, "Model saved in file: %s" % save_path)
	

		
def get_train(sess, ops, train_writer):
	
	is_training_train = True
	
	# Break up the order of the training data (to prevent overfitting)
	train_file_idxs = np.arange(0, len(TRAIN_FILES))
	np.random.shuffle(train_file_idxs)
	
	# The input TRAIN_FILES is the file that implements an epoch
	num_batch = len(TRAIN_FILES) // BATCH_SIZE
		
	total_correct = 0.0 # Total number of correct classifications
	total_seen = 0.0 # Number of samples traversed
	loss_sum = 0.0 # total loss
	total_seen_class = [0.0 for _ in range(NUM_CLASSES)] # Number of each category
	total_correct_class = [0.0 for _ in range(NUM_CLASSES)] # Number of correct categories
	
	# Get a list of data by index, a one-dimensional array
	filelist = TRAIN_FILES
		
	for batch_idx in range(num_batch):
		
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx + 1) * BATCH_SIZE
		
		filenames = filelist[start_idx: end_idx]
		
		# Read data
		data_train, label_train = read_csv(FILELIST, filenames, BATCH_SIZE, DATA_SIZE, 
		                       DATA_SIZE, DATA_SIZE, NUM_INPUT_CHANNELS)
		
		feed_dict = {ops['data']: data_train,
		             ops['label']: label_train,
		             ops['is_training']: is_training_train}
		             
		summary, step, _, loss, pred = sess.run([ops['merged'], ops['step'], 
			ops['train_op'], ops['loss'], ops['pred']], feed_dict = feed_dict)
		
		train_writer.add_summary(summary, step)
		pred = np.argmax(pred, -1)
		correct = np.sum(pred == label_train)
		total_correct += correct
		total_seen += (BATCH_SIZE * DATA_SIZE * DATA_SIZE * DATA_SIZE)
		loss_sum += loss
		
		# Calculate the average category accuracy
		pred_cls = pred.reshape(-1)
		label = label_train.reshape(-1)
		for i in range(len(label)):
			mark = int(label[i])
			total_seen_class[mark] += 1
			if(mark == int(pred_cls[i])):
				total_correct_class[mark] += 1
		
	train_mean_loss = loss_sum / float(num_batch)
	train_accuracy = total_correct / float(total_seen)
	train_avg_class_acc = np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))
	log_string(LOG_FOUT, 'train mean loss: %f' % (train_mean_loss))
	log_string(LOG_FOUT, 'train accuracy: %f' % (train_accuracy))
	log_string(LOG_FOUT, 'train total correct class1: %f' % (total_correct_class[0]))
	log_string(LOG_FOUT, 'train total correct class2: %f' % (total_correct_class[1]))
	log_string(LOG_FOUT, 'train total seen class1: %f' % (total_seen_class[0]))
	log_string(LOG_FOUT, 'train total seen class2: %f' % (total_seen_class[1]))
	log_string(LOG_FOUT, 'train avg class acc: %f' % (train_avg_class_acc))
	
	return train_mean_loss, train_accuracy
    
def get_test(sess, ops, test_writer):
	
	is_training = False
	
	# The input TEST_FILES is the file that implements an epoch
	num_batch = len(TEST_FILES) // BATCH_SIZE
	
	total_correct = 0.0 # Total number of correct classifications
	total_seen = 0.0 # Number of samples traversed
	loss_sum = 0.0 # Total loss
	total_seen_class = [0.0 for _ in range(NUM_CLASSES)] # Number of each category
	total_correct_class = [0.0 for _ in range(NUM_CLASSES)] # Number of correct categories
	
	# Get a list of data by index, a one-dimensional array
	filelist = TEST_FILES
	
	for batch_idx in range(num_batch):
		
		start_idx = batch_idx * BATCH_SIZE
		end_idx = (batch_idx + 1) * BATCH_SIZE
		
		filenames = filelist[start_idx: end_idx]
		
		# Read the data
		data, label = read_csv(FILELIST, filenames, BATCH_SIZE, DATA_SIZE, 
		                       DATA_SIZE, DATA_SIZE, NUM_INPUT_CHANNELS)
		
		feed_dict = {ops['data']: data,
		             ops['label']: label,
		             ops['is_training']: is_training}
		
		summary, step, loss, pred = sess.run([ops['merged'], ops['step'],
			ops['loss'], ops['pred']], feed_dict = feed_dict)
			
		test_writer.add_summary(summary, step)
		pred = np.argmax(pred, -1)
		correct = np.sum(pred == label)
		total_correct += correct
		total_seen += (BATCH_SIZE * DATA_SIZE * DATA_SIZE * DATA_SIZE)
		loss_sum += (loss * BATCH_SIZE * DATA_SIZE * DATA_SIZE * DATA_SIZE)
		
		# Calculate average category accuracy
		pred = pred.reshape(-1)
		label = label.reshape(-1)
		for i in range(len(label)):
			mark = int(label[i])
			total_seen_class[mark] += 1
			if(mark == int(pred[i])):
				total_correct_class[mark] += 1

	test_mean_loss = loss_sum / float(total_seen)
	test_accuracy = total_correct / float(total_seen)
	test_avg_class_acc = np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))
	log_string(LOG_FOUT, 'test mean loss: %f' % (test_mean_loss))
	log_string(LOG_FOUT, 'test accuracy: %f'% (test_accuracy))
	log_string(LOG_FOUT, 'test total correct class1: %f' % (total_correct_class[0]))
	log_string(LOG_FOUT, 'test total correct class2: %f' % (total_correct_class[1]))
	log_string(LOG_FOUT, 'test total seen class1: %f' % (total_seen_class[0]))
	log_string(LOG_FOUT, 'test total seen class2: %f' % (total_seen_class[1]))
	log_string(LOG_FOUT, 'test avg class acc: %f' % (test_avg_class_acc))
	
	return test_mean_loss, test_accuracy, test_avg_class_acc
	
if __name__ == "__main__":
	train()
	LOD_FOUT.close()
	
