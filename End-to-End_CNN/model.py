# encoding:gbk
from __future__ import division
import os
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

'''Organizational Data Format'''
def change_format(data, batchsize, deep, height, weight, channel, mark):
	
	data = np.array(data)
	if mark == 'TRUE':
		outdata = data.reshape(batchsize, deep, height, weight, channel)
		# Turn one-dimensional arrays into n-dimensional arrays
		# Tensor to array, list can not be converted to tensor, otherwise there will be problems with subsequent input
		
	elif mark == 'FALSE':
		outdata = data.reshape(batchsize, deep, height, weight)
		# Turn a one-dimensional array into an n-dimensional tensor
	
	return outdata

'''Read csv data'''
'''
Just read the features sequentially element by element, 
the number of NDWHC brackets from outside to inside.
eg.[a1,a2,a3,b1,b2,b3,c1,c2,c3]
N:batch_size; D:K; W:J; H:I; C:Feature
'''
def read_csv(filepath, filenames, batchsize, deep, height, weight, channel):
	
	data = []
	label = []
	
	# open() reads the path plus the filename, so the path needs to be passed as an argument to make a splice
	for name in filenames:
		
		csvfile = open(os.path.join(filepath, name), 'r')
		reader = csv.reader(csvfile)
		for item in reader:
			if reader.line_num == 1: # The first line is the attribute name
				continue
			
			a = np.array(item[3:10])
			b = np.array(item[16:20])
			data.append(np.concatenate((a,b)).tolist())
			label.append(item[-1]) # The next two columns are labels, and the last column is read as non-sparse
		
	data = np.array(data)
	label = np.array(label)
	data = data.flatten() # Convert a multi-dimensional array to a one-dimensional array
	data = data.astype(np.float) # Convert a string to a float
	label = label.astype(np.int)
	data = data.tolist() # Array to list
	label = label.tolist()
	outdata = change_format(data, batchsize, deep, height, weight, channel, 'TRUE') # Convert a list to a multidimensional array
	label = change_format(label, batchsize, deep, height, weight, channel, 'FALSE') # List to multidimensional array, non-sparse
			
	return outdata, label

'''k-fold cross-validation'''
def cross_validation(filelist, fold):
	'''
	input: filelist input is a list of file names, in array form
	output: index of each category contained
	The StratifiedKFold method will stratify the data (x data) based on the category (y data), 
	which is not used because there is no y data.
	The KFold method is used here.
	'''
	
	filelist = np.array(filelist)
	
	kf = KFold(n_splits = fold, random_state=2020, shuffle=True)
	#random_state: random number seed, only effective when shuffle is TRUE
	kf.get_n_splits(filelist) # The query is divided into several groups
	
	train_list = []
	test_list = []
	
	# The index is passed after the division, and the index is converted to a list of data
	for train_index, test_index in kf.split(filelist):
		X_train,X_test = filelist[train_index],filelist[test_index]
		train_list.append(X_train.tolist())
		test_list.append(X_test.tolist())
	
	return train_list, test_list
	
'''Placeholder'''
def placeholder_inputs(batch_size, size, channel):
	
    pointclouds_pl = tf.placeholder(tf.float32,
                                     shape=(batch_size, size, size, size, channel))
    labels_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, size, size, size)) #非稀疏
    
    return pointclouds_pl, labels_pl

'''Output'''
def log_string(LOG_FOUT, out_str):
	
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
    
'''Calculate learning rate'''
def get_learning_rate(base_learning_rate, 
                      batch, batch_size,
                      decay_step,
                      decay_rate):
						  
    learning_rate = tf.train.exponential_decay(
                        base_learning_rate,  # Base learning rate.
                        batch * batch_size,  # Current index into the dataset.
                        decay_step,          # Decay step.
                        decay_rate,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    
    return learning_rate        

'''Calculate decay rate'''
def get_bn_decay(bn_init_decay,
                 batch, batch_size,
                 bn_decay_decay_step,
                 bn_decay_decay_rate,
                 bn_decay_clip):
					 
    bn_momentum = tf.train.exponential_decay(
                      bn_init_decay,
                      batch * batch_size,
                      bn_decay_decay_step,
                      bn_decay_decay_rate,
                      staircase=True)
    bn_decay = tf.minimum(bn_decay_clip, 1 - bn_momentum)
    
    return bn_decay

'''
Perform batch normalization :
This is a pre-processing operation in the middle of the neural network layer, 
i.e., the input of the previous layer is normalized before entering the next layer of the network.
This can effectively prevent "gradient dispersion", accelerate network training, and speed up convergence.
The use of DropOut can be reduced or even eliminated.
'''
def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
  """ Batch normalization on convolutional maps and beyond.
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch-normalized maps
  """
  with tf.variable_scope(scope) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(is_training,
                           lambda: ema.apply([batch_mean, batch_var]),
                           lambda: tf.no_op())
    
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
  return normed
  
'''Three-dimensional convolution normalization'''
def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 3D convolutional maps.
  Args:
      inputs:      Tensor, 5D BDHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2,3], bn_decay)

'''y=wx+b, w为weight_variable, b为bias_variable'''
'''Initialize the filter, w with respect to the input image'''
def weight_variable(shape, seed=0):
	
    initial = tf.truncated_normal(shape, stddev=0.1, seed=seed) # Initial assignment, normal distribution, standard deviation of 0.1
    w = tf.Variable(initial)
    
    return w
 
'''Initialize bias, b with respect to the output image'''
def bias_variable(shape):
	
    initial = tf.constant(0.1, shape=shape) # Initialize bias, padding with 0.1
    b = tf.Variable(initial)
    
    return b
    
'''3D convolution'''
'''
data_format: optional string, can be: "NDHWC", "NCDHW"; default is "NDHWC". 
             The data format of the input and output data.
With the default format "NDHWC", the data is stored in the following order: 
             [batch, in_depth, in_height, in_width, in_channels].
Alternatively, the format can be "NCDHW", and the data is stored in the following order: 
             [batch,in_channels,in_depth,in_height,in_width].
The type of input data is dtype = tf.float32, otherwise it may be wrong.
'''
def conv3D(inputs, 
           kernel_size, input_channels, output_channels,
           strides, 
           padding = 'SAME',
           seed = 0,
           dp = False,
           bn = False,
           bn_decay = None,
           is_training = None):
	
	kernel_d, kernel_h, kernel_w = kernel_size
	kernel_shape = [kernel_d, kernel_h, kernel_w, input_channels, output_channels]
	w = weight_variable(kernel_shape, seed)
	
	b = bias_variable([output_channels])
	
	stride_d, stride_h, stride_w = strides
	stride = [1, stride_d, stride_h, stride_w, 1]
	
	result = tf.nn.bias_add(tf.nn.conv3d(inputs, w, stride, padding = "SAME"), b)
	
	if dp:
		result = tf.nn.dropout(result, 0.8)
	
	if bn:
		result = batch_norm_for_conv3d(result, is_training, bn_decay=bn_decay, scope='bn')
	
	result = tf.nn.relu(result)
	
	return result

'''3D maximum pooling'''
'''data_format='NDHWC'''
def max_pooling_3D(inputs, 
                   kernel_size, 
                   strides, 
                   padding = 'VALID'):
	
	kernel_d, kernel_h, kernel_w = kernel_size
	ksize = [1, kernel_d, kernel_h, kernel_w, 1]
	
	stride_d, stride_h, stride_w = strides
	stride = [1, stride_d, stride_h, stride_w, 1]
	
	result = tf.nn.max_pool3d(inputs, ksize, stride, padding)
	
	return result

'''3D average pooling'''
'''data_format='NDHWC'''
def avg_pooling_3D(inputs, 
                   kernel_size, 
                   strides, 
                   padding = 'VALID'):
	
	kernel_d, kernel_h, kernel_w = kernel_size
	ksize = [1, kernel_d, kernel_h, kernel_w, 1]
	
	stride_d, stride_h, stride_w = strides
	stride = [1, stride_d, stride_h, stride_w, 1]
	
	result = tf.nn.avg_pool3d(inputs, ksize, stride, padding)
	
	return result
	
'''3D deconvolution'''
def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
	
	dim_size *= stride_size
	
	if padding == 'VALID' and dim_size is not None:
		dim_size += max(kernel_size - stride_size, 0)
		
	return dim_size
          
def transport_conv3D(inputs,
                     kernel_size, input_channels, output_channels,
                     strides,
                     padding = 'SAME',
                     seed = 0,
                     bn = False,
					 bn_decay = None,
					 is_training = None):
	
	kernel_d, kernel_h, kernel_w = kernel_size
	kernel_shape = [kernel_d, kernel_h, kernel_w, output_channels, input_channels]
	w = weight_variable(kernel_shape, seed)
	
	b = bias_variable([output_channels])
	
	stride_d, stride_h, stride_w = strides
	stride = [1, stride_d, stride_h, stride_w, 1]
	
	#output shape
	batch_size = inputs.get_shape()[0].value
	deep = inputs.get_shape()[1].value
	height = inputs.get_shape()[2].value
	width = inputs.get_shape()[3].value
	out_deep = get_deconv_dim(deep, stride_d, kernel_d, padding)
	out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
	out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
	output_shape = [batch_size, out_deep, out_height, out_width, output_channels]
	
	result = tf.nn.bias_add(tf.nn.conv3d_transpose(inputs, w, output_shape, stride, padding = "SAME"), b)

	return result
	

'''model'''
def get_model(inputs, is_training, bn_decay = None):
	
	'''
	deepening the feature channels of the data.
    Depending on the size of the received domain a 7×7×7 convolution can be replaced by three 3×3×3 convolutions.
	Similarly, a 5×5×5 convolution can be replaced by two 3×3×3 convolutions.
	This operation can reduce the computational effort and computational parameters.
	'''
	#input:16×16×16×11
	layer = conv3D(inputs, [3,3,3], 11, 20, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	               
	#input:16×16×16×20
	layer = conv3D(layer, [3,3,3], 20, 32, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	               
	#input:16×16×16×32
	layer = conv3D(layer, [3,3,3], 32, 64, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	
	#input:16×16×16×64, output:8×8×8×64
	layer_one_out = max_pooling_3D(layer, [3,3,3], [2,2,2], padding = 'SAME')
	'''
	valid: no padding
	same: padding with 0
	'''
	
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	
	#input:8×8×8×64
	layer = conv3D(layer_one_out, [3,3,3], 64, 128, [1,1,1],
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	
	#input:8×8×8×128
	layer = conv3D(layer, [3,3,3], 128, 192, [1,1,1],
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	
	#input:8×8×8×192, output:8×4×4×192
	layer = max_pooling_3D(layer, [3,3,3], [2,2,2], padding = 'SAME')
	
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	
	#input:8×4×4×192; output:4×4×4×64
	layer_1 = conv3D(layer, [1,1,1], 192, 32, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#input:8×4×4×192; output:4×4×4×128
	layer_2 = conv3D(layer, [1,1,1], 192, 32, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	                 
	layer_2 = conv3D(layer_2, [3,3,3], 32, 64, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	                 
	layer_2 = conv3D(layer_2, [3,3,3], 64, 128, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#input:4×4×4×192; output:4×4×4×32
	layer_3 = conv3D(layer, [1,1,1], 192, 32, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	                 
	layer_3 = conv3D(layer_3, [3,3,3], 32, 64, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#input:4×4×4×192; output:4×4×4×32
	layer_4 = max_pooling_3D(layer, [3,3,3], [1,1,1], padding = 'SAME')
	                 
	layer_4 = conv3D(layer_4, [1,1,1], 192, 32, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#output:8×4×4×256
	layer = tf.concat([layer_1, layer_2, layer_3, layer_4], -1)
	
	layer = conv3D(layer, [1,1,1], 256, 256, [1,1,1],
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	               
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	
	#input:4×4×4×256; output:4×4×4×64
	layer_1 = conv3D(layer, [1,1,1], 256, 64, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#input:4×4×4×192; output:4×4×4×128
	layer_2 = conv3D(layer, [1,1,1], 256, 64, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	                 
	layer_2 = conv3D(layer_2, [3,3,3], 64, 128, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	                 
	layer_2 = conv3D(layer_2, [3,3,3], 128, 256, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#input:4×4×4×192; output:4×4×4×32
	layer_3 = conv3D(layer, [1,1,1], 256, 64, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	                 
	layer_3 = conv3D(layer_3, [3,3,3], 64, 128, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#input:4×4×4×192; output:4×4×4×32
	layer_4 = max_pooling_3D(layer, [3,3,3], [1,1,1], padding = 'SAME')
	                 
	layer_4 = conv3D(layer_4, [1,1,1], 256, 64, [1,1,1], 
	                 padding = 'SAME', seed = 0, dp = False, bn = True, 
	                 bn_decay = bn_decay, is_training = is_training)
	
	#output:4×4×4×512
	layer = tf.concat([layer_1, layer_2, layer_3, layer_4], -1)
	
	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	
	#upsampling:input:8×4×4×512; output:16×8×8×512
	layer = transport_conv3D(layer, [5,5,5], 512, 512, [2,2,2],
							 padding = 'SAME', seed = 0, bn = False,
							 bn_decay = None, is_training = None)
							 
	#output: 8×8×8×576
	layer = tf.concat([layer, layer_one_out], -1)
	
	#output: 16×16×16×576
	layer = transport_conv3D(layer, [3,3,3], 576, 576, [2,2,2],
							 padding = 'SAME', seed = 0, bn = False,
							 bn_decay = None, is_training = None)

	''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	
	# Feature channel reduction
	layer = conv3D(layer, [3,3,3], 576, 256, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	               
	layer = conv3D(layer, [1,1,1], 256, 128, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	               
	layer = conv3D(layer, [3,3,3], 128, 64, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	               
	layer = conv3D(layer, [1,1,1], 64, 32, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	
	layer = conv3D(layer, [3,3,3], 32, 16, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	
	layer = conv3D(layer, [1,1,1], 16, 2, [1,1,1], 
	               padding = 'SAME', seed = 0, dp = False, bn = True, 
	               bn_decay = bn_decay, is_training = is_training)
	#output:16×16×16×2
	#Because it is binary classification, two feature maps are output
	
	return layer

'''
dice loss
Consider two classes
'''
def dice_loss(output, target, loss_type='jaccard', axis = (1, 2, 3, 4), smooth=1e-5):
	#dice_loss(predict, gt, loss_type='jaccard', axis = (-1), smooth=1e-5)
	#dice_loss(output, target, loss_type='jaccard', axis = (1, 2, 3, 4), smooth=1e-5)
	"""
	Soft dice (Sorensen or Jaccard) coefficient for comparing the similarity of two batch of data, 
	usually be used for binary image segmentation
	i.e. labels are binary. 
	The coefficient between 0 to 1, 1 means totally match.
	
	Parameters
	-----------
	output : Tensor
		A distribution with shape: [batch_size, ....], (any dimensions).
	target : Tensor
		The target distribution, format the same with `output`.
	loss_type : str
		``jaccard`` or ``sorensen``, default is ``jaccard``.
	axis : tuple of int
		All dimensions are reduced, default ``[1,2,3]``.
	smooth : float
	This small value will be added to the numerator and denominator.
		- If both output and target are empty, it makes sure dice is 1.
		- If either output or target are empty (all pixels are background), 
		dice = '''smooth/(small_value + smooth)''', 
		then if smooth is very small, 
		dice close to 0 (even the image values lower than the threshold), 
		so in this case, higher smooth can have a higher dice.
		
	Examples
	---------
		>>> outputs = tl.act.pixel_wise_softmax(network.outputs)
		>>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)
	References
	-----------
		- `Wiki-Dice <https://en.wikipedia.org/wiki/Sorensen–Dice_coefficient>`__
	"""
	#convert non-sparse labels to sparse labels
	#tf.one_hot(indices, depth, on_value, off_value, axis)
	# 'indices' is a list specifying the unique hot positions of the unique hot vectors in the tensor, 
	# or a list of labels whose indeces are non-negative integers. len(indices) is the number of categories in the classification.
	# The order of the tensor returned by tf.one_hot is the order of indeces + 1.
	# When a component of indices takes -1, i.e., the corresponding vector has no unique hot value.
	# 'depth' is the dimension of each indeces vector
	# 'on_value' is the unique heat value
	# 'off_value' is the non-unique value
	# 'axis' specifies the depth dimension of the unique heat vector, the default is -1, i.e., the last dimension of the specified tensor is the unique heat vector.
	# For example, for a tensor of order 2, axis=0, i.e., each column vector is a depth dimension vector of unique heat
	# axis=1, i.e., each row vector is a unique hot depth dimension vector. axis=-1 is equivalent to axis=1
	# tf.one_hot(indices, depth, on_value, off_value, axis)

	N = len(np.array(target).reshape(-1))
	target = tf.to_float(tf.one_hot(indices = target, depth = 2, on_value = 1, off_value = 0))
	
	output = tf.nn.softmax(output)
	
	#dice loss: Sparse type tags only, unique heat code
	inse = tf.reduce_sum(output * target, axis=axis)
	
	if loss_type == 'jaccard':
		l = tf.reduce_sum(output * output, axis=axis)
		r = tf.reduce_sum(target * target, axis=axis)
	elif loss_type == 'sorensen':
		l = tf.reduce_sum(output, axis=axis)
		r = tf.reduce_sum(target, axis=axis)
	else:
		raise Exception("Unknow loss_type")
		
	dice_coe = (2. * inse + smooth) / (l + r + smooth)
	dice_coe = tf.reduce_mean(dice_coe) / N
	dice = 1 - dice_coe
	
	return dice

'''
focal loss
'''
def focal_loss(pred, label, alpha = 0.25, gamma = 2):
	
	alpha = tf.constant(alpha, dtype=tf.float32)
	gamma = tf.constant(gamma, dtype=tf.float32)
	epsilon = 1.e-8
	
	y_true = tf.one_hot(label, 2) # binary classification
	probs = tf.nn.sigmoid(pred) # The sigmoid is used
	
	y_pred = tf.clip_by_value(probs, epsilon, 1. - epsilon)
	
	weight = tf.multiply(y_true, tf.pow(tf.subtract(1., y_pred), gamma))
	if alpha != 0.0:
		alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
	else:
		alpha_t = tf.ones_like(y_true)
		
	xent = tf.multiply(y_true, -tf.log(y_pred))
	focal_xent = tf.multiply(alpha_t, tf.multiply(weight, xent))
	reduced_fl = tf.reduce_max(focal_xent, axis=1)
	focal_loss = tf.reduce_mean(reduced_fl)

	return focal_loss
	
'''
Loss function
'''
def get_loss(pred, label):
	
	dice = dice_loss(pred, label)
	
	focal = focal_loss(pred, label, alpha = 0.25, gamma = 2)
	
	hybird_loss = focal + dice * 0.3
	
	return hybird_loss
