import tensorflow as tf
import os
import numpy as np
import cv2


from models import H_estimator, output_H_estimator
from utils import DataLoader, load, save
import constant
import skimage


os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU
train_folder = constant.TRAIN_FOLDER
test_folder = constant.TEST_FOLDER
snapshot_dir =  constant.SNAPSHOT_DIR + '/model.ckpt-1000000'
batch_size = constant.TEST_BATCH_SIZE

# define dataset
with tf.name_scope('dataset'):
    ##########testing###############
    test_inputs = tf.placeholder(shape=[batch_size, None, None, 3 * 2], dtype=tf.float32)
    test_size = tf.placeholder(shape=[batch_size, 2, 1], dtype=tf.float32)
    print('test inputs = {}'.format(test_inputs))
    print('test size = {}'.format(test_size))



with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_coarsealignment = output_H_estimator(test_inputs, test_size, False)
    


config = tf.ConfigProto()
config.gpu_options.allow_growth = True      
with tf.Session(config=config) as sess:


    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)

    def inference_func(ckpt):
        print("============")
        print(ckpt)
        load(loader, sess, ckpt)
        print("============")
        
        # Training set (preskoci ako TRAIN_FOLDER ne postoji ili nema parova)
        try:
            print("------------------------------------------")
            print("generating aligned images for training set")
            data_loader_train = DataLoader(train_folder)
            if getattr(data_loader_train, 'flat_pairs', None) is not None:
                length_train = len(data_loader_train.flat_pairs)
            else:
                length_train = data_loader_train.datas['input1']['length']
            out_train = '../output/training'
            for sub in ['warp1', 'warp2', 'mask1', 'mask2']:
                os.makedirs(os.path.join(out_train, sub), exist_ok=True)
            for i in range(0, length_train):
                input_clip = np.expand_dims(data_loader_train.get_data_clips(i, None, None), axis=0)
                size_clip = np.expand_dims(data_loader_train.get_size_clips(i), axis=0)
                coarsealignment = sess.run(test_coarsealignment, feed_dict={test_inputs: input_clip, test_size: size_clip})
                coarsealignment = coarsealignment[0]
                warp1 = (coarsealignment[...,0:3]+1.)*127.5
                warp2 = (coarsealignment[...,3:6]+1.)*127.5
                mask1 = coarsealignment[...,6:9] * 255
                mask2 = coarsealignment[...,9:12] * 255
                cv2.imwrite(os.path.join(out_train, 'warp1', str(i+1).zfill(6) + '.jpg'), warp1)
                cv2.imwrite(os.path.join(out_train, 'warp2', str(i+1).zfill(6) + '.jpg'), warp2)
                cv2.imwrite(os.path.join(out_train, 'mask1', str(i+1).zfill(6) + '.jpg'), mask1)
                cv2.imwrite(os.path.join(out_train, 'mask2', str(i+1).zfill(6) + '.jpg'), mask2)
                print('i = {} / {}'.format(i+1, length_train))
            print("-----------training set done--------------")
        except (ValueError, OSError) as e:
            print("Training set preskocen (TRAIN_FOLDER nedostaje ili je prazan):", e)
            print("------------------------------------------")

        print()
        print("------------------------------------------")
        print("generating aligned images for testing set")
        data_loader = DataLoader(test_folder)
        if getattr(data_loader, 'flat_pairs', None) is not None:
            length = len(data_loader.flat_pairs)
        else:
            length = data_loader.datas['input1']['length']
        out_test = '../output/testing'
        for sub in ['warp1', 'warp2', 'mask1', 'mask2']:
            os.makedirs(os.path.join(out_test, sub), exist_ok=True)
        for i in range(0, length):
            input_clip = np.expand_dims(data_loader.get_data_clips(i, None, None), axis=0)
            size_clip = np.expand_dims(data_loader.get_size_clips(i), axis=0)
            
            coarsealignment = sess.run(test_coarsealignment, feed_dict={test_inputs: input_clip, test_size: size_clip})
            
            coarsealignment = coarsealignment[0]
            warp1 = (coarsealignment[...,0:3]+1.)*127.5
            warp2 = (coarsealignment[...,3:6]+1.)*127.5
            mask1 = coarsealignment[...,6:9] * 255
            mask2 = coarsealignment[...,9:12] * 255
            
            path1 = os.path.join(out_test, 'warp1', str(i+1).zfill(6) + '.jpg')
            path2 = os.path.join(out_test, 'warp2', str(i+1).zfill(6) + '.jpg')
            path3 = os.path.join(out_test, 'mask1', str(i+1).zfill(6) + '.jpg')
            path4 = os.path.join(out_test, 'mask2', str(i+1).zfill(6) + '.jpg')
            cv2.imwrite(path1, warp1)
            cv2.imwrite(path2, warp2)
            cv2.imwrite(path3, mask1)
            cv2.imwrite(path4, mask2)
                     
            print('i = {} / {}'.format(i+1, length))

        print("-----------testing set done--------------")
        print("------------------------------------------")

     
    inference_func(snapshot_dir)



