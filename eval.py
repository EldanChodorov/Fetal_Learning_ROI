'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-25 02:20:32
 * @modify date 2017-05-25 02:20:32
 * @desc [description]
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import os, sys
import numpy as np
import scipy.misc as misc
from model import UNet
from utils import dice_coef, dice_coef_loss
from loader import dataLoader, deprocess
from PIL import Image
from utils import VIS, mean_IU

# configure args
from opts import *
from opts import dataset_mean, dataset_std # set them in opts

opt.data_path = os.getcwd()

vis = VIS(save_path=opt.load_from_checkpoint)

# configuration session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# define data loader
img_shape = [opt.imSize, opt.imSize, opt.num_channels]
test_generator, test_samples = dataLoader(opt.data_path+'\\test\\', 1,  img_shape, train_mode=False)
# test_generator, test_samples = dataLoader(opt.data_path+'/train/', 1,  img_shape, train_mode=False)
# define model, the last dimension is the channel
label = tf.placeholder(tf.int32, shape=[None]+img_shape[:-1])
with tf.name_scope('unet'):
    # model = UNet().create_model(img_shape=img_shape+[3], num_class=opt.num_class)
    model = UNet().create_model(img_shape=img_shape, num_class=opt.num_class)
    img = model.input
    pred = model.output
# define loss
# with tf.name_scope('cross_entropy'): 
#     cross_entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=pred))
with tf.name_scope('dice_loss'):
    dice_loss = dice_coef_loss(label, pred)

saver = tf.train.Saver() # must be added in the end

''' Main '''
init_op = tf.global_variables_initializer()
sess.run(init_op)
with sess.as_default():
    # restore from a checkpoint if exists
    try:
        last_checkpoint = tf.train.latest_checkpoint(opt.checkpoint_path)
        # saver.restore(sess, opt.load_from_checkpoint)
        print ('--> load from checkpoint '+last_checkpoint)
        saver.restore(sess, last_checkpoint)
    except:
        print ('unable to load checkpoint ...')
        sys.exit(0)
    dice_score = 0
    for it in range(0, test_samples):
        x_batch, y_batch = next(test_generator)
        # tensorflow wants a different tensor order
        feed_dict = {   
                        img: x_batch,
                        label: y_batch
                    }
        # loss, pred_logits = sess.run([cross_entropy_loss, pred], feed_dict=feed_dict)
        # pred_map = np.argmax(pred_logits[0], axis=2)
        loss, pred_logits = sess.run([dice_loss, pred], feed_dict=feed_dict)

        pred_map_batch = pred_logits > 0.001
        pred_map = pred_map_batch.squeeze()
        score = vis.add_sample(pred_map, y_batch[0])

        im, gt = deprocess(x_batch[0], dataset_mean, dataset_std, y_batch[0])
        vis.save_seg(pred_map, name='{0:04d}_{1:.3f}.png'.format(it, score), im=im, gt=gt)

        print ('[iter %f]: loss=%f, meanIU=%f' % (it, loss, score))

    vis.compute_scores()
