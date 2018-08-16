

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import os, sys
import numpy as np
# import scipy.misc as misc
from model import UNet, creat_deep_Unet
from utils import dice_coef, dice_coef_loss
from loader import dataLoader, deprocess
from PIL import Image
from utils import VIS, mean_IU

# configure args
from opts import *
from opts import dataset_mean, dataset_std # set them in opts


# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == np.max(GT)
    SR = SR.squeeze()
    holder = np.zeros(GT.shape)
    holder[np.where(SR==GT)] = 1
    corr = np.sum(holder)
    tensor_size = SR.size
    acc = float(corr)/float(tensor_size)

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == np.max(GT)
    SR = SR.squeeze()
    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1).astype(np.int32)+(GT==1).astype(np.int32))==2
    FN = ((SR==0).astype(np.int32)+(GT==1).astype(np.int32))==2

    SE = float(np.sum(TP))/(float(np.sum(TP+FN)) + 1e-6)

    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == np.max(GT)
    SR = SR.squeeze()
    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0).astype(np.int32)+(GT==0).astype(np.int32))==2
    FP = ((SR==1).astype(np.int32)+(GT==0).astype(np.int32))==2

    SP = float(np.sum(TN))/(float(np.sum(TN+FP)) + 1e-6)

    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == np.max(GT)
    SR = SR.squeeze()
    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1).astype(np.int32)+(GT==1).astype(np.int32))==2
    FP = ((SR==1).astype(np.int32)+(GT==0).astype(np.int32))==2

    PC = float(np.sum(TP))/(float(np.sum(TP+FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == np.max(GT)
    SR = SR.squeeze()
    Inter = np.sum((SR.astype(np.int32)+GT.astype(np.int32))==2)
    Union = np.sum(((SR.astype(np.int32)+GT.astype(np.int32))>=1).astype(np.int32))

    JS = float(Inter)/(float(Union) + 1e-6)

    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == np.max(GT)
    SR = SR.squeeze()
    Inter = np.sum(((SR.astype(np.int32)+GT.astype(np.int32))==2).astype(np.int32))
    DC = float(2*Inter)/(float(np.sum(SR.astype(np.int32))+np.sum(GT.astype(np.int32))) + 1e-6)

    return DC



opt.data_path = os.getcwd()

vis = VIS(save_path=opt.load_from_checkpoint)

# configuration session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# define data loader
img_shape = [opt.imSize, opt.imSize, opt.num_channels]
test_generator, test_samples = dataLoader(opt.data_path+'\\test1\\', 1,  img_shape, train_mode=False)
# test_generator, test_samples = dataLoader(opt.data_path+'/train/', 1,  img_shape, train_mode=False)
# define model, the last dimension is the channel
label = tf.placeholder(tf.int32, shape=[None]+img_shape[:-1])
with tf.name_scope('unet'):
     # model = UNet().create_model(img_shape=img_shape+[3], num_class=opt.num_class)
    model = UNet().create_model(img_shape=img_shape, num_class=opt.num_class)
    img = model.input
    pred = model.output

# with tf.name_scope('unet'):
#    model = creat_deep_Unet().create_model(img_shape=img_shape, num_class=opt.num_class)
#    img = model.input
#    pred = model.output
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
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    precision_list = []
    F1_list = []
    JS_list = []
    DC_list = []

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

        pred_map_batch = pred_logits > 0.5
        pred_map = pred_map_batch.squeeze()
        score = vis.add_sample(pred_map, y_batch[0])
        im, gt = deprocess(x_batch[0], dataset_mean, dataset_std, y_batch[0])
        vis.save_seg(pred_map, name='{0:04d}_{1:.3f}.png'.format(it, score), im=im, gt=gt)
        accuracy = get_accuracy(np.copy(pred_logits),np.copy(y_batch[0]))
        sensitivity = get_sensitivity(np.copy(pred_logits),np.copy(y_batch[0]))
        specificity = get_specificity(np.copy(pred_logits),np.copy(y_batch[0]))
        precision = get_precision(np.copy(pred_logits),np.copy(y_batch[0]))
        F1 = get_F1(np.copy(pred_logits),np.copy(y_batch[0]))
        JS = get_JS(np.copy(pred_logits),np.copy(y_batch[0]))
        DC = get_DC(np.copy(pred_logits),np.copy(y_batch[0]))

        print ('[accuracy %f],[sensitivity %f],[specificity %f],[precision %f],[F1 %f],[JS %f],[DC %f]' % (accuracy, sensitivity, specificity,precision,F1,JS,DC))
        accuracy_list.append(accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        precision_list.append(precision)
        F1_list.append(F1)
        JS_list.append(JS)
        DC_list.append(DC)
    vis.compute_scores()
    accuracy_mean = np.mean(np.array(accuracy_list))
    sensitivity_mean= np.mean(np.array(sensitivity_list))
    specificity_mean= np.mean(np.array(specificity_list))
    precision_mean= np.mean(np.array(precision_list))
    F1_mean= np.mean(np.array(F1_list))
    JS_mean= np.mean(np.array(JS_list))
    DC_mean= np.mean(np.array(DC_list))
    print ('[mean accuracy %f],[mean sensitivity %f],[mean specificity %f],[mean precision %f],[mean F1 %f],[mean JS %f],[mean DC %f]' %
           (accuracy_mean, sensitivity_mean, specificity_mean,precision_mean,F1_mean,JS_mean,DC_mean))



