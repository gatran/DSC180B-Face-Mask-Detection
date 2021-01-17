#!/usr/bin/env python
# coding: utf-8

# In[53]:


import tensorflow as tf
import os
import numpy as np


# In[80]:


#Get paths 
train_corr = "/datasets/MaskedFace-Net/train/covered"
train_incorr = "/datasets/MaskedFace-Net/train/incorrect"
train_uncovered = "/datasets/MaskedFace-Net/train/uncovered"

#Holdout == Test set
holdout_corr = "/datasets/MaskedFace-Net/holdout/covered"
holdout_incorr = "/datasets/MaskedFace-Net/holdout/incorrect"
holdout_uncovered = "/datasets/MaskedFace-Net/holdout/uncovered"

val_corr = "/datasets/MaskedFace-Net/validation/covered"
val_incorr = "/datasets/MaskedFace-Net/validation/incorrect"
val_uncovered = "/datasets/MaskedFace-Net/validation/uncovered"


# In[81]:


#filename_dataset = tf.data.Dataset.list_files(train_corr)

#image_dataset = filename_dataset.map(lambda x: tf.io.decode_jpeg(tf.read_file(x)))


# In[82]:


#[1, 1, 1] instead?
train_corr_imgs = os.listdir(train_corr)
labels = ([1] * len(train_corr_imgs))#tf.constant([1] * len(corr_imgs))


# In[83]:


#[?, ?, ?] instead?
train_incorr_imgs = os.listdir(train_incorr)
labels = ([0] * len(train_incorr_imgs))#tf.constant([1] * len(corr_imgs))


# In[84]:


#[0, 0, 0] instead?
train_uncovered_imgs = os.listdir(train_uncovered)
labels = ([0] * len(train_uncovered_imgs))#tf.constant([1] * len(corr_imgs))

