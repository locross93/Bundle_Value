#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:14:42 2021

@author: logancross
"""
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

base_model = VGG16(weights='imagenet')
pool1 = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_pool').output)
pool2 = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_pool').output)
pool3 = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
pool4 = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
pool5 = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

#food
num_food_imgs = 70
img_feats_block1 = np.zeros([num_food_imgs, 112, 112, 64])
img_feats_block2 = np.zeros([num_food_imgs, 56, 56, 128])
img_feats_block3 = np.zeros([num_food_imgs, 28, 28, 256])
img_feats_block4 = np.zeros([num_food_imgs, 14, 14, 512])
img_feats_block5 = np.zeros([num_food_imgs, 7, 7, 512])

for img_num in range(num_food_imgs):
    img_str = str(img_num+1)
    img_path = bundle_path+'stim_presentation/Bundles_fMRI/data/imgs_food/item_'+img_str+'.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    block1_pool_features = pool1.predict(x)
    block2_pool_features = pool2.predict(x)
    block3_pool_features = pool3.predict(x)
    block4_pool_features = pool4.predict(x)
    block5_pool_features = pool5.predict(x)
    
    img_feats_block1[img_num,:,:,:] = block1_pool_features
    img_feats_block2[img_num,:,:,:] = block2_pool_features
    img_feats_block3[img_num,:,:,:] = block3_pool_features
    img_feats_block4[img_num,:,:,:] = block4_pool_features
    img_feats_block5[img_num,:,:,:] = block5_pool_features
    
np.save(bundle_path+'mvpa/stimulus_features/vgg16/block1_pool_feats_food',img_feats_block1)
np.save(bundle_path+'mvpa/stimulus_features/vgg16/block2_pool_feats_food',img_feats_block2)
np.save(bundle_path+'mvpa/stimulus_features/vgg16/block3_pool_feats_food',img_feats_block3)
np.save(bundle_path+'mvpa/stimulus_features/vgg16/block4_pool_feats_food',img_feats_block4)
np.save(bundle_path+'mvpa/stimulus_features/vgg16/block5_pool_feats_food',img_feats_block5)

#trinket
num_trinket_imgs = 40
img_feats_block1 = np.zeros([num_trinket_imgs, 112, 112, 64])
img_feats_block2 = np.zeros([num_trinket_imgs, 56, 56, 128])
img_feats_block3 = np.zeros([num_trinket_imgs, 28, 28, 256])
img_feats_block4 = np.zeros([num_trinket_imgs, 14, 14, 512])
img_feats_block5 = np.zeros([num_trinket_imgs, 7, 7, 512])

for img_num in range(num_trinket_imgs):
    img_str = str(img_num+1)
    img_path = bundle_path+'stim_presentation/Bundles_fMRI/data/imgs_trinkets/item_'+img_str+'.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    block1_pool_features = pool1.predict(x)
    block2_pool_features = pool2.predict(x)
    block3_pool_features = pool3.predict(x)
    block4_pool_features = pool4.predict(x)
    block5_pool_features = pool5.predict(x)
    
    img_feats_block1[img_num,:,:,:] = block1_pool_features
    img_feats_block2[img_num,:,:,:] = block2_pool_features
    img_feats_block3[img_num,:,:,:] = block3_pool_features
    img_feats_block4[img_num,:,:,:] = block4_pool_features
    img_feats_block5[img_num,:,:,:] = block5_pool_features

np.save(bundle_path+'mvpa/stimulus_features/vgg16/block1_pool_feats_trinket',img_feats_block1)
np.save(bundle_path+'mvpa/stimulus_features/vgg16/block2_pool_feats_trinket',img_feats_block2)
np.save(bundle_path+'mvpa/stimulus_features/vgg16/block3_pool_feats_trinket',img_feats_block3)
np.save(bundle_path+'mvpa/stimulus_features/vgg16/block4_pool_feats_trinket',img_feats_block4)
np.save(bundle_path+'mvpa/stimulus_features/vgg16/block5_pool_feats_trinket',img_feats_block5)