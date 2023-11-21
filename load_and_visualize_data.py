"""
# LOAD data and visualize it
Created on Tue Nov 21 09:34:26 2023

@author: catarinalopesdi
""" 
from keras.layers import Input ,Conv3D, Conv3DTranspose, LeakyReLU, UpSampling3D, Concatenate , Add, BatchNormalization
from keras.models import Model
from keras.initializers import GlorotNormal
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import pickle
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow.keras.models import save_model
from datahandling import read_and_decode_tf, visualize_all


def visualize_files_dir(input_f, output_f , title ):
  
  #shape input
  input_shape = list(input_f.shape)
  #print("Input shape:", input_shape)
  #shape output
  output_shape = list(output_f.shape)
  #
  
  #cut 3 slices from input 
  input_slice_x = input_f[input_shape[0]//2, :, :]
  input_slice_y = input_f[:, input_shape[1]//2, :]
  input_slice_z = input_f[:, :, input_shape[2]//2]
  #input_data_shape
  #cut 3 slices from output 
  output_slice_x = output_f[output_shape[0]//2, :, :]
  output_slice_y = output_f[:, output_shape[1]//2, :]
  output_slice_z = output_f[:, :, output_shape[2]//2]

  
  #Get max min of reference
  ref_min = -1
  ref_max = 1

  ####################################################################
  fig = plt.figure(figsize=(5, 5),  edgecolor="black" )
  fig.suptitle(title)

  
  grid = ImageGrid(fig, 211,
     nrows_ncols = (1,3),
     axes_pad = 0.3,
     cbar_location = "right",
     cbar_mode="single",
     cbar_size="5%",
     cbar_pad=0.3,
     share_all=True
     )

  grid[0].imshow(input_slice_x, cmap='gray', vmin = ref_min, vmax = ref_max)
  grid[0].set_title(" X plane")
  grid[0].get_xaxis().set_ticks([])
  grid[0].get_yaxis().set_ticks([])
  grid[0].set_ylabel("Input data")
   
  grid[1].imshow(input_slice_y, cmap='gray',vmin = ref_min, vmax = ref_max)
  grid[1].set_title("Y plane")
     
  kk = grid[2].imshow(input_slice_z, cmap='gray',vmin = ref_min, vmax = ref_max)
  grid[2].set_title("Z plane")
   
  grid.cbar_axes[0].colorbar(kk)

   #### output
   
  grid = ImageGrid(fig, 212,
      nrows_ncols = (1,3),
      axes_pad = 0.2,
      cbar_location = "right",
      cbar_mode="single",
      cbar_size="5%",
      cbar_pad=0.3,
      share_all=True
      )

  grid[0].imshow(output_slice_x, cmap='gray',  vmin=ref_min, vmax=ref_max)
  grid[0].set_title("X plane")
  grid[0].get_xaxis().set_ticks([])
  grid[0].get_yaxis().set_ticks([])
  grid[0].set_ylabel("Output data")
    
  grid[1].imshow(output_slice_y, cmap='gray',  vmin=ref_min, vmax=ref_max)
  grid[1].set_title("Y plane")

  ll = grid[2].imshow(output_slice_z, cmap='gray',  vmin=ref_min, vmax=ref_max)
  grid[2].set_title("Z plane")
  grid.cbar_axes[0].colorbar(ll)    
  

 ####################################################
  ##    Process data for training and define hyperparameters
  ####################################################

tfrecord_files = []
tfrecord_dir = "processed/trn_synthetic50from100" 


# List all files in the directory
files_in_directory = os.listdir(tfrecord_dir)

for file in files_in_directory:
    if file.endswith(".tfrecords"):
        full_path =os.path.join(tfrecord_dir, file)
        tfrecord_files.append(full_path)

# Create a dataset from the list of TFRecord files -Load dataset
tfrecord_dataset = tf.data.TFRecordDataset(tfrecord_files)   

parsed_dataset = tfrecord_dataset.map(read_and_decode_tf)

dataset_np = parsed_dataset.as_numpy_iterator()
#Returns an iterator which converts all elements of the dataset to numpy.


counter = 0;
# visualize the first tst data sample
for data_sample in dataset_np:

    input_ds, output_ds = data_sample #element of parsed test data

    #break
    
    title = "Folder "+ tfrecord_dir +"\n File "+ str(counter)
    visualize_files_dir(input_ds, output_ds ,title)
    counter = counter +1
    if counter > 3:
        break
    #break
    
    
    


