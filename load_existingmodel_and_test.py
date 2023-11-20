"""#Created on Mon Nov 20 10:03:11 2023

@author: catarinalopesdias
"""

# load and evaluate a saved model
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
#import matplotlib.pyplot as plt
#import numpy as np
from datahandling import read_and_decode, visualize_all

###############################################################################
# Load existing model
###############################################################################

# model data
num_epochs = 250
batch_size = 2
folder_trn = "trn_synthetic10"
model_name = "models/model_" + str(num_epochs) +"epochs_" + "batchsize"+ str(batch_size) + "_trnfolder_" + folder_trn+".h5"
####


#load model 
model = load_model(model_name)
# summarize model.
model.summary()

#load checkpoints
checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

latest_weights = tf.train.latest_checkpoint(checkpoint_dir)
print(latest_weights)
model.load_weights(latest_weights)

#################################################################
#################################################################

##############################################################
### Preprocess data for testing
#################################################################
tfrecord_files_tst = []
#folder_test = "tst_synthetic50"
folder_test = "tst_2019"

tfrecord_dir_tst = "processed/" + folder_test

# List all files in the directory
files_in_tst_directory = os.listdir(tfrecord_dir_tst)

for file in files_in_tst_directory:
    if file.endswith(".tfrecords"):
        full_path =os.path.join(tfrecord_dir_tst, file)
        tfrecord_files_tst.append(full_path)

tfrecord_dataset_tst = tf.data.TFRecordDataset(tfrecord_files_tst)

# Apply the parsing function to each record
parsed_dataset_tst = tfrecord_dataset_tst.map(read_and_decode)

##################################################################
############ Predict and plot original data
#################################################################
dataset_test_np = parsed_dataset_tst.as_numpy_iterator()

counter = 0
for data_sample_tst in dataset_test_np:
    input_ds_tst, output_ds_tst = data_sample_tst
    
    X_test = tf.expand_dims(input_ds_tst, 0) #batch dimension
    X_test = tf.expand_dims(X_test, 4)  # channel dimension
    
    predicted = model.predict(X_test)

    counter+=1
    print(f"Test set {counter}")
    #visualize(X_test[0,:,:,:,0], para="input")
    #visualize(output_d, para="ref")
    #visualize(predicted[0,:,:,:,0],para="pred")
    
    #prediction_title ="prediction_batch" + str(batch_size) + "_" + str(num_epochs) + "epochs_trn_" folder_trn +"tst" + 
    prediction_title ="prediction_batch" + str(batch_size) + "_" + str(num_epochs) + "epochs_" + folder_trn +"_"+ folder_test + "_" + str(counter)
    print(prediction_title)
    visualize_all(X_test[0,:,:,:,0], output_ds_tst, predicted[0,:,:,:,0], prediction_title)
  # for plotting just the first image - uncomment if you want to see every prediction
    #break



