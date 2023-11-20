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

# Check https://colab.research.google.com/drive/1Omj-taD4P4oBBZOrZKf8gdfp8sMP972r?usp=sharing

###########################################
#build convolutional network
def build_CNN(input_tensor):


#########################################################################################################################
    # First convolutional layer with activation and batch normalization
    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(input_tensor)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    # Second convolutional layer with activation and batch normalization
    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    # Add the input tensor to the output tensor (residual connection)
    X_conc3 = Add()([X_save, X])

    #down convolutional layer
    encoding_down_1 = Conv3D(filters=16,
                            kernel_size=[3, 3, 3],
                            activation=LeakyReLU(alpha=0.2),
                            padding='same',
                            kernel_initializer=GlorotNormal(42),
                            strides=(2,2,2))(X_conc3)
    batch_norm_layer_3 = BatchNormalization()(encoding_down_1)
    # batch_norm_layer_3 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(batch_norm_layer_3)                    #i dont know if i need that-check it

#############################################################################################################################
    #Second Layer-->down
    # First convolutional layer with activation and batch normalization
    X = Conv3D(filters=32, kernel_size=[3, 3, 3],padding='same')(batch_norm_layer_3)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    # Second convolutional layer with activation and batch normalization
    X = Conv3D(filters=32, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=32, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    # Add the input tensor to the output tensor (residual connection)
    X_conc2 = Add()([X_save, X])


    encoding_down_2 = Conv3D(filters=32,
                            kernel_size=[3, 3, 3],
                            activation=LeakyReLU(alpha=0.2),
                            padding='same',
                            kernel_initializer=GlorotNormal(42),
                            strides=(2,2,2)
                            )(X_conc2)
    batch_norm_layer_2 = BatchNormalization()(encoding_down_2)
    # batch_norm_layer_2 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(batch_norm_layer_2)
###################################################################################################################################################
    #Third Layer-->down
    # First convolutional layer with activation and batch normalization
    X = Conv3D(filters=64, kernel_size=[3, 3, 3],padding='same')(batch_norm_layer_2)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    # Second convolutional layer with activation and batch normalization
    X = Conv3D(filters=64, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=64, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    # Add the input tensor to the output tensor (residual connection)
    X_conc1 = Add()([X_save, X])
    encoding_down_3 = Conv3D(filters=64,
                            kernel_size=[3, 3, 3],
                            activation=LeakyReLU(alpha=0.2),
                            padding='same',
                            kernel_initializer=GlorotNormal(42),
                            strides=(2,2,2)
                            )(X_conc1)

    batch_norm_layer_3 = BatchNormalization()(encoding_down_3)
    # batch_norm_layer_3 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(batch_norm_layer_3)
#######################################################################################################################
    #Fourth Layer = Connection between Layer
    X = Conv3D(filters=128, kernel_size=[3, 3, 3],padding='same')(batch_norm_layer_3)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    X = Conv3D(filters=128, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=128, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

#####################################################################################################
    #Third Layer --> up NOte: if not workin, you have to slice
    decoder_up_1 = UpSampling3D(size=(2,2,2))(X)
    decoder_1 = Conv3DTranspose(filters=64,
                        kernel_size=[3,3,3],
                        activation=LeakyReLU(alpha=0.2),
                        padding='same')(decoder_up_1)

    #vertical connection between these convolutional layers--> adds the output together
    #TensorShape([None, 3, 32, 32, 64])


    combine_conc1_dec1 = Concatenate()([X_conc1, decoder_1])

    X = Conv3D(filters=64, kernel_size=[3, 3, 3],padding='same')(combine_conc1_dec1)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    X = Conv3D(filters=64, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=64, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)
##########################################################################################################
    #Second layer -->up
    decoder_up_2 = UpSampling3D(size=(2,2,2))(X)
    decoder_2 = Conv3DTranspose(filters=32,
                        kernel_size=[3,3,3],
                        activation=LeakyReLU(alpha=0.2),
                        padding='same')(decoder_up_2)

    #vertical connection between these convolutional layers--> adds the output together

    combine_conc2_dec2 = Concatenate()([X_conc2, decoder_2])

    X = Conv3D(filters=32, kernel_size=[3, 3, 3],padding='same')(combine_conc2_dec2)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    X = Conv3D(filters=32, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=32, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)
########################################################################################################################
    # Third Layer --> up
    decoder_up_3 = UpSampling3D(size=(2,2,2))(X)
    decoder_3 = Conv3DTranspose(filters=16,
                        kernel_size=[3,3,3],
                        activation=LeakyReLU(alpha=0.2),
                        padding='same')(decoder_up_3)

    combine_conc3_dec3 = Concatenate()([X_conc3, decoder_3])

    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(combine_conc3_dec3)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)
###################################################################################################################
    # output before residual conenction
    output_layer = Conv3D(filters=1 ,kernel_size=[3,3,3], padding='same')(X)

    #residual connection between input and output
    residual_conn = Add()([input_tensor, output_layer])
    output_tensor=residual_conn

    return output_tensor


######################################################################################
######################################################################################
### Read tfrecord-files
######################################################################################
######################################################################################


def read_and_decode(example_proto):
    feature_description = {
        'input_data_raw': tf.io.FixedLenFeature([], tf.string),
        'output_data_raw': tf.io.FixedLenFeature([], tf.string),
        'input_height': tf.io.FixedLenFeature([], tf.int64),
        'input_width': tf.io.FixedLenFeature([], tf.int64),
        'input_depth': tf.io.FixedLenFeature([], tf.int64),
        'output_height': tf.io.FixedLenFeature([], tf.int64),
        'output_width': tf.io.FixedLenFeature([], tf.int64),
        'output_depth': tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(example_proto, feature_description)
    #For dense tensors, the returned Tensor is identical 
    #to the output of parse_example, except there is no batch dimension,
    #the output shape is the same as the shape given in dense_shape.



    input_data = tf.io.decode_raw(example['input_data_raw'], tf.float32)
    output_data = tf.io.decode_raw(example['output_data_raw'], tf.float32)
    #tf.io.decode_raw(input_bytes, out_type, little_endian=True, fixed_length=None, name=None)
    #Every component of the input tensor is interpreted as a sequence of bytes.
    #These bytes are then decoded as numbers in the format specified by out_type.
    input_shape = [
        example['input_height'],
        example['input_width'],
        example['input_depth']
    ]

    output_shape = [
        example['output_height'],
        example['output_width'],
        example['output_depth']
    ]

    input_data = tf.reshape(input_data, input_shape)
    output_data = tf.reshape(output_data, output_shape)

    return input_data, output_data


#####################################################################
### VISUALIzE RESULTS
def visualize_all(resized_input, reference , predicted, title ):
  
  #shape input
  input_data_shape = list(resized_input.shape)
  print("Input shape:", input_data_shape)
  
  #shape reference
  reference_shape = list(reference.shape)
  print("reference shape:", reference_shape)
  
  #shape predicted
  predicted_shape = list(predicted.shape)
  print("predicted shape:", predicted_shape)
  
  
  error = predicted - reference
  
  #error shape
  error_shape = list(error.shape)
  print("error shape:", error_shape)
  
  #cut 3 slices from input 
  #input_slice_x = resized_input[input_data_shape[0]//2, :, :]
  #input_slice_y = resized_input[:, input_data_shape[1]//2, :]
  #input_slice_z = resized_input[:, :, input_data_shape[2]//2]
  
  #cut 3 slices from reference 
  reference_slice_x = reference[reference_shape[0]//2, :, :]
  reference_slice_y = reference[:, reference_shape[1]//2, :]
  reference_slice_z = reference[:, :, reference_shape[2]//2]
  
  #cut 3 slices from predicted
  predicted_slice_x = predicted[predicted_shape[0]//2, :, :]
  predicted_slice_y = predicted[:, predicted_shape[1]//2, :]
  predicted_slice_z = predicted[:, :, predicted_shape[2]//2]
  
  
  # 3 slices of error
  
  error_slice_x = predicted_slice_x - reference_slice_x
  error_slice_y = predicted_slice_y - reference_slice_y
  error_slice_z = predicted_slice_z - reference_slice_z
  
  
  #Get max min of reference
  ref_min = tf.reduce_min(reference).numpy()
  ref_max = tf.reduce_max(reference).numpy()
  print("Reference max value", ref_max, "Reference min value", ref_min)

  ####################################################################
  fig = plt.figure(figsize=(10, 10), dpi=100, edgecolor="black" )
  fig.suptitle(title, fontsize=12)

  #plt.title("Model results")
  
  grid = ImageGrid(fig, 311,
     nrows_ncols = (1,3),
     axes_pad = 0.5,
     cbar_location = "right",
     cbar_mode="single",
     cbar_size="5%",
     cbar_pad=1,
     share_all=True
     )


  grid[0].imshow(reference_slice_x, cmap='gray', vmin = ref_min, vmax = ref_max)
  #grid[0].axis('off')
  grid[0].set_title("Reference data X-dim")
  grid[0].get_xaxis().set_ticks([])
  grid[0].get_yaxis().set_ticks([])
   
  grid[1].imshow(reference_slice_y, cmap='gray',  vmin=ref_min, vmax=ref_max)
  grid[1].set_title("Reference data Y-dim")
     
  kk = grid[2].imshow(reference_slice_z, cmap='gray', vmin=ref_min, vmax=ref_max)
  grid[2].set_title("Reference data Z-dim")
   
  grid.cbar_axes[0].colorbar(kk)

   #### predicted
   
  grid = ImageGrid(fig, 312,
      nrows_ncols = (1,3),
      axes_pad = 0.5,
      cbar_location = "right",
      cbar_mode="single",
      cbar_size="5%",
      cbar_pad=1,
      share_all=True
      )

  grid[0].imshow(predicted_slice_x, cmap='gray',  vmin=ref_min, vmax=ref_max)
  grid[0].set_title("predicted data X-dim ")
  grid[0].get_xaxis().set_ticks([])
  grid[0].get_yaxis().set_ticks([])
    
  grid[1].imshow(predicted_slice_y, cmap='gray',  vmin=ref_min, vmax=ref_max)
  grid[1].set_title("predicted data Y-dim ")

  ll = grid[2].imshow(predicted_slice_z, cmap='gray',  vmin=ref_min, vmax=ref_max)
  grid[2].set_title("predicted data Z-dim")
  grid.cbar_axes[0].colorbar(ll)

  
    
  grid = ImageGrid(fig, 313,
     nrows_ncols = (1,3),
     axes_pad = 0.5,
     cbar_location = "right",
     cbar_mode="single",
     cbar_size="5%",
     cbar_pad=1,
     share_all=True
     )
   #### error
  
  error_max = 0.3
  error_min = -0.3
  
  
  grid[0].imshow(error_slice_x, cmap='seismic',aspect='equal', vmin=error_min, vmax=error_max)
  grid[0].set_title("Error data X-dim")
  grid[0].get_xaxis().set_ticks([])
  grid[0].get_yaxis().set_ticks([])
    
  grid[1].imshow(error_slice_y, cmap='seismic',aspect='equal', vmin=error_min, vmax=error_max)
  grid[1].set_title("Error data Y-dim")


  jj = grid[2].imshow(error_slice_z, cmap='seismic',aspect='equal', vmin=error_min, vmax=error_max)
  grid[2].set_title("Error data Z-dim ")
  grid.cbar_axes[0].colorbar(jj)
  
  filename = "images/" + title + ".png"
  plt.savefig(filename)
  plt.show()
  
  




  ####################################################
  ##    Process data for training and define hyperparameters
  ####################################################
tfrecord_files_train = []
#folder_trn = "trn_synthetic100"
folder_trn = "trn_synthetic10"

tfrecord_dir_trn = "processed/" + folder_trn


# List all files in the directory
files_in_trn_directory = os.listdir(tfrecord_dir_trn)

for file in files_in_trn_directory:
    if file.endswith(".tfrecords"):
        full_path =os.path.join(tfrecord_dir_trn, file)
        tfrecord_files_train.append(full_path)

# Create a dataset from the list of TFRecord files -Load dataset
tfrecord_dataset_train = tf.data.TFRecordDataset(tfrecord_files_train)   

# Apply the parsing function to each record
tfrecord_dataset_train = tfrecord_dataset_train.map(read_and_decode)

#tf.data.TFRecordDataset(filenames, compression_type=None,
#buffer_size=None,    num_parallel_reads=None,name=None)
#This dataset loads TFRecords from the files as bytes, exactly as they were written.
#TFRecordDataset does not do any parsing or decoding on its own. 
#Parsing and decoding can be done by applying Dataset.map transformations after the TFRecordDataset.

################################################################################
##################################################################################
# batch size and number of epochs
batch_size = 2#2
num_epochs = 250#2#250#200

# Shuffle and batch the dataset
tfrecord_dataset_train = tfrecord_dataset_train.shuffle(buffer_size=1000)
tfrecord_dataset_train = tfrecord_dataset_train.batch(batch_size)
tfrecord_dataset_train = tfrecord_dataset_train.repeat(num_epochs)

##############################################################
### Preprocess data for testing
#################################################################
tfrecord_files_tst = []
folder_test = "tst_synthetic50"
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



#############################################################
#### Compile the model
#############################################################
input_shape = (None, None, None, 1) # shape of pict, last is the channel
input_tensor = Input(shape = input_shape, name="input")

ushape1 = build_CNN(input_tensor)
ushape2 = build_CNN(ushape1)

model = Model(input_tensor, ushape2) #get from orig deepQSM algorithm

model.compile(optimizer='adam', loss='mean_absolute_error')


###################################################"model.h5"#########
### Predict model without training
############################################################
dataset_test_np = parsed_dataset_tst.as_numpy_iterator()
#Returns an iterator which converts all elements of the dataset to numpy.

#Use as_numpy_iterator to inspect the content of your dataset. 
#To see element shapes and types, 
#print dataset elements directly instead of using as_numpy_iterator.
#Eager execution is a powerful execution environment that evaluates operations immediately. 
#It does not build graphs, and the operations return actual values instead of computational graphs to run later.


counter= 0
target_shape= [128,128,128]

print("Elements of parsed test data: \n", dataset_test_np)


# visualize the first tst data sample
for data_sample_tst in dataset_test_np:
    input_ds_tst, output_ds_tst = data_sample_tst #element of parsed test data

    print("Input ds test : shape", input_ds_tst.shape)
    ref_data = output_ds_tst #for plotting
    
    X_test = tf.expand_dims(input_ds_tst, 0)      #Returns a tensor with a length 1 axis inserted at index axis. ???
    X_test = tf.expand_dims(X_test, 4) #Returns a tensor with a length 1 axis inserted at index axis. add teh channel dim
    
    predicted = model.predict(X_test)

    counter+=1
    print(f"Test set {counter}")
    print("X_test")
    
  # for plotting just the first image - uncomment if you want to see every prediction
    visualize_all(X_test[0,:,:,:,0], ref_data, predicted[0,:,:,:,0], "no_training") #why??? what are these 0 0 batch and channel
    break
    


#################################################################
### Train the model
#################################################################
def data_generator(train_data):
    for input_data_sample, output_data_sample in train_data:
        
        input_data_shape = list(input_data_sample.shape)
        #y= list(output_data_sample.shape)
        
        print("input data shape:", input_data_shape) 
        input_data_shape
        target_shape = [input_data_shape[0], input_data_shape[1],input_data_shape[2],input_data_shape[3]] # ?????? 2 128 128 #batch size

        #target_shape = [input_data_shape[0], 128,128,128] # ?????? 2 128 128 #batch size
        print("target shape:", target_shape) #2 128 128 
        
        #resized_input = resize(input_data, new_shape=target_shape)
        #resized_output = resize(output_data, new_shape=target_shape)
        
        #X_train = tf.expand_dims(resized_input, 4)
        #y_train = tf.expand_dims(resized_output, 4)
        X_train = tf.expand_dims(input_data_sample, 4)
        y_train = tf.expand_dims(output_data_sample, 4)

        yield (X_train, y_train)
        #Return sends a specified value back to its caller whereas Yield can produce a sequence of values.
        #We should use yield when we want to iterate over a sequence,
        #but don’t want to store the entire sequence in memory. 
        #Yield is used in Python generators. 
        #A generator function is defined just like a normal function,
        #but whenever it needs to generate a value, it does so with the yield keyword rather than return. 
        #If the body of a def contains yield, the function automatically becomes a generator function. 

checkpoint_path1 = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir1 = os.path.dirname(checkpoint_path1)

cp_callback = ModelCheckpoint(checkpoint_path1,
                                save_weights_only = True,
                                save_freq = 10,
                                verbose = 1)

steps_p_e = int(len(tfrecord_files_train)//batch_size)
print("steps_p_epoch", steps_p_e)

dataset_train_np = tfrecord_dataset_train.as_numpy_iterator()

model_train = model.fit(data_generator(train_data=dataset_train_np),
                        steps_per_epoch=steps_p_e, epochs=num_epochs, 
                        shuffle=True, callbacks = [cp_callback] )

loss_history1 = model_train.history['loss']

with open('loss_history1.pickle', 'wb') as f:
    pickle.dump([loss_history1, num_epochs], f)

# if the demo_folder directory is not present then create it. 
if not os.path.exists("models"): 
    os.makedirs("models") 
    
model_name = "models/model_" + str(num_epochs) +"epochs_" + "batchsize"+ str(batch_size) + "_trnfolder_" + folder_trn+".h5"
#model.save(model_name)
save_model(model, model_name)
########################################################################
### Load latest checkpoints from training
########################################################################

checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

input_shape = (None, None, None, 1)
input_tensor = Input(shape= input_shape, name="input")
ushape1 = build_CNN(input_tensor)
ushape2 = build_CNN(ushape1)
model = Model(input_tensor, ushape2)

model.compile(optimizer='adam', loss='mean_absolute_error')

latest_weights = tf.train.latest_checkpoint(checkpoint_dir)
print(latest_weights)
model.load_weights(latest_weights)


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

#x_axis = np.linspace(1,num_epochs,num_epochs)
x_axis = np.linspace(1,num_epochs,num_epochs)
plt.plot(x_axis, loss_history1)
plt.ylabel('Loss')
plt.xlabel('Epochs')
#plt.yscale('log')
plt.ylim(0, 0.25) 
plt.xlim(0, num_epochs) 
plt.title("Loss history")
plt.ticklabel_format(style='plain')




loss_title ="images/loss_batch" + str(batch_size) + "_" + str(num_epochs) + "epochs_" + folder_trn +"_"+ folder_test 
plt.savefig(loss_title)
plt.show()

################################################################
## Visualize data
################################################################
# def visualize(resized_input, para):
#   data_shape=list(resized_input.shape)
#   slice_x = resized_input[data_shape[0]//2, :, :]
#   slice_y = resized_input[:, data_shape[1]//2, :]
#   slice_z = resized_input[:, :, data_shape[2]//2]
  
#   print("data shape", data_shape)
#   print("global max", tf.reduce_max(resized_input))
#   print("global min", tf.reduce_min(resized_input))

#   global_min = tf.reduce_min(resized_input)
#   global_max = tf.reduce_max(resized_input)
#   print("dimensions of input data slice y", tf.shape(slice_y))
#   print("max x", tf.reduce_max(slice_x).numpy())
#   print("min x", tf.reduce_min(slice_x).numpy())
#   print("max y",tf.reduce_max(slice_y).numpy())
#   print("min y",tf.reduce_min(slice_y).numpy())
#   print("max z", tf.reduce_max(slice_z).numpy())
#   print("min z", tf.reduce_min(slice_z).numpy())
  
#   fig = plt.figure(figsize=(10, 6))
#   grid = ImageGrid(fig, 111,
#                 nrows_ncols = (1,3),
#                 axes_pad = 0.05,
#                 cbar_location = "right",
#                 cbar_mode="single",
#                 cbar_size="5%",
#                 cbar_pad=0.05
#                 )

#   if para=="pred":
           
#       grid[0].imshow(slice_x, cmap='gray',aspect='equal', vmin=global_min, vmax=global_max)
#       grid[0].axis('off')
#       grid[0].set_title("predicted data X-dim (32x32)")
      
#       grid[1].imshow(slice_y, cmap='gray',aspect='equal', vmin=global_min, vmax=global_max)
#       grid[1].axis('off')
#       grid[1].set_title("predicted data Y-dim (32x32)")
    
#       imc =grid[2].imshow(slice_z, cmap='gray',aspect='equal', vmin=global_min, vmax=global_max)
#       grid[2].axis('off')
#       grid[2].set_title("predicted data z-dim (32x32)")

#       plt.colorbar(imc, cax=grid.cbar_axes[0])
#       plt.show()

#   elif para=="ref":
      
#       grid[0].imshow(slice_x, cmap='gray',aspect='equal', vmin=global_min, vmax=global_max)
#       grid[0].axis('off')
#       grid[0].set_title("Reference data X-dim (32x32)")
      
#       grid[1].imshow(slice_y, cmap='gray',aspect='equal', vmin=global_min, vmax=global_max)
#       grid[1].axis('off')
#       grid[1].set_title("Reference data Y-dim (32x32)")
    
#       imc =grid[2].imshow(slice_z, cmap='gray',aspect='equal', vmin=global_min, vmax=global_max)
#       grid[2].axis('off')
#       grid[2].set_title("Reference data z-dim (32x32)")

#       plt.colorbar(imc, cax=grid.cbar_axes[0])
#       plt.show()
      


#   elif para=="input":
#       grid[0].imshow(slice_x, cmap='gray',aspect='equal', vmin=global_min, vmax=global_max)
#       grid[0].axis('off')
#       grid[0].set_title("Input data X-dim (32x32)")
      
#       grid[1].imshow(slice_y, cmap='gray',aspect='equal', vmin=global_min, vmax=global_max)
#       grid[1].axis('off')
#       grid[1].set_title("Input data Y-dim (32x32)")
    
#       imc =grid[2].imshow(slice_z, cmap='gray',aspect='equal', vmin=global_min, vmax=global_max)
#       grid[2].axis('off')
#       grid[2].set_title("Input data z-dim (32x32)")

#       plt.colorbar(imc, cax=grid.cbar_axes[0])
#       plt.show()

#   else:
#     raise ValueError("Wrong argument for para")
  
# ####################################################################################################
# ###################################################################################################
# ########################################## VISUALIYE NEW ###############################################################
# ########################################################################################################


# def resize_tst(array, new_shape):
#     old_shape = array.shape
#     x_ratio = old_shape[0] // new_shape[0]
#     y_ratio = old_shape[1] // new_shape[1]
#     z_ratio = old_shape[2] // new_shape[2]

#     resized_array = np.zeros(new_shape, dtype=array.dtype)

#     for i in range(new_shape[0]):
#         for j in range(new_shape[1]):
#             for k in range(new_shape[2]):
#                 # Calculate the coordinates
#                 x = i * x_ratio
#                 y = j * y_ratio
#                 z = k * z_ratio
#                 # Take the value from the original array
#                 resized_array[i, j, k] = array[x, y, z]

#     return resized_array


######################################################
########## Data for training
######################################################
# def resize(array, new_shape):
#     old_shape = array.shape
#     x_ratio = old_shape[1] // new_shape[1]
#     y_ratio = old_shape[2] // new_shape[2]
#     z_ratio = old_shape[3] // new_shape[3]

#     resized_array = np.zeros(new_shape, dtype=array.dtype)

#     for i in range(new_shape[1]):
#         for j in range(new_shape[2]):
#             for k in range(new_shape[3]):
#                 # Calculate the coordinates
#                 x = i * x_ratio
#                 y = j * y_ratio
#                 z = k * z_ratio
#                 # Take the value from the original array
#                 resized_array[:, i, j, k] = array[:, x, y, z]

#     return resized_array