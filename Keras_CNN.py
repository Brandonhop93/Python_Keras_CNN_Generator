###########################
# Module imports.
###########################
from keras.preprocessing.image import ImageDataGenerator, image, load_img, img_to_array
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam, Adadelta, Adagrad, RMSprop, SGD
from keras import backend, callbacks
import matplotlib.pyplot as plt
import csv
###########################



###########################
# Params
###########################
""""""""" TRAINING SETTINGS """""""""
# EPOCH: Iteration over the whole training set, made up of many batches.
EPOCHS = 30

# BATCH_SIZE: Number of forward passed done before an optimization step.
# A higher batch size means more memory utilisation and less (but better) optimization steps are done in each epoch.
BATCH_SIZE = 32

# LEARNING_RATE: Optimization step size.
LEARNING_RATE = 0.001

# Save best epoch based on validation error.
SAVE_BEST_MODEL = True

# Early stopping based on validation error.
EARLY_STOP = True

# Augment images, shuffle images, adaptive learning rate, dropout.
VALIDATION_EXTRAS = True
""""""""""""""""""""""""""""""""""""



""""""""" MODEL SETTINGS """""""""
# Convolution layers in network (includes max pooling).
# E.g. (1, 2, 3, 4, 5)
CONVOLUTION_COUNT = 3

# Filter count per layer.
# E.g. (32, 64, 96, 128)
FILTER_COUNT = [64, 64, 64, 96, 256]

# Filter size squared.
# E.g. (3, 5, 7)
FILTER_SIZE = 3

# Pool size squared.
# E.g. (2, 3, 4, 5)
POOL_SIZE = 3

# Fully connected layers at the end of the network
# E.g. (1, 2, 3)
FULLY_CONNECTED_COUNT = 1

# Fully connected layer sizes
# E.g. (256, 512, 1024)
FULLY_CONNECTED_SIZE = [512, 256, 256]

# Network optimiser.
# E.g. (Adam, Adadelta, Adagrad, RMSprop, SGD)
OPTIMISER = Adam

# Layer activation.
# E.g. (relu, selu, elu, sigmoid, tanh, etc)
ACTIVATION = "relu"
""""""""""""""""""""""""""""""



""""""""" PRINTOUT SETTINGS """""""""
TRAINING_PRINT_MODE = 2  # 0:silent, 1:animated, 2:numeric.
PRINT_MODEL_SUMMARY = True
PRINT_ACCURACY_GRAPH = True
PRINT_ERROR_GRAPH = True
""""""""""""""""""""""""""""""



""""""""" IMAGE SETTINGS """""""""
# Squared image size.
IMG_SIZE = 128

# Training data: 80%
T_SAMPLE_DIR = 'data/train'
T_SAMPLE_CNT = 3456

# Validation data: 20%
V_SAMPLE_DIR = 'data/valid'
V_SAMPLE_CNT = 866
""""""""""""""""""""""""""""""



""""""""" PRE-PROCESSING """""""""
# Validation: Augments all images within a training batch.
AUGMENT_IMAGES = False
# Validation: Shuffle training order.
SHUFFLE_IMAGES = False
# Validation: Reduce learning rate if stall.
REDUCE_LEARNING_RATE = False
# Validation: Dropout
DROPOUT = False

if VALIDATION_EXTRAS is True:
    AUGMENT_IMAGES = True
    SHUFFLE_IMAGES = True
    REDUCE_LEARNING_RATE = True
    DROPOUT = True

# Configure Keras input.csv data format
if backend.image_data_format() == 'channels_first':
    input_shape = (3, IMG_SIZE, IMG_SIZE)
else:
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
""""""""""""""""""""""""""""""
###########################



###########################
# Dataset Augmentation
###########################
# Test dataset augmentation (augments every new batch of data).
# Hence, every optimisation step will be dynamic.

if AUGMENT_IMAGES:
    training_augmentor = ImageDataGenerator(
        # Normalise image pixel values from 0-255 to 0-1.
        rescale = 1. / 255,
        # # Randomly flip image horizontally.
        # horizontal_flip = True,
        # # Randomly flip image vertically.
        # vertical_flip = True,
        # Randomly rotate image +- 180 degrees.
        rotation_range = 180,
        # Randomly shift image vertically by +- 20%.
        height_shift_range = 0.2,
        # Randomly shift image horizontally by +- 20%.
        width_shift_range = 0.2,
        # Randomly shear/warp image by +- 25 degrees.
        shear_range = 25,
        # Randomly zoom image by +- 10%
        zoom_range = 0.1,
        # Randomly adjust brightness +- 15%.
        brightness_range = [0.85, 1.15],
        # Randomly adjust colour channels by += 20%.
        # channel_shift_range = 51,  # (255 * 0.20 = 51)
        # Blank space fill mode
        fill_mode = 'reflect')
else:
    # Training dataset augmentation.
    training_augmentor = ImageDataGenerator(
        # Normalise image pixel values from 0-255 to 0-1.
        rescale= 1. / 255)

# Validation dataset augmentation.
validation_augmentor = ImageDataGenerator(
    # Normalise image pixel values from 0-255 to 0-1.
    rescale = 1. / 255)


# Training dataset generator.
training_generator = training_augmentor.flow_from_directory(
    # Training dataset directory.
    T_SAMPLE_DIR,
    # Resize image.
    target_size = (IMG_SIZE, IMG_SIZE),
    # Number of images supplied in a generation.
    batch_size = BATCH_SIZE,
    # Assign images with categories.
    class_mode = 'categorical')

# Validation dataset generator.
validation_generator = validation_augmentor.flow_from_directory(
    # Validation dataset directory.
    V_SAMPLE_DIR,
    # Resize image.
    target_size = (IMG_SIZE, IMG_SIZE),
    # Number of images supplied in a generation.
    batch_size = BATCH_SIZE,
    # Assign images with categories.
    class_mode = 'categorical')
###########################



###########################
# Keras CNN Model
###########################
model = Sequential()

model.add(Conv2D(FILTER_COUNT[0], (FILTER_SIZE, FILTER_SIZE), padding = 'Same', input_shape = input_shape))
model.add(Activation(ACTIVATION))
model.add(MaxPooling2D(pool_size = (POOL_SIZE, POOL_SIZE)))
if DROPOUT:
    model.add(Dropout(0.2))

for i in range(CONVOLUTION_COUNT - 1):
    model.add(Conv2D(FILTER_COUNT[i + 1], (FILTER_SIZE, FILTER_SIZE), padding = 'Same'))
    model.add(Activation(ACTIVATION))
    model.add(MaxPooling2D(pool_size = (POOL_SIZE, POOL_SIZE), strides = (2, 2)))
    if DROPOUT:
        model.add(Dropout(0.2))

model.add(Flatten())

for i in range(FULLY_CONNECTED_COUNT):
    model.add(Dense(FULLY_CONNECTED_SIZE[i]))
    model.add(Activation(ACTIVATION))
    if DROPOUT:
        model.add(Dropout(0.2))

model.add(Dense(5))
model.add(Activation('softmax'))

# Compile Keras model.
model.compile(loss = 'categorical_crossentropy', optimizer = OPTIMISER(lr = LEARNING_RATE), metrics = ['accuracy'])

if PRINT_MODEL_SUMMARY:
    model.summary()
###########################



###########################
# Keras Callbacks
###########################
training_callbacks = []

if SAVE_BEST_MODEL:
    # Remove . from learning rate.
    learning_rate_string = str(LEARNING_RATE).replace('.', '_', 1)

    # Save best model to file based on monitored value.
    training_callbacks.append(
        callbacks.ModelCheckpoint(
            filepath = 'models/Epc' + str(EPOCHS) + '_Btc' + str(BATCH_SIZE) + '_Lr' + str(learning_rate_string) + '_Vld' +
            str(VALIDATION_EXTRAS) + '_Con' + str(CONVOLUTION_COUNT) + '_Fil' + str(FILTER_COUNT[0]) + '_Siz' +
            str(FILTER_SIZE) + '_Pol' + str(POOL_SIZE) + '_Ful' + str(FULLY_CONNECTED_COUNT) + '_FCS' + str(FULLY_CONNECTED_SIZE[0]) + '_Opp' +
            str(OPTIMISER.__name__) + '.h5',
            # Printout mode. (0, 1)
            verbose = 0,
            # Value to monitor while training. (loss, val_loss, accuracy, val_accuracy)
            monitor = 'val_loss',
            # Save only the best epoch weights.
            save_best_only = True,
            # Minimise or maximise monitored value. (min, max, auto)
            mode = 'auto'))

if REDUCE_LEARNING_RATE:
    # Reduce training learning rate if monitored value plateaus.
    training_callbacks.append(
        callbacks.ReduceLROnPlateau(
            # Value to monitor while training. (loss, val_loss, accuracy, val_accuracy)
            monitor = 'val_loss',
            # Maximum number of epochs to wait for improvement.
            patience = 3,
            # Printout mode. (0, 1)
            verbose = 1,
            # Factor by which the learning rate will be reduced.
            factor = 0.1,
            # Minimum learning rate.
            min_lr = 0.0001,
            # Minimise or maximise monitored value. (min, max, auto)
            mode = 'auto'))

if EARLY_STOP:
    # Stop training if monitored value doesn't change by threshold.
    training_callbacks.append(
        callbacks.EarlyStopping(
            # Value to monitor while training. (loss, val_loss, accuracy, val_accuracy)
            monitor = "val_loss",
            # Maximum number of epochs to wait for improvement.
            patience = 6,
            # Minimum change monitored value to qualify as an improvement (2.5%).
            min_delta = 0.025,
            # Minimise or maximise monitored value. (min, max, auto)
            mode = "auto",
            # Printout mode. (0, 1)
            verbose = 1,
            # Training will stop if the model doesn't show improvement over the baseline.
            baseline = None,
            # Restore model weights from the epoch with the best value of the monitored quantity.
            restore_best_weights = True
        ))
###########################



###########################
# Keras Training
###########################
print('\nTraining: Started\n')

history = model.fit_generator(
    # Training data generator.
    generator = training_generator,
    # Total number of training epochs.
    epochs = EPOCHS,
    # Number of training generations required.
    steps_per_epoch = (T_SAMPLE_CNT // BATCH_SIZE),  # // is floor division
    # Validation data generator.
    validation_data = validation_generator,
    # Number of validation generations required.
    validation_steps = (V_SAMPLE_CNT // BATCH_SIZE),
    # Show training progress: 0:silent, 1:animated, 2:numeric.
    verbose = TRAINING_PRINT_MODE,
    # Shuffle training data order before each epoch.
    shuffle = SHUFFLE_IMAGES,
    # Callback function ran on epoch.
    callbacks = training_callbacks)

print('\nTraining: Finished\n')
###########################



###########################
# Display Results
###########################
# Training and validation data over all epochs.
training_error = history.history['loss']
training_accuracy = history.history['accuracy']
validation_error = history.history['val_loss']
validation_accuracy = history.history['val_accuracy']

# [T Error, T Acc, V Error, V Acc]
best_epoch_values = [training_error[0], training_accuracy[0], validation_error[0], validation_accuracy[0]]
best_epoch = 0

# Get best epoch based on minimum validation error.
for i in range(len(validation_error)):
    if validation_error[i] <= best_epoch_values[2]:
        best_epoch = i
        best_epoch_values[0] = training_error[i]
        best_epoch_values[1] = training_accuracy[i]
        best_epoch_values[2] = validation_error[i]
        best_epoch_values[3] = validation_accuracy[i]

# Write to .csv file.
with open('output/trainingData.csv', 'a', newline = '') as file:
    fileWriter = csv.writer(file)

    # CSV FORMAT:
    # [T_ERR, T_ACC, V_ERR, V_ACC, EPC, BST_EPC, BTC, LR, CON_L, FIL_C, FIL_S, POL_S, FC_C, FC_S, OPP, USE_V]
    fileWriter.writerow(
        ['{:.4f}'.format(best_epoch_values[0]),
         '{:.2f}'.format(best_epoch_values[1]),
         '{:.4f}'.format(best_epoch_values[2]),
         '{:.2f}'.format(best_epoch_values[3]),
         len(training_error),
         best_epoch,
         BATCH_SIZE,
         LEARNING_RATE,
         CONVOLUTION_COUNT,
         str(FILTER_COUNT),
         FILTER_SIZE,
         POOL_SIZE,
         FULLY_CONNECTED_COUNT,
         str(FULLY_CONNECTED_SIZE),
         str(OPTIMISER.__name__),
         str(VALIDATION_EXTRAS)])


# Print spreadsheet data.
print('\n-------------------------------------------------------')
print(f'FINAL DATA:')
print(f'Epochs: {len(training_error)} Best Epoch: {best_epoch}  Batch Size: {BATCH_SIZE}   Learn Rate: {LEARNING_RATE}')
print(f'ConV Layers: {CONVOLUTION_COUNT}  Filter Count: {FILTER_COUNT}  Filter Size: {FILTER_SIZE}')
print(f'Pool Size: {POOL_SIZE}  Full Layers: {FULLY_CONNECTED_COUNT}  Full Size: {FULLY_CONNECTED_SIZE}')
print(f'Optimiser: {str(OPTIMISER.__name__)}  Use Extras: {str(VALIDATION_EXTRAS)}')
print('-------------------------------------------------------')
print('    Training Error       :   ', end = '')
print('{:.4f}'.format(best_epoch_values[0]))
print('    Training Accuracy    :   ', end = '')
print('{:.2f}'.format(best_epoch_values[1]))
print('    Validation Error     :   ', end = '')
print('{:.4f}'.format(best_epoch_values[2]))
print('    Validation Accuracy  :   ', end = '')
print('{:.2f}'.format(best_epoch_values[3]))
print('-------------------------------------------------------')


if PRINT_ACCURACY_GRAPH:
    # Training accuracy and validation accuracy graph.
    plt.rcParams['figure.figsize'] = (6, 5)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc = 'upper left')
    plt.show()
    plt.close()

if PRINT_ERROR_GRAPH:
    # Training error and validation error graph.
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Error")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc = 'upper right')
    plt.show()
    plt.close()
###########################
