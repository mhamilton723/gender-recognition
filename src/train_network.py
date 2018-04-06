import sys
sys.path.append("../")
from os.path import join
import keras
from keras import backend as K
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from src.visualization import CustomTensorboard, convKernelSummary, convActivationSummary
from src.get_data import install_data_if_not_exists, make_if_not_exisits

# this will automatically install the data if none exists,
# if you need to re-install this data delete the data folder
install_data_if_not_exists()

# Tunable Parameters
img_width, img_height = 128, 128          # What dimensions to resize the input images to
name = "run_2"                            # What to name the checkpoints and logs
batch_size = 128                          # The size of the minibatches
nb_train_samples = batch_size * 20        # The number of training samples per epoch
nb_validation_samples = batch_size * 10   # The number of validation samples per epoch
epochs = 1000                             # The number of epochs
reg = keras.regularizers.l2(0.0)          # What kind of regularization we use (0.0 means none)
learning_rate = 0.001                     # The (initial) learning rate

# Location of directories
top_dir = "../"
checkpoint_dir = join(top_dir, "checkpoints")
make_if_not_exisits(checkpoint_dir)

log_dir = join(top_dir, "logs", name)
make_if_not_exisits(join(top_dir, "logs"))

saved_model_dir = join(top_dir, "saved_models")
make_if_not_exisits(saved_model_dir)

data_dir = join(top_dir, "data", "clean_lfw")
train_data_dir = join(data_dir, 'train')
validation_data_dir = join(data_dir, 'val')

print("Building Model")
# Useful if one uses a different backend (like CNTK or Theano)
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Start building our model
model = Sequential()

model.add(Conv2D(8, (10, 10), use_bias=False, input_shape=input_shape, kernel_regularizer=reg))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (10, 10), kernel_regularizer=reg))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (10, 10), kernel_regularizer=reg))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(1))
model.add(Activation('sigmoid'))

# Specification of our optimizer, this one is a industry favorite
optimizer = keras.optimizers.Adam(lr=learning_rate)

# Compile the model for training
print("Compiling Model")
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Configure an Image Data reader for training
# this augments the dataset for extra robustness
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=10,
    width_shift_range=.1,
    height_shift_range=.1,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# We use this custom tensorboard visualizer because
# Keras doesent provide us with all of the bells and whistles
# that we want
tb_viz = CustomTensorboard(log_dir=log_dir, validation_generator=validation_generator)\
    .add_summary(convKernelSummary)\
    .add_summary(convActivationSummary)

# We add checkpointing in case we need to restart from a checkpoint
checkpoint = ModelCheckpoint(join(checkpoint_dir, name), period=10)

print("Training model")
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[tb_viz, checkpoint],
    class_weight={0: 3.6, 1: 1.0}
    # There are roughly 3.6x as many male faces in the dataset
    # so up-weight the female faces (class 0)
    # This will cut down on some of the network's
    # bias towards predicting males.
)

print("Saving trained model weights")
model.save(join(saved_model_dir, "{}.h5".format(name)))
