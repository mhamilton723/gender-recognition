# Deep Learning for Image Classification

In this repository you will find some simple code for how to train your own deep image classifier. In our example we will train a network to discern gender from the [Labelled Faces in the Wild dataset](http://vis-www.cs.umass.edu/lfw/).

The goal of this code is to provide clear examples of how one can create and visualize a complex network with only a few lines of code. This project is intended for academic use.

## How to Use the Command Line

### Mac
Type "terminal" in the finder and you should be able to pop up a window. See [Beginner Guide To Terminal Usage](https://lifehacker.com/5633909/who-needs-a-mouse-learn-to-use-the-command-line-for-almost-anything) for more details.

### Windows

Unfortunately you'll have to deal with powershell. If you are feeling ambitious, you can create a linux virtual machine using [this guide](https://www.windowscentral.com/how-run-linux-distros-windows-10-using-hyper-v) If virtual machines sound like too much, just type "powershell" into the search bar and you should boot up a terminal. See [this very small guide](https://wiki.communitydata.cc/Windows_terminal_navigation) for some basic usage.

### Linux
If you are using ubuntu, search for the terminal app just like on a mac. See the mac guide for usage tips.

## Getting Started

### 1. Check your python Version

To get started, first make sure that you are working you have a copy of python >= 3.5

Typing:
```bash
python
```
will yield something like:
```bash
Python 3.5.4 |Continuum Analytics, Inc.| (default, Aug 14 2017, 13:41:13) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

Which will show you the version. Notice the `>>>` prompt which tells you you are typing python commands in something called a REPL. To exit the repl and get back to the terminal type `quit()`. To those who want to use python 2.7, just dont do it. You are on the wrong side of history and need to switch over to 3.5.

If you have an older version of python. Go grab yourself [Anaconda 3.6](https://www.anaconda.com/download/) and repreat the above version check. NOTE:

**After install you need to close your terminal and re-open it to make sure it picks up on the newly installed python**

### 2. Download the Project

To grab a copy of this project fire up the terminal, navigate (within the terminal) to a location that you would like to hold this project (like the Documents folder or something therein)

You can check that you are in your location of choice by using the `pwd` command (**P**resent **W**orking **D**irectory)

Now you can "clone" the project using git. 

```bash
git clone https://github.com/mhamilton723/gender-recognition.git
```

This will pull the source code onto your computer. If you do not have git, install it by following [this guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 

You can now navigate to the main code of the repositoty by navigating to the **s**ou**rc**e code folder.  

```bash
cd gender-recognition/src
```

### 3. Install Dependencies

To install the dependencies for the project you must first decide whether you would like to use the GPU enabled version, or the CPU-only version. GPU support dramatically speeds training time (10x - 50x), but is only available if you have a GPU and download special software from NVIDIA. If you are feeling ambitious you can follow this [guide to set up tensorflow with GPU support](https://www.tensorflow.org/install/). **IF AND ONLY IF** you have installed GPU enabled tensorflow run the following in the `gender-recognition` directory

```bash
pip install -r ./requirements-gpu.txt
```

If you opted for the CPU option run
```bash
pip install -r ./requirements.txt
```


### 4. Run the Code
 Make sure your terminal is inside of the source folder. `pwd` Should yield something akin to (on windows)
 
 ```bash
 C:\Users\You\YourFantasticLocation\gender-recognition\src
 ```

Now you can run:
```bash
python train_network.py

```

This script will automatically
 1) Download and unpack the training data (which will take a few minutes depending on your internet speed)
 2) Build and Compile the deep network
 3) Train the network on the data
 4) Log interesting data, checkpoint, and save the model

If you see output that looks something like this :

```bash
Using TensorFlow backend.
Found data, skipping download
Found 11910 images belonging to 2 classes.
Found 662 images belonging to 2 classes.
2018-03-26 21:49:12.876116: I C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
Epoch 1/1000
20/20 [==============================] - 112s 6s/step - loss: 0.5730 - acc: 0.7347 - val_loss: 0.5237 - val_acc: 0.7717
Epoch 2/1000
20/20 [==============================] - 108s 5s/step - loss: 0.5195 - acc: 0.7801 - val_loss: 0.4972 - val_acc: 0.7650
Epoch 3/1000
20/20 [==============================] - 104s 5s/step - loss: 0.4387 - acc: 0.8016 - val_loss: 0.4084 - val_acc: 0.7996
Epoch 4/1000
20/20 [==============================] - 105s 5s/step - loss: 0.4235 - acc: 0.8023 - val_loss: 0.3940 - val_acc: 0.8322
Epoch 5/1000
20/20 [==============================] - 104s 5s/step - loss: 0.4158 - acc: 0.8105 - val_loss: 0.4140 - val_acc: 0.8137
Epoch 6/1000
.....
```

Then do a little celebration dance and sit back and wait. Deep learning is cool, but definitely not fast. Using a GPU will speed up computation by a factor of 10x-50x.

### 5. Visualizing the model with Tensorboard

Once you have kicked off the model training pop open a new terminal window and head into the gender-recognition directory (One above the src directory)

You can now monitor your running network with the [Tensorboard visualization tool](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard). Simply execute the following:

```bash
tensorboard --logdir logs
```

This will point tensorboard at the "logs" directory in your project. This will output something like the following:

```bash
TensorBoard 0.4.0 at http://5147322-0829:6006 (Press CTRL+C to quit)
```

Now copy paste the url, in this case `http://5147322-0829:6006` and check out your model in action. 


## Downloading Trained Models and Logs

If you cannot get the above working follow these steps to download copies of the tensorboard logs and the final trained model

## Explore 

This is a simple network, feel free to
   - Explore the hyper parameter space
   - Add more dropout
   - Change the number and size of layers
   - Add regularization
   - Change the activation function (maybe try SELU)
   - Change the dataset to your favorite classification problem ([These are fun](https://github.com/junyanz/CycleGAN#datasets))
   - Go wild

## Contributing

This is an open source project and contributions are welcome. To contribute, fork this project and take out a PR. If you used this project or found it helpful, drop me an email and let me know!