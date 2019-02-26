# DigitRecognition
Implementation of Neural Networks for classification from Scratch.

I have added modules to work with images and the MNIST Database of
handwritten digits (see [here](http://yann.lecun.com/exdb/mnist/)).

Neural Networks can be trained, saved and loaded. For instance,
the Neural Network described in the plain ASCII file MNistNN867.txt
has one hidden layer of 28 nodes and
was trained with the MNist Database of 60000 training examples. Each
picture was reduced from its original size (28x28) to 10x10
(thus 100 input nodes, without counting the bias unit) in order
to speed up the training significantly.


##Note

I will soon add modules to read and write images from common file
types. For instances the ones in the folder ''pgm''.
