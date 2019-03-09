# DigitRecognition
Implementation of Neural Networks for classification from Scratch.

I have added modules to work with images and the MNIST Database of
handwritten digits (see [here](http://yann.lecun.com/exdb/mnist/)).

Neural Networks can be trained, saved and loaded. For instance,
the Neural Network described in the plain ASCII file MNistNN93-56.txt
has one hidden layer of 28 nodes and
was trained (10000 iterations) with the MNist Database of 60000 training examples and
has an accuracy of 93.56% on the test set (of 1000 images). Each
picture was first reduced from its original size (28x28) to 10x10
(thus 100 input nodes, without counting the bias unit) in order
to speed up the training significantly.


## Note

I will soon add modules to read and write images from common file
types. For instances the ones in the folder ''pgm''.
