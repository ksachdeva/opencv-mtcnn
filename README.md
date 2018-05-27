# opencv-mtcnn

This is an inference implementation of MTCNN (Multi-task Cascaded Convolutional Network) to perform Face Detection and Alignment using OpenCV's DNN module.

## MTCNN

[ZHANG2016] Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499–1503.

https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf

## OpenCV's DNN module

Since OpenCV 3.1 there is a module called DNN that provides the inference support. The module is capable of taking models & weights from various popular frameworks such as Caffe, tensorflow, darknet etc.

You can read more about it here - https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV

Note that at present there is no support to perform training in OpenCV's DNN module and if I understood correctly there is no intention either.

## Compile / Run

### Requirements

* OpenCV 3.4+
* Boost FileSystem (1.58+)  [only required for the sample application]
* CMake 3.2+

I am using CMake as the build tool. Here are the steps to try the implementation -

```bash
# compiling the library and the sample application
git clone https://github.com/ksachdeva/opencv-mtcnn
cd opencv-mtcnn
mkdir build
cd build
cmake ..
cmake --build .
```

```bash
# running the sample application
cd build
./sample/app <path_to_models_dir> <path_to_test_image>

# here is an example cmd line to run with the model and image in the test repository
./sample/app ../data/models ../data/Aaron_Peirsol_0003.jpg
```

## Acknowledgments

Most of the implementations of MTCNN are based on either Caffe or Tensorflow. I wanted to play with OpenCV's DNN implementation and understand the paper bit better. While implementing it, I did look at various other C++ implementations (again all of them use Caffe) and more specifically borrowed utilities from https://github.com/golunovas/mtcnn-cpp. IMHO, I found his implementation (in C++) that is based on Caffe to be the cleanest amongst many others.

The model files are taken from https://github.com/kpzhang93/MTCNN_face_detection_alignment/tree/master/code

The image file "Aaron_Peirsol_0003.jpg" is from the LFW database (http://vis-www.cs.umass.edu/lfw/)
