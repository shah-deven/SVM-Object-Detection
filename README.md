# SVM-Object-Detection
Train a SVM and for detecting human upper bodies in TV series The Big Bang Theory.

## Data
The training data is typically a set of images with bounding boxes of the upper bodies. Positive training examples are image patches extracted at the annotated locations. A negative training example can be any image patch that does not significantly overlap with the annotated upper bodies. Thus there potentially many more negative training examples than positive training examples. 

Due to memory limitation, it will not be possible to use all negative training examples at the same time. Hence, implemented hard-negative mining to find hardest negative examples and iteratively train an SVM.

Training images are provided in the subdirectory trainIms. The annotated locations of the upper bodies are given in trainAnno.mat. This file contains a cell structure ubAnno; ubAnno{i} is the annotated locations of the upper bodies in the i th image. ubAnno{i} is 4 × k matrix, where each column corresponds to an upper body. The rows encode the left, top, right, bottom coordinates of the upper bodies (the origin of the image coordinate is at the top left corner).

Images for validation and test are given in valIms, testIms respectively. The annotation file for test images is not released. We have also extracted some image regions of test images, and the regions are saved as 64×64 jpeg images in testRegs. Only small portion of these images correspond to upper bodies.

## External library
Raw image intensity values are not robust features for classification. Used Histogram of Oriented Gradient (HOG) as image features. HOG uses the gradient information instead of intensities, and this is more robust to changes in color and illumination conditions. 
To use HOG, install an VL FEAT: http://www.vlfeat.org. 
