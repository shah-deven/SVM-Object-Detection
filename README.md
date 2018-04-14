# SVM-Object-Detection
Train a SVM and for detecting human upper bodies in TV series The Big Bang Theory.

## Data
The training data is typically a set of images with bounding boxes of the upper bodies. Positive training examples are image patches extracted at the annotated locations. A negative training example can be any image patch that does not significantly overlap with the annotated upper bodies. Thus there potentially many more negative training examples than positive training examples. 

Due to memory limitation, it will not be possible to use all negative training examples at the same time. Hence, implemented hard-negative mining to find hardest negative examples and iteratively train an SVM.
