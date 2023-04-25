# Image-Classification-CNN
In the code, we are using two models for image classification:

1. A CNN (Convolutional Neural Network) model: This is a custom-built model that 
consists of several convolutional and pooling layers, followed by two fully connected 
layers. The model is trained from scratch on the dataset of images with multiple classes.
The CNN model has the following convolutional and pooling layers:
• Conv2D layer with 32 filters and a kernel size of 3x3
• MaxPooling2D layer with a pool size of 2x2
• Conv2D layer with 64 filters and a kernel size of 3x3
• MaxPooling2D layer with a pool size of 2x2
• Conv2D layer with 128 filters and a kernel size of 3x3
• MaxPooling2D layer with a pool size of 2x2
These layers are responsible for extracting features from the input images.

2. A VGG16 model with transfer learning: This is a pre-trained model that has already been 
trained on a large dataset of images (ImageNet) and is known to perform well on image 
classification tasks. We use the pre-trained weights of this model as initial weights for 
our own model, which consists of the VGG16 model followed by a few additional layers 
for classification. This allows us to take advantage of the powerful feature extraction 
capabilities of the VGG16 model, while still fine-tuning the model for our specific task.
The VGG16 model has the following convolutional and pooling layers:
• Conv2D layer with 64 filters and a kernel size of 3x3
• Conv2D layer with 64 filters and a kernel size of 3x3
• MaxPooling2D layer with a pool size of 2x2
• Conv2D layer with 128 filters and a kernel size of 3x3
• Conv2D layer with 128 filters and a kernel size of 3x3
• MaxPooling2D layer with a pool size of 2x2
• Conv2D layer with 256 filters and a kernel size of 3x3
• Conv2D layer with 256 filters and a kernel size of 3x3
• Conv2D layer with 256 filters and a kernel size of 3x3
• MaxPooling2D layer with a pool size of 2x2
• Conv2D layer with 512 filters and a kernel size of 3x3
• Conv2D layer with 512 filters and a kernel size of 3x3
• Conv2D layer with 512 filters and a kernel size of 3x3
• MaxPooling2D layer with a pool size of 2x2
• Conv2D layer with 512 filters and a kernel size of 3x3
• Conv2D layer with 512 filters and a kernel size of 3x3
• Conv2D layer with 512 filters and a kernel size of 3x3
• MaxPooling2D layer with a pool size of 2x2
These layers are also responsible for extracting features from the input images. The 
VGG16 model has a much deeper architecture than the CNN model and uses smaller 
filter sizes with more filters in each layer, which allows it to learn more complex and 
abstract features from the input images.

3. Data Loading
In the train_generator and test_generator are created using the flow_from_directory 
method from ImageDataGenerator class.
When creating these generators, we pass the directory paths (train_dir and test_dir) as 
arguments. The flow_from_directory method then automatically infers the labels for each 
image based on the directory structure. Specifically, it assumes that each subdirectory 
within the directory path corresponds to a different class, and that the images within that 
subdirectory all belong to that class.
For example, if the directory structure of train_dir is as follows:
train_dir/
 class1/
 image1.jpg
 image2.jpg
 ...
 class2/
 image1.jpg
 image2.jpg
 ...
 class3/
 image1.jpg
 image2.jpg
 ...
 ...
Conclusion
Both models are trained and evaluated on the same dataset of images with multiple classes, using 
image augmentation and normalization techniques to improve performance. The performance of 
the two models is then compared to determine which one performs better on the task of image 
classification
