[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)




## Deep Learning Project ##

In this project, I trained a deep neural network to track a target in simulation. The goal was to achieve a score of
0.4 using the IOU metric. I was ultimately able to achieve a score of 0.412. This write-up includes a discussion of what worked,
what didn't and how the project implementation could be improved going forward.

### Architecture ###

**Why Fully Convolutional Networks**

The goal of this project was to train a neural network to run inference on every single pixel in an image in order to 
determine 3 classes- the background, other people and the hero target. 

A typical convolutional network consists of a series of convolutions followed by a fully connected layer. A fully
connected layer is comprised of neurons that have connections to all activations in the previous layer as seen in
regular networks. The fully connected layer does not preserve spatial information as it flattens the spatial dimensions
out of the matrix. A convolutional network is great at classifying objects but since it does not preserve
spatial information, it can tell you if an object exists in an image, but it cannot tell you where that object is
in the image.

That is where the fully convolutional network comes in. In a fully convolutional network, the fully connected layer
is replaced by a 1x1 convolutional network. The point of this is to preserve the spatial information. Then subsequent
up-sampled transposed convolution layers are added to a point where the final upsampled block is the same 
size as the input block. The convolutional layers before the 1x1 convolutional layer are called encoder blocks
and the transposed layers after the 1x1 convolutional layer are called decoder blocks.

Skip connections are also used in fully convolutional networks to concatenate nonadjacent layers 
with previous layers that have the same height and width. This allows us to use different resolutions 
from previous layers resulting in a network that can make more precise segmentation decisions.

**My Architecture**

The architecure I chose has 2 encoder blocks followed by a convolutional layer followed by 2 decoder blocks as shown in
the figure below. The first encoder layer and last decoder layer have 64 filters. The second encoder and first decoder layer have 128 filters
and the convolutional layer has 256 filters. 

[image_0]: ./figures/architecture.jpg
![alt text][image_0] 


The python code that returns the model assembly followed by a 3 class softmax activation in the final output layer is:

```
def fcn_model(inputs, num_classes):   
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encoder_layer1 = encoder_block(inputs, 64, 2)    
    encoder_layer2 = encoder_block(encoder_layer1, 128, 2)   
   
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_layer = conv2d_batchnorm(encoder_layer2, 256, 1, 1)
  
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks   
    decoder_layer1 = decoder_block(conv_layer, encoder_layer1, 128)    
    x = decoder_block(decoder_layer1,inputs,64)  
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)
```


***Parameters***

The following are the final parameters I chose: 

* Learning Rate = .001
* Batch Size = 25
* Steps Per Epoch = 406 (for a larger than original sample size as discussed below)
* Num Epochs = 20

***Process of Optimizing Architecture and Parameters***

I started with an architecture that has 2 encoders and 2 decoders as shown above with
32 filters for encoder 1. Then I varied learning rate, batch size and steps per epoch 
while keeping number of epochs set at 10 to find IOU's that I could compare against. Since I originally
only used the 4100 given training images I modified the steps per epoch to be 4100 / (batch size) = 164. 
I had tried .01 and .1 as learning rates. Both of those were very large and my validation loss
result did not converge well. When I tried .001 I got a slower initial convergence rate
that ultimately settled to a better validation loss accuracy. With the learning rate set, I varied the batch size between 25, 35, and 50. I achieved
the lowest validation loss with 25 so I kept that as my batch size. All of the IOU's at this point were in the range
of .27 to .35.  

After finding this set of base parameters, I kept the parameters constant and then tried 
adding a third encoder and decoder and another architecture with a fourth encoder
and decoder. Both of those architectures seemed to overfit the data and provided no performance gains 
in validation loss or IOU. The architecture
with the 2 encoders and 2 decoders seemed to work the best. From there I adjusted the
number of filters in Encoder 1 from 32, 64 and 128. I found 64 filters achieved the best
result here. 128 seemed to overfit and 32 seemed to underfit. 64 filters did a much better
job of segmenting people from far away than the 32 filters did as I could see in the sample
evaluation images.

The IOU I was receiving at this point was in the high 30% range but not good enough to pass 
for the 40% IOU requirement. To increase the final score I had to increase the final_IoU or the weight
or both. So I noticed that the weight could be improved mostly by decreasing some false negatives
from hero detection from far away and that the final_IoU could be improved mostly by better
detection of the hero from far away as well. 

At this point I decided to add more input training images of spotting the hero and others from
a distance. I did this by setting up the following hero points and spawn points in this
section of the simulation as seen below in the figure.

[image_1]: ./figures/patrol.jpg
![alt text][image_1] 

I chose a few different spawn points within that simulation mostly 
so that I could get more images of the hero and other people from far away. I ended up collecting
6000 more images. Adding these testing images to the original ones and training the network
on them and increasing the number of epochs to 20, got me to a final result of 0.4127. I was
expecting higher, but this was good enough to pass and I was running out of time. The graph below shows the
final training curves. The validation loss was consistently trending downward throughout the training.

[image_2]: ./figures/validation.jpg
![alt text][image_2]

I had also tried 40 epochs with this sample size
and the parameters I had chosen above and was able to get just above 43% accuracy but the images
were overfitted. The figure below shows an example. There were a lot of sections on the concrete walkway
 showing up as others so I kept 20 for the number of epochs to avoid the overfitting.

[image_3]: ./figures/noise.jpg
![alt text][image_3] 

***Would this work with other images as classes such as a dog?***

My trained network would not work very well recognizing a dog, because none of my test images have
a dog in them. I would need several test images of a dog to train on for this to function properly.

### Future Enhancements ###

There are 3 things I believe would increase my IOU score that I would try if I had more time.
The first thing I would try
would be building and using the FCN-8 architecture with my current parameters and additional images I was
using for my final score. I suspect this would give me significant gains as this is a proven model. I 
would also look up other architectures that have won convolutional challenges.

The second thing I would try would be to record from different heights in the quadcopter. I suspect
I may have been flying a bit low when I took the additional 6000 images. This may give better features 
from which to train on.

And finally, I would like to investigate how I might tune these parameters more efficiently instead
of just brute forcing them. 