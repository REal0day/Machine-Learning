# MNIST Data

This is a classic dataset in Deep Learning. The data is easily accessible via TensorFlow as it has:

* 55k training images
* 10k test images
* 5k validation images

If you downloaded the source material, you already have it.

#### MNIST Dataset

The data contains handwritten single digits from 0 to 9. 

![](/assets/Screen Shot 2018-01-31 at 8.02.45 PM.png)

Single digit images can be represented as an array.

![](/assets/Screen Shot 2018-01-31 at 8.02.58 PM.png)We can also see this as an a array of 28x28 pixel image.

![](/assets/Screen Shot 2018-01-31 at 8.05.05 PM.png)

Values represent the greyscale. We can flatten this array to a 1-D Vector of 784 numbers. Either \[784;1\] or \[1;784\] works, long things are consistence through the rest of the numbers. Change it from 2D to 1D does change some properties, but we don't need to worry about that. We can think of the entire group of the 55k images as a **tensor** \(an n-dimensional array\).

![](/assets/Screen Shot 2018-01-31 at 8.09.05 PM.png)For labels, we'll be using **One-Hot Encoding**. This means that instead of having labels such as "one", "two", etc.... we'll have a single array for each image

The label is represented based off the index position in the label array. The corresponding label will be a 1 at the index location and zero every where else. 

Ex\)

Number 4 == \[0,0,0,0,1,0,0,0,0,0,0\]

The result of the training data ends up being a large 2D array.





