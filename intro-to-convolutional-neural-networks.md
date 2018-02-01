# Introduction to Convolutional Neural Networks

![](/assets/CNN.png)

Things we'll go over

* Quick Review of Previous Topics and NNs
* New Theory Topics
* Discuss Famouse MNIST Data Set
* Solve MNIST with a "normal" NN
* Learn about CNN
* Solve MNIST with CNN
* CNN Exercise and Solutions Afterwards

---

## Quick Review

We know how to create calculations for a Single Neuron. 

* w \* x + b = z
* a = sigma\(z\)
  _Again, "z" is the result aka y._

---

We then pass z into an **activation function** which include:

* Perceptrons
* Sigmoid
* Tanh
* ReLU
* \(We will discuss more later on\)

---

We can connect these neurons to create a **Neural Network**.

* Input Layer
* Hidden Layer
* Output Layer

More layers --&gt; More Abstraction_  
Will take a longer amount of time/epoch._

---

To "learn", we need some **measurement of error.** We use a Cost/Loss Function

* Quadratic
* Cross-Entropy

---

Once we have the measurement of **error**, we need to **minimize** it by choosing the correct weight and bias values. We use **gradient descent** to find the optimal values.That is our learning process.

---

We can then **backpropgate the gradient descent** through multiple layers, from the output layer back to the input layer. We also know of dense layers, and later on we'll introduce **softmax layer.**

---

## New Theory Topic Lecture

This section goes over the following:

* Initialization of Weights Options
  * Zeros
    * No Randomness
    * Not a great choice
  * Random Distribution Near Zero
    * Not Optimal
    * Activation Functions Distortion
  * Xavier \(Glorot\) Initialization
    * Uniform / Normal
  * Draw weights from a distribution with zero mean and a specific variance

  ![](/assets/Screen Shot 2018-01-31 at 7.33.41 PM.png)

All weight is set by the value of its corresponding x, and together, they have a mean of 0. 

* x = 2, weight starts at -2, because \(2 + - 2\) / 2 =  0.

Var\(W\) = 1/n

Var\(W\) = 2 / n.in + n.out

n = number of neurons feeding into your neuron.

We will be using this in tf. **Don't have to memorize it**.

[More info.](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization)

---

**Learning Rate**: Defies the step size during gradient descent  
**Batch Size**: Batches all us to use stochastic gradient descent.

* Smaller ==&gt; less representative of data
* Larger ==&gt; longer training time

**Second-Order Behavior**: Allows us to adjust our learning rate based off the rate of descent

* AdaGrad
* RMSProp
* Adam

We will be focusing on Adam

**Unstable / Vanishing Gradients**: As you increase the number of layers in a network, the layers towards the input will be affected less by the error calcuations occurring at the output as you go backwards through the network.

Initialization and Normalization will help us mitigate these issues. We will discuss vanishing gradients again in more detail in the Recurrent Neural Networks portion. 

---

#### Overfitting Mitigation Techniques

With potentially hundreds of parameters in a deep learning neural network, the possibility of overfitting is very high. There are a few ways to help mitigate this issue.

* **L1/L2 Regularization**
  * Adds a penalty for larger weights in the model
  * Not unique to neural networks
* **Dropout**
  * Unique to neural networks
  * Remove neurons during training randomly
  * Network doesn't over rely on any particular neuron
* **Expanding Data**
  * Artificially expand data by adding noise, tilting images adding low white noise to sound data, etc.



