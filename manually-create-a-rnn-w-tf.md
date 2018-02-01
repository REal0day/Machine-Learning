# Manually Create a RNN w/ TF

We will create a 3 Neuron Recurrent Neural Network with TensorFlow. The main idea to focus on here is the input format of the data.![](/assets/Screen Shot 2018-02-01 at 3.09.41 AM.png)

Let's start by running the RNN for 2 batches of data, t=0 and t=1.

Each Recurrent Neruon has 2 sets of weights:

* Wx for input weights on X
* Wy for weights on output of original X

![](/assets/Screen Shot 2018-02-01 at 3.20.19 AM.png)  
**num\_batches**: size of one sample.  
**batch\_size: **Samples of dataset  
**time\_steps**: Intervals per Sample  


Feed in based on the timestamp. from t=0 --&gt; t=4  


