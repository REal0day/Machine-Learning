# Long Short-Term Memory Neurons and Gated-Recurrent Units

Many of the solutions previously presented for vanishing gradients an also apply to RNN: different activation functions, batch normalizations, etc... However because of the length of time series input, these could slow down training.

A possible solution would be to just shorten the time steps used for predictions, but this makes the model worse at predicting longer trends.

Another issue RNN face is that after awhile the network will begin to f"forget" the first inputs as information is lost at each step going through the RNN. **We need some sort of "long-term memory" for our networks.**

The **Long Short-Term Memory \(LSTM\)** cell was created to help address these RNN issues. Created in 1997.

## RNN Structure![](/assets/Screen Shot 2018-02-01 at 4.33.14 AM.png)![](/assets/Screen Shot 2018-02-01 at 4.33.46 AM.png) 

These outputs are typically called **h**idden.  
h of t is a typical output of a RNN cell.

## LSTM Structure![](/assets/Screen Shot 2018-02-01 at 4.35.15 AM.png)

Here is a more detailed picture.![](/assets/Screen Shot 2018-02-01 at 4.36.14 AM.png)

**h** is our typcial input and output based on time, just like **c-**input and **c**-output, our memory/LSTM cells.



