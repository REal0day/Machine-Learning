# Long Short-Term Memory Neurons and Gated-Recurrent Units

Many of the solutions previously presented for vanishing gradients an also apply to RNN: different activation functions, batch normalizations, etc... However because of the length of time series input, these could slow down training.

A possible solution would be to just shorten the time steps used for predictions, but this makes the model worse at predicting longer trends.

Another issue RNN face is that after awhile the network will begin to f"forget" the first inputs as information is lost at each step going through the RNN. **We need some sort of "long-term memory" for our networks.**



