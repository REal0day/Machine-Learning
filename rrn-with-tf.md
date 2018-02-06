# Recurrent Neural Network with TensorFlow

We are going to go over how to use TensorFlow built-in tf.nn function API to solve sequence problems. Our previous example was to solve the next array after \[0,1,2,3,4,5\]. Now we will solve a more complex on like \[0, 0.84, 0.91, 0.14, -0.75, -0.96, -0.28\]

It appears pretty confusing, but if you feed this into a graph, it's simply **sin\(x\)**.

![](/assets/sinx.png)

We will start be creating a RNN that attempts to predict a timeseries shifted over 1 unit into the future. Then we'll attempt to generate new sequences with a seed series.

We'll first create a simple class to generate sin\(x\).![](/assets/Screen Shot 2018-02-02 at 7.05.46 PM.png)  
Then we will be able to feed in random batches of sin\(x\).This is the training batch added to our RNN.![](/assets/Screen Shot 2018-02-02 at 7.05.58 PM.png)If we viewed it on the graph, it'd look like this.![](/assets/Screen Shot 2018-02-02 at 7.06.13 PM.png)Then the trained model will be given a time series and will attempt to predict a time series shifted one time step ahead.![](/assets/Screen Shot 2018-02-02 at 7.08.42 PM.png)As you can see, the prediction gets better and bette as time goes on. The first point is far off, the 2nd is almost right on top.

# Generating New Sequences {#Generating-New-Sequences}

After that, we will give it a seed series, and ask it to predict a new sequence.Here, we feed it zeros. It's passing this off of what it knows about the sin\(x\). We

```py
with tf.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model")

    # SEED WITH ZEROS
    zero_seq_seed = [0. for i in range(num_time_steps)]
    for iteration in range(len(ts_data.x_data) - num_time_steps):
        X_batch = np.array(zero_seq_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        zero_seq_seed.append(y_pred[0, -1, 0])
```

We first restore our model.

X\_batch = _  
_y\_batch =  
zero\_seq\_seed =

Now time to plot.

```py
plt.plot(ts_data.x_data, zero_seq_seed, "b-")
plt.plot(ts_data.x_data[:num_time_steps], zero_seq_seed[:num_time_steps], "r", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")
```

![](/assets/download-1.png)

