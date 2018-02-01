# Vanishing Gradient

Backpropagation goes backwards from the output to the input layer, propagating the error gradient. For deeper networks issues can arise from backpropgation, vanishing and exploding gradients! As you go back to the 'lower' layers, gradients often get _smaller_, eventually causing weights to never change at lower levels.

The opposite can also occur gradients explode on the way back, causing issues.

**Vanishing gradient problem happens more than exploding gradient problem** \(which is the opposeite of the vanishing gradient problem\).

Why do the vanishing gradients happen in relations to the activation choice?

Some of the typical activation functions \(like **sigmoid**\), we have this curvative that goes from \(0,1\)

![](/assets/Screen Shot 2018-02-01 at 4.08.18 AM.png)

Issue is that if you put a huge number, it'll be very close to 0 or 1. It will change the gradient exponetially with n as you get more layers, while the front layers change very slowly.

You can solve this by using different activation functions.**ReLu** won't saturate large positive values. Although, with negative values, it'll be 0. 

![](/assets/Screen Shot 2018-02-01 at 4.08.03 AM.png)

There's the **Leeky** **ReLU** which will handle lower values past 0.

![](/assets/Screen Shot 2018-02-01 at 4.08.29 AM.png)

