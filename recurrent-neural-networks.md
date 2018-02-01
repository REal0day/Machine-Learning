# Recurrent Neural Networks

Examples

* Times Series Data
* Sentences
* Audio
* Car Trajectories
* Music



Let's imagine a sequence:

* \[1,2,3,4,5,6\]

Can you predict the future?

* \[2.3.4.5.6.7\]

**Unrolling a neuron.**

![](/assets/Screen Shot 2018-02-01 at 3.08.10 AM.png)

Each neuron has 2 inputs. previous and data.

* Cells that are a function of inputs from previous time steps are also known as **memory cells**.
* RNN are also flexible in their inputs and outputs, for both sequences and single vector values.
* It's easy to make a layer of recurrent neurons.
* RNN Layer with 3 Neurons![](/assets/Screen Shot 2018-02-01 at 3.09.41 AM.png)

* **Sequence --&gt; Sequence **
  * \[1,2,3,4,5,6\]

  * \[2.3.4.5.6.7\]
* **Sequence --&gt; Vector**
  * Did you like his music?
    no.

  ![](/assets/Screen Shot 2018-02-01 at 3.12.27 AM.png)

* **Vector --&gt; Sequence**
  * Send in an image, and receive a caption. Auto generating text. Picture of guy on beach. Text says, "Guy on beach."



