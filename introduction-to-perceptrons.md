# Introduction to Perceptrons

They're different types of functions one can use to help represent their data.

**sigmoid\(\): **All output is a range\[0,1\].  
![](https://lh5.googleusercontent.com/qu0hRWOsK4Ni2edDUI9hA3H92qTMgM_jc9ljlRuTo-XYmRyXtkpbKOHuQwqGebF44u2793AHMSjkb6xnLscKgBT7_g4Xxy-Ix5hrA9e6wyg7cb-_D6mq2hyaalR1GtYoiFO6TmxB)

**    
**

**Hyperbolic Tangent aka tanh\(z\): **All output is a range\[-1,1\]

![](https://lh6.googleusercontent.com/yhy1MEvaHE5YXL3qxMc0d7jP2ifVjMd7y8zLRYw94S3SnzlfLip262p7C1JHgUXyGr37xNWd-2bMha4FZ9dblhai17MbtC6Ixcpz7gmPahL_5zc7_iFfjv4h8HtqRTjR_KxGqReK)

**Rectified LInear Unit \(ReLUE\): **max\(0,z\)  
Pass your z value into this function\(\)  
Based off of z and 0, return the max value.

![](https://lh6.googleusercontent.com/lE9vM2vtsg5AtKgOYfmjxd3iiQE45ePy4OrY57DOgymWEKN56I_w_TAiGAdpcU6PhCZlu5hnGcxOwHjJbgaZclmLBBJWqkGZEPMXixz-xX8QDJU0cxdqq74FdU7o-q_LW9azGSTK)

**    
**

* ReLu and tanh tend to have the best performance, so we will focus on these two

* Deep Learning libraries have these built in for us, so we don’t need to worry about having to implement them manually.



