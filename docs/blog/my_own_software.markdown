---
layout: default
title: My Own Software
permalink: /my_own_software/
---

# My Own Software

I am a self taught programmer working with artificial neural networks, and my first software package, Parana(Parmeter Analysis) was a way of teaching myself how to code by building something. I had to read through tensorflow documentation, sometimes go through code, this was all good practice. The repo is up on github, being a first project there are some big problems with it. Some of these problems are from a lack of planning that comes from adding features as they were needed. I learned some valuable lessons, and my second project Sparana(sparse parameter analysis) is better designed and built. Sparana is built without Tensorflow, it uses [CuPy](https://github.com/cupy/cupy) for matrix operations, and [Numba](http://numba.pydata.org/) for custom operations that are not supported by CuPy. It has some support for sparse weight matrices, which will be expanded as I develop more experiments for these types of models. I train small models on a GPU on my home computer. I have had some success improving performance against adversarial attacks, and training models on a small subset of (<1%) parameters. 

---

### Why use an old GPU for training? 

GPU services cost money, this is not making me any money. I will never be able to compete with large tech companies for computational power. There is plenty of research being done by building bigger and bigger models, using more and more computational resources. Using a single GPU with limited space forces me to think about memory and computational efficiency in a way that leads to making better software. Thinking about solving problems with less resources means I might find solutions that nobody else is even thinking about trying. Algorithms that are efficient at a small scale will be efficient at a large scale. 

Established neural network software supports multiple GPUs, and can be run on services like AWS(Amazon Web Service) but developing for this would mean another layer of complexity that is difficult and time consuming for someone who is self taught. I do experiments that I find interesting without spending extra time developing software. If it comes time to scale up, I could buy a new GPU, or implement whatever works with existing frameworks. My software still runs slower than Tensorflow, I'll work on improving that, but I wouldn't recommend it for anyone else.

---

### Only ReLu and Quadratic cost

With Sparana I have only built layers with ReLu activations, and optimizers with quadratic cost. I am building features as I need them (I'll get around to convolutional layers one day), I found good results with only ReLu and quadratic cost. I will write another blog about my choice of cost function soon. Gradients are hard coded so no fancy calculus software, I have had perfectly good performance with this. 

---

### Next

I have a couple more blogs to write based on work that I have done. A quick intro into parameter analysis and things that I have tried and failed. If you want to look at parameters in the same way I do, it helps to know what doesn't work as well as what does. 