# Study of the reliability of a complex system

We have a system composed of three types of elements,
each having their own fail rate. The global system requires a certain number of
elements of every type to work. All types of element have their own repair time.

This python code uses numpy to model such a system, and compute its global
reliability as a function of different parameters.

###Schemas

![system schema](http://pix.toile-libre.org/upload/original/1463020658.png)


![Graph of the sytem states, with X = number of failing elements of type 2 and Y =  number of failing elements of type 3](http://pix.toile-libre.org/upload/original/1463020482.png)
