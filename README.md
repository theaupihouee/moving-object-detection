# Moving Object Detection

The objective is to extract moving objects from a video. The video is encoded as a data matrix M with m rows (number of pixels) and n columns (number of video frames). If all the columns are the same, then there is no movement in the video and the data matrix M has rank one. **If one or more objects are moving, then they can be viewed as noise N (same dimensions as M) that is being added to a rank one matrix u×vT (the fixed backround)** for some u (m rows) and v (n rows). 

In other words, we have **M = u×vT + N**. It is thus possible to **recover the background (and hence the moving objects) by finding a closest rank one matrix x×yT to the data matrix M**, as follows : 


The goal is to solve this optimization problem and find the solutions x and y. To do so, we will use **stochastic gradient descent** with the following update rule : 
(see update rule from correction) 

Describe libraries used and structure of the code 

Put screenshots of the results 
