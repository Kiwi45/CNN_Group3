# CNN_Group3

In this repo we present our approach to apply CNNs on a structured dataset in order to forecast sales.


Motivation
 
The increasing popularity of websites such as Instagram, Facebook or Youtube has lead to an increase in visual data over the last few years. These websites have become part of everyday life for many people and are therefore used excessively. The use of such websites has contributed among other things to enormously increasing amount of visual data in the process. Every day thousands of images and videos are uploaded, making it virtually impossible to analyze them by hand, as the sheer mass of content does not allow it. This observation clearly underlines the need for algorithms such as Convolutional Neural Networks (CNNs) to process and analyze visual data. This is particularly relevant as the amount of data from visual files will continue to increase in the future and therefore cannot be analyzed manually. For this reason, advanced concepts related to CNNs that enable the processing of visual data that target a different application area besides image classification are examined. Next, since a detailed description of the theory of CNNs has already been presented by our colleagues, this blog focuses on the practical application of CNNs to real-world sales forecasting, rather than an introduction of the implementation of CNNs in general. However, the key concepts of the CNNs will be explored and explained so that this blog post can inform and eventually educate. This allows readers of this blogpost to understand the different steps of a CNN and consequently show how we apply a CNN to a structured dataset. In addition, this blog discusses possible problems that may arise in the application of CNNs and possible solutions. As an example, the pre-processing of the data set or the handling of different image sizes as input for CNN can be mentioned. 

Before CNNs are applied to our use case, which will be further discussed in this blog, a literature review is be conducted to show the state of the art regarding CNN. This will make it possible to gain an even more detailed insight into the subject.


Literature Review

The authors Shaojie Bai, J. Zico Kolter, and Vladlen Koltun (2018) conclude in their paper that CNNs inherit the potential to outperform for example recurrent networks on tasks that involve audio synthesis and machine translation. The results of the authors indicate that simple CNNs are able to outperform anonical recurrent networks such as LSTMs across a diverse range of tasks and datasets. In addition the authors state that CNNs also provide longer effective memory than comparable models (Shaojie Bai et. al., 2018). Further it is concluded that CNNs should represent a natural starting point for sequence modeling tasks. With respect to the sales forecasts that have already been extensively researched in the literature (Montgomery et. al., 1990), some authors refer to the identification of human forecasts, which, however, are not of constant quality (Lawrence et. al., 1992). Whereas others examine the effects of different sales strategies on sales forecasting in combination with time series forecasts (Chatfield, 200). In addition, there are customer-focused strategies to increase sales, such as product consulting and monetary strategies that include, among other things, free benefits or other discounts to increase sales. All these effects have an impact and can then be presented in the sales forecast. Compared to other application, such as store sales, there is very little research in the field of sales forecasting using CNNs. We have therefore identified Carson Yan's blog post (2018) as the most relevant research approach in this context. The author provides a blog post in which CNNs are applied to structured bank customer data, taking into account product-related features such as product diversity and a time horizon. The goal was to determine whether customers will buy certain products in the future or not. Derived from this, the author gives a general approach on how CNNs can be applied to structured banking data. We use the described approach as a benchmark for our own model. However, unlike the approach described in the blog post, we also use other methods to model the sales forecast with a CNN. Therefore, our approach will be explained in detail throughout this blog post.

Use Case
As a starting point, a sales forecast is defined as a tool to enhance the understanding of one’s business and ultimately plays a major role in the companies success, with respect to the short and long run. However, many companies rely on human forecasts that do not hold the constant quality needed or use standard tools that are not flexible enough to suit their needs. In this blog we create a new approach to forecast sales using a CNN.

Therefore, this blog will present the practical use of deep learning in Convolution Neural Network and we will present our approach on the task of predicting the daily sales. In this blog, CNN’s are applied to the Rossmann Dataset that can be found on Kaggle competition (https://www.kaggle.com/c/rossmann-store-sales). Since Rossmann operates more than 3,000 drug stores in seven European countries, the initial challenge was to forecast six weeks of daily sales for 1,115 stores located across Germany. Overall the challenge attracted more than 3,738 data scientists, which consequently made it the second most popular competition by the number of participants.


Architecture Overview 
To begin with, a CNN provides similar characteristics to a multi-layer perceptron network. However, the major differences are what exactly the network learns, and moreover, how they are structured and ideally which purpose they are used for. In addition, CNNs are inspired by biological processes, thus their structure has a semblance of the visual cortex for example presented in an animal. In the following an overview of the architecture is given.




As already mentioned CNNs find application in computer vision and therefore have been successful with respect to the performance of image classification on a variety of test cases. CNNs take advantage of the fact that the input volume are images. 

In comparison with regular neural networks, the different layers of a CNN have the neurons arranged in three dimensions: width, height and depth. Further due to the spatial architecture of of CNNs, the neurons in a layer are only connected to a local region of the layer that comes before it. Normally, the neurons in a regular neural network are connected in a fully-connected manner. Finally the output layer of a CNN reduces the images into a single vector of class scores, that are arranged along the depth dimension. 

In the following the different layers that a CNN inherits are examined to gain further insight.

Convolutional Layers

The conv layer is the core building block of a CNN that does most of the computational heavy lifting.
The Conv layer read an input, such as a 2D image or a 1D signal using a kernel that reads in small segments at a time and steps across the entire input field. Each read results in an interpretation of the input that is projected onto a filter map and represents an interpretation of the input.


In the figure below is an example of a conv operation in 2D, but in reality convolutions are performed in 3D.
Let’s explain how is the Conv network performed. First of all the convolution is performed on the input volume with using of a filter to then produce a feature map.
During the execution of the convolutional, the filter slides over the input and at every location a matrix multiplication is performed and sums the results onto the feature map. The area of our filter is also called the receptive field and the size of it is 7x20.




Pooling Layer

The pooling layer follows the convolutional layer, in which the aim is dimension reduction. The reason is that training a model can take a large amount of time, due to the excessive data size. The pooling layer represents a solution to this issue. For example one can consider the use of max pooling, in which only the most activated neurons are considered. Moreover, pooling does not affect the depth dimension of the input image. However, the reduction in size leads to a loss of information, referred to as down-sampling. Nonetheless, pooling enables a decrease in computational power for the following layers and also works against overfitting. 
 
Further, pooling can be used to extract rotational and position invariant feature extraction, as max pooling, as stated before preserves only the dominant feature value. Thus, the extracted dominant feature is potentially from any position inside the region. 




Activation function

An activation function is f(x) is important to make the network more powerful and add ability to it to learn something complex and complicated from data.
The CNN calculates the weighted sum of its input, adds a bias and then decides it should be activated or not.
A neuron is calculated:
Y = ∑(weight * input) + bias
The value of  Y can be ranging from –inf to +inf. By adding the activation function the output of neural network will be determined and the resulting values will be mapped between 0 to 1.

The Activation function can be basically divided into 2 types:
1.	Linear Activation Function
2.	Non-Linear Activation Functions

 

Linear Activation Function

As you can see the function is linear and the functions will not be confined between any range.
F(x) = x
As range the linear function has –infinity to infinity and it doesn’t help with the complexity of usual data that is fed to the neural networks.

Non-linear activation function

The Non-linear Activation Functions are the most used activation functions. It makes it easy for the model to generalize or adapt with variety of data and to differentiate between the output.
Also another important feature of a Non linear Activation function is that it should be differentiable. It means that the slope(the change in y-axis and x-axis) isn’t constant.  
For our model we used the Relu (Rectified Linear Unit) function.The ReLu is the most used activation function in the world.Since, it is used in almost all the convolutional neural networks or deep learning.



Relu function

The function returns 0 if it receives any negative input, but for any positive value x it returns that value back. It can be written as:
        	F(x)0 max(0,max)

As you can see the ReLu function is simple and can allow our model to account for non-linearities and interactions.
How ReLu captures Interactions and Non-linearities

Interactions

Example: Let’s say we have a neural network with a single node. For simplicity, we assume it has two inputs A and B. The wights from A and B into our node are 2 and 3 respectively. So the output is f(2A + 3B).
We use the ReLu function four our f. An output we’ll have 2A+3B if our f positiv, otherwise the output value of our value is 0, if our f in negative.

No-linearities

A function is non-linear if the slope isn’t constant. That means, that the ReLu function is non-linear around 0, but the slope is always either 0 (for negative values) or 1 (for positive values).
As we saw before a Bias is important to calculate the weight of the neuron. 

Example: Let’s consider a node with a single input called A, and a bias. If the bias term takes a value of 7, then the code output is (7+A).
If A is less than -7, the output is 0 and the slope is 0. If A is greater than -7, then the node’s output is 7+A, and the slope is 1.
So the bias term allows us to move where the slope changes. So far, it still appears we can have only two different slopes.
Real models have many nodes. Each node can have a different value for it’s bias, so each node can change slope at different values for our input. 


Fully-connected Layer

The last layer within a CNN is usually the fully-connected layer that tries to map the 3-dimensional activation volume into a class probability distribution. Further, it is to mention that the fully-connected layer is structured like a regular neural network. Therefore, an activation function, in our case ReLu, is used to activate the output to map it to sales value.

Summary of layers 

As we described above, a CNN can be thought of as a sequence of layers, in which every layer of the CNN transforms the input volume of activations to another layer through a differentiable function. Therefore, three main types of layer to build a CNNs architecture can be identified, namely: Convolutional layer, Pooling Layer and Fully-Connected Layer. 

In summary:
The simplest CNN architecture is a list of layers which consequently transform the input image into an output image 
Distinct types of Layers are CONV/FC/RELU/POOL
Moreover the layers accept an input 3D volume, which is transformed into an output 3D volume 
The layers consist of parameters (e.g. CONV/FC do, but RELU/POOL do not)
Certain layers potentially have additional hyperparameters (e.g. CONV/FC/POOL do, RELU doesn’t)
 

Comparison CNN vs. TCN

We want to identify key differences with respect to Temporal Convolutional Networks (TCNs). The term Temporal Convolutional Networks is a vague term that could represent a wide range of network architectures.
The differentiate of the characteristics of TCNs are:
 Causality of the architecture (no information from future to past)
 Mapping any sequence of any length to an output sequence of the same length.
In other words the TCNs do is simply stacking a number of residual blocks together to get the receptive field that we desire.
 
Shaojie Bai, J. Zico Kolter, and Vladlen Koltun (2018) also provide this useful list of advantages and disadvantages of TCNs.
 TCN‘s provides a massive parallelism and short both training and evaluation cycle.
Lower memory requirement for training, especially in the case of long input sequences.
TCNs offer more flexibility in changing its receptive field size, principally by stacking more convolutional layers, using larger dilation factors, or increasing filter size. This offers better control of the model’s memory size.
However, the researchers note that TCNs may not be as easy to adapt to transfer learning as regular CNN applications because different domains can have different requirements on the amount of history the model needs in order to predict.
Hence, when transferring a model from a domain where only little memory is needed to a domain where much longer memory is required, the TCN may perform poorly for not having a sufficiently large receptive field. Therefore we are using in this blog the CNN application to predict the daily sales for the Rossman stores.
An important secondary benefit of using CNNs is that they can support multiple 1D inputs in order to make a prediction. This is useful if the multi-step output sequence is a function of more than one input sequence. This can be achieved using two different model configurations.
Multiple Input Channels: This is where each input sequence is read as a separate channel. 
Multiple Input Heads: This is where each input sequence is read by a different CNN sub-model and the internal representations are combined before being interpreted and used to make a prediction.


Technical Considerations

The biggest difficulty related to the architecture of a CNN that needs to be noted is the memory bottleneck. Ultimately, most modern GPUs have a 3/4/6GB memory limit. This results in the following main source of memory, which should be considered. First, the intermediate volume sizes that describe the number of activations at each level of the CNN, plus their gradients. Normally, this is where most of the activations are located. These activations are retained because they are needed for backpropagation. However, it is possible to reduce this through an intelligent implementation, as long as a CNN only executes the current activations at the given test time and stores them at any level. Previous activations are then stored at the lower level. Also of consideration are the parameter sizes, as these contain the numbers holding the network parameters and their gradients during back propagation. If the network does not fit, it is possible to reduce the batch size, where most of the memory is normally consumed by the activations.


