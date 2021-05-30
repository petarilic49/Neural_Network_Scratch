# Neural Network to Predict Heart Failure

# Introduction
With the current COVID-19 Pandemic, machine learning and artificial intelligence within the healthcare industry has been advancing rapidly. In particular, the fundamental principle within machine learning and deep learning is the concept of artificial neural networks. Artificial Neural Networks (ANN) are computing systems vaguely inspired by the biological neural networks that constitute human and animal brains. ANN's utilize a collection of connected units or nodes called artificial neurons, which model the neurons in a biological brain. Before diving deep into how artifical neurons work and how they are connected to create a neural network, it is important to understand how a simple biological neuron operates. Below is an image of a biological neuron. 
![image](https://user-images.githubusercontent.com/37299986/120113421-6cacc480-c148-11eb-947f-8364dd75e204.png)
Biological neurons produce short electrical impulses called _Action Potentials_ which travel along the axons and make the synapses release chemical signals called neurotransmitters. When a neuron recieves a sufficient amount of these neurotransmitters within a few milliseconds, it fires its own electircal impulses. The telodendria contains minuscule structures called synaptic terminals which are connected to dendrites or cell bodies of other neurons. Therefore when a network of neurons are connected together, a forward wave, or propagation, occurs when neurons within each layer fire simultaneously. This brings us to how a simple artificial neuron works (also known as perceptron) and how they can be modeled together. 
![image](https://user-images.githubusercontent.com/37299986/120113724-c5c92800-c149-11eb-9e6f-f4d74507a939.png)
The inputs and outputs are numerical values, with each input associated with a weight value. The perceptron computes the weighted sum between each input and its associated weight and then applies a _Step Function (or Activation Function)_ to the weighted sum. To model a combination of neurons, we can structure them in a vertical fashion. Each output of the previous neuron is fed through the next neuron as an input. When neurons in the curretn vertical layer are connected to every neuron in the previous vertical layer, the layer is called a _dense layer._ In addition, each neuron contain an extra bias feature which is used to provide flexibility to the neural network model. This bias value is typically initialized as 1 or 0. The image below displays a simple neural network system containing a dense hidden layer and output layer. The first layer of a network is always refered to as the input layer, the last layer of a network is always refered to the output layer, while the n number of layers in between are referred to as hidden layers. 
![image](https://user-images.githubusercontent.com/37299986/120114204-06c23c00-c14c-11eb-8eee-41a37850cfc1.png)

# Problem Description

Neural Network Visual Model: 
![image](https://user-images.githubusercontent.com/37299986/120112367-87306f00-c143-11eb-8706-97683beacd93.png)

Individual Neuron Model (Hidden Layer and Output Layer):
![image](https://user-images.githubusercontent.com/37299986/120112536-3d945400-c144-11eb-81c8-35ef59d0e96f.png)
