# Neural Network to Predict Heart Failure

# Introduction
With the current COVID-19 Pandemic, machine learning and artificial intelligence within the healthcare industry has been advancing rapidly. In particular, the fundamental principle within machine learning and deep learning is the concept of artificial neural networks. Artificial Neural Networks (ANN) are computing systems vaguely inspired by the biological neural networks that constitute human and animal brains. ANN's utilize a collection of connected units or nodes called artificial neurons, which model the neurons in a biological brain. Before diving deep into how artifical neurons work and how they are connected to create a neural network, it is important to understand how a simple biological neuron operates. Below is an image of a biological neuron. 
![image](https://user-images.githubusercontent.com/37299986/120113421-6cacc480-c148-11eb-947f-8364dd75e204.png)
Biological neurons produce short electrical impulses called _Action Potentials_ which travel along the axons and make the synapses release chemical signals called neurotransmitters. When a neuron recieves a sufficient amount of these neurotransmitters within a few milliseconds, it fires its own electircal impulses. The telodendria contains minuscule structures called synaptic terminals which are connected to dendrites or cell bodies of other neurons. Therefore when a network of neurons are connected together, a forward wave, or propagation, occurs when neurons within each layer fire simultaneously. This brings us to how a simple artificial neuron works (also known as perceptron) and how they can be modeled together. 
![image](https://user-images.githubusercontent.com/37299986/120113724-c5c92800-c149-11eb-9e6f-f4d74507a939.png)
The inputs and outputs are numerical values, with each input associated with a weight value. The perceptron computes the weighted sum between each input and its associated weight and then applies a _Step Function (or Activation Function)_ to the weighted sum. To model a combination of neurons, we can structure them in a vertical fashion. Each output of the previous neuron is fed through the next neuron as an input. When neurons in the curretn vertical layer are connected to every neuron in the previous vertical layer, the layer is called a _dense layer._ In addition, each neuron contain an extra bias feature which is used to provide flexibility to the neural network model. This bias value is typically initialized as 1 or 0. The image below displays a simple neural network system containing a dense hidden layer and output layer. The first layer of a network is always refered to as the input layer, the last layer of a network is always refered to the output layer, while the n number of layers in between are referred to as hidden layers. 
![image](https://user-images.githubusercontent.com/37299986/120114204-06c23c00-c14c-11eb-8eee-41a37850cfc1.png)

# Problem Description
After completeing this project, you will have a deeper understanding of how to construct a simple ANN from scratch which will play a major role in building different adaptions of neural networks (such as recurrent NN, convolution NN, etc). With the healthcare industry always searching for new and improved machine learning algorithms and models, in addition to my passion and curiousity for the human body, a simple one hidden layer neural network will be trained to predict the probability that a given patient endured a heart failure.

# Obtaining the Data
The most important step in creating a machine learning model is selecting the dataset. Fortunately there are many open source and public dataset available for free online, and one specific website that provides easy access data is Google LLC's subsidairy, Kaggle. After doing some quick searches the Heart Disease UCI dataset was selected. This dataset contained 13 inputs and one output (if the patient experienced heart failure or not). Though the dataset only containes 304 entries which is extremely small when it comes to machine learning models, the purpose of the project is to familiarize myself with how neural networks work and how they can be optimized. Therefore a smaller dataset may even be more appropriate for the sake of debugging. The inputs to the dataset are as follows: 
- Age
- Sex
- Chest Pain Type (4 values)
- Resting Blood Pressure
- Serum Cholestoral in mg/dl
- Fasting Blood Sugar > 120 mg/dl
- Resting Electrocardiographic Results (values 0,1,2)
- Maximum Heart Rate Achieved
- Exercise Induced Angina
- Oldpeak = ST Depression Induced by Exercise Relative to Rest
- The Slope of the Peak Exercise ST Segment
- Number of Major Vessels (0-3) Colored by Flourosopy
- Thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

Thus with this dataset, our neural network model will take in the 13 inputs and output a probability as to if the patient obtained a heart failure or not. This makes our neural network a **Binary Classification Neural Network**. However, if we take a closer look at the dataset we see that some imputs range between 0 and 1, while others range between 0 and 200. Therefore if we use this dataset as it is the neural network may bias its output to the higher numerical valued inputs. Thus it is crucial that the program normalizes all data entries between the value of 0 and 1. This will eliminate this problem as all inputs will be within a shared numerical range. 

# Splitting the Dataset
Once the dataset is normalized, the next crucial step is to split the dataset into training and testing. For this problem we will split the data into x_training/testing which represents the inputs for training and testing, and y_training/testing which represents the outputs of the training and testing data. However, another problem is encountered. AFter analyzing the dataset some more it was realized that the first 60% of the data obtained heart failures while the last 40% did not. If we train take the dataset as is and just assign the first 70% as training and last 30% as testing our model will not be accurate as we are introducing a sampling bias with the training data being mostly heart failure outputs. Therefore before splitting the data, the dataset was randomly shuffled to ensure that the model will see a more distributed number of heart failures in the training dataset. Once this was complete, the training data was split to 70% of the total dataset and the testing data was split to 30% of the total dataset. 

# Visualizing the Neural Network Structure
Before beggining to code the neural network functionalities, it is beneficial to create a visuale representation of how the neural network must look like, as well as researching common best practises for Binary Classification Neural Networks. Since we are creating the neural network from scratch using C++, and this is a fairly small dataset we will assume that our model will only contain one hidden layer. We know that the input layer will contain 13 nodes while the output layer will only contain one node since its a binary classification problem. Determining the number of neurons/nodes for the hidden layer was done through some trial and error once the model was created. The final number of hidden layer nodes/neurons was chosen to be 15. The diagram below displays neural network that needs to be created for this problem. 
Neural Network Visual Model: 
![image](https://user-images.githubusercontent.com/37299986/120112367-87306f00-c143-11eb-8706-97683beacd93.png)

# Forward Propagation
Now that the structure of the neural network is agreed upon, the next step is to focus on how the forward propagation of the network should be performed. For starters, the program must first randomly assign weight values to each neuron input. For consistency the C++ code initializes a weight between -1 and 1 for each input to a neuron in a given layer. From there, each bias per neuron is initialized as 0. This value will change minimally throughout the training process and initalizing them to 0 at the start will ensure that the weights are the driving factor when training the neural network model. Once all the weights and biases are initialized for each layer of the network, it is time to obtain the weighted sum. Through research, C++ contains a matrix algebra library called Eigen. Using this library allowed me to perform matrix algebra throughout the network very easily and efficiently. Thus this library was used in the 'weightedSum()' function within the neural_layer class. 

Once the weighted sum is obtained, the output must be passed through an activiation function. After research the current best practise for hidden layer activation functions is the ReLU Activation function. The function details are shown below. This activiation function is most oftenly used within hidden layers as it can effectivly model multdimensional characteristics, but also it is less susceptible to vanishing gradients that prevent deep models from being trained. 
![image](https://user-images.githubusercontent.com/37299986/120119522-1189ca80-c166-11eb-88c6-c49dd09b98fa.png)

However, the output layer of our model should be bounded between 0 and 1 since this is a binary classification problem. Therefore the output activation function used is the Sigmoid Activiation Function. Once again the Sigmoid Activiation Function details are shown below. As you can see below, the function takes any real value as input and outputs values in the range 0 to 1. The larger the input (more positive), the closer the output value will be to 1.0, whereas the smaller the input (more negative), the closer the output will be to 0.0.
![image](https://user-images.githubusercontent.com/37299986/120119606-9aa10180-c166-11eb-9e14-6e41fb9dde02.png)

Now that the activation functions have been chosen correctly, the following figure below displays the forward propagation for one neuron in the hidden layer and the output layer. 
Individual Neuron Model (Hidden Layer and Output Layer):
![image](https://user-images.githubusercontent.com/37299986/120112536-3d945400-c144-11eb-81c8-35ef59d0e96f.png)

# Backward Propagation (Training the Model) 
