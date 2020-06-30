# Vectorized Shallow (2 Layer) Neural Network Library

This project implements a two layer neural network using Python 3 and Numpy. 

## Getting Started

These instructions will get you a copy of the module up and running on your local machine for development and testing purposes. 


## Installing

To install this module, drop and include the following files into your project: 
1. shallow_nn.py.js
2. latest version of numpy (required)

## Functions 
### Name: shallowNN
##### Description

Creates a 2-layer Neural Net object and trains it based on the provided arguments

##### Usage
> shallowNN( X, Y, num_iterations, learning_rate, print_cost)

##### Parameters

1. X - Numpy Matrix containing your training examples, with training examples stacked up in columns 

![X](https://github.com/allpark/Python-Shallow-NN/blob/master/doc_images/x_matrix_diagram.jpg)

2. Y - Numpy Matrix containing your "true" label values corresponding to each training example

![Y](https://github.com/allpark/Python-Shallow-NN/blob/master/doc_images/y_vector_diagram.jpg)

3. num_iterations - number of iterations of gradient descent to be performed 

4. learning_rate - hyperparameter for controlling how much weights and biases are changed in respect to loss 

5. print_cost - print loss every iteration of gradient descent

##### Return values

1. object  - containing our trained model 

##### Example

This trains a network to be able to perform XOR, given two input features
```
X = np.array([[0,0,1,1],[0,1,0,1]])
Y = np.array([[0,1,1,0]])

nn = shallowNN( X, Y, num_iterations = 1000, learning_rate = 0.1, print_cost = False)

predictions = np.squeeze( nn.predict(np.array([[0,0,1,1],[0,1,0,1]])) > 0.5)

```


## Technologies

* Python 3
* Numpy 


## Authors

* **Allan Parker** - *Initial work* - [AllPark](https://github.com/allpark)
* **Andrew Ng** - *course mentor* 

