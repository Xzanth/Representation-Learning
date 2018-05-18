# Representation Learning

Unsupervised learning of the generating parameters of a linear model using
gradient descent.

## Description

We generate data **Y** from our parameters **X** with the following linear
mapping:

![Data](images/data_eqn.png)

Where our parameter **X** is a 100x2 matrix generated as such:

![Param](images/param_eqn.png)

From only the output data **Y** we wish to learn the generating parameter
**X**. We do this by finding the maximum likelihood estimate for W and
reversing the linear mapping. We find the MLE by minimising the negative log
likelihood using gradient descent. The negative log likelihood is our objective
function &#8466;(W).

![Arg Min](images/arg_min.png)

###  Equations

We use the following equations in our code, the terms are labelled as the
variables are named in python. For derivations of these equations please
consult `derivations.pdf` in this repository:

#### The objective function

![Objective Function](images/objective_function.png)

#### The derivative of the objective function

![Derivative Objective](images/derivative_objective.png)

## Results

![All data sets](images/original.png)

![Subset data sets](images/learned.png)

### Prerequisites

This code relies upon the python packages
* numpy
* matplotlib
* scipy

### Running

To produce the above graphs run:

```
./code.py
```
