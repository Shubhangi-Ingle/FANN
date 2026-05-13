# importing Python library
import numpy as np

# define Unit Step Function
def unitStep(v):
    if v >= 0:
        return 1
    else:
        return 0

# design Perceptron Model
def perceptronModel(x, w, b):
    v = np.dot(w, x) + b
    y = unitStep(v)
    return y

# AND Logic Function
def AND_Logic_Function(x):
    w = np.array([1, 1])   # weights
    b = -1.5               # bias
    return perceptronModel(x, w, b)

# testing the Perceptron Model
test_inputs = [
    np.array([0, 1]),
    np.array([1, 1]),
    np.array([0, 0]),
    np.array([1, 0])
]

print("__________ AND Function __________")

for x in test_inputs:
    print(f"AND({x[0]}, {x[1]}) = {AND_Logic_Function(x)}")
