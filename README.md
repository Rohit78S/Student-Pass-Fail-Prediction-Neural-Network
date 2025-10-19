# Student Pass/Fail Prediction Neural Network

## How It Works

The neural network takes student exam scores and study hours, then predicts if they will PASS or FAIL.

### Network Diagram

```
Input Layer          Hidden Layer         Output Layer
(2 neurons)          (4 neurons)          (2 neurons)
    
[Score]                                   
      \                                   
       [●]──Sigmoid──[●]                  
      /                \                  
[Hours]                 [●]───Softmax───[PASS/FAIL]
                        [●]
                        [●]
```

### Process

1. **Input** - Student score and study hours (2 values)
2. **Hidden Layer** - 4 neurons process data with sigmoid activation
3. **Output** - 2 neurons give PASS/FAIL probabilities using softmax
4. **Prediction** - Pick the highest probability

## What I Used

- **Python** - Programming language
- **NumPy** - Matrix operations and numerical calculations

## Why NumPy?

1. **Fast matrix multiplication** - Neural networks need lots of math. NumPy does it quickly.
2. **Easy arrays** - Simple way to work with student data
3. **Built-in functions** - `dot()` for matrix multiplication, `exp()` for sigmoid/softmax
4. **One-line operations** - Instead of loops:
   ```python
   result = np.dot(inputs, weights) + biases
   ```

## Simple Code Example

```python
# Input data
inputs = np.array([[90, 20.5], [80, 18.5]])

# Forward pass
hidden_output = sigmoid(np.dot(inputs, hidden_weights) + hidden_biases)
output = softmax(np.dot(hidden_output, output_weights) + output_biases)

# Prediction
prediction = np.argmax(output)
```

**Output:** `Student PASS (52% confidence)`
