# Matrix multiplication function
def matrix_mult(matrix1, matrix2):
    result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]
    #print(result)
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                #print(matrix1[i][k], matrix2[k][j])
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    #print(result)
    return result

def relu(matrix):
    return [[max(0,x) for x in row] for row in matrix]

# Define the neural network evaluation function
def evaluate_neural_network(input_matrix, weights, biases, use_relu):
    layer_input = input_matrix
    for i in range(len(weights)):
        layer_output = matrix_mult(weights[i], layer_input)
        for j in range(len(layer_output)):
            layer_output[j][0] += biases[i][j][0]
        if use_relu[i]:
            layer_output = relu(layer_output)
        layer_input = layer_output
        #print(layer_input)
    return layer_output

# Given network parameters for part 1
input_matrix1 = [[1.0]]
weights1 = [[[3.0]]]
biases1 = [[[3.0]]]
use_relu1 = [False]

input_matrix2 = [[1.0],[1.0]]
weights2 = [[[6.0, 8.0]]]
biases2 = [[[0.0]]]
use_relu2 = [False]

input_matrix3 = [[1.0]]
weights3 = [[[0.0], [-2.0], [9.0]], [[5.0, 4.0, -8.0]]] 
biases3 = [[[8.0], [7.0], [-2.0]], [[-2.0]]]
use_relu3 = [True,False]

input_matrix4 = [[1.0],[1.0], [1.0]]
weights4 = [[[-5.0, 3.0, -4.0], [-6.0, 9.0, 9.0], [7.0, 6.0, -4.0]], [[2.0, -8.0, -4.0], [-7.0, -4.0, -6.0]], [[1.0, 3.0]]]
biases4 = [[[-5.0], [1.0], [9.0]], [[4.0], [9.0]], [[0.0]]]
use_relu4 = [True, True, False]

input_matrix5 = [[1.0]]
weights5 = [[[-8.0], [2.0], [-9.0], [4.0]], [[8.0, 0.0, 6.0, 4.0], [8.0, -6.0, -3.0, -7.0], [-7.0, -7.0, -4.0, -4.0]], [[5.0, -7.0, 6.0]]]
biases5 = [[[9.0], [7.0], [1.0], [5.0]], [[0.0], [-8.0], [5.0]], [[0.0]]]
use_relu5 = [True, True, False]

if __name__ == "__main__":
    # Evaluate the network for the given input
    #output = evaluate_neural_network(input_matrix1, weights1, biases1, use_relu1)
    #output2 = evaluate_neural_network(input_matrix2, weights2, biases2, use_relu2)
    #output3 = evaluate_neural_network(input_matrix3, weights3, biases3, use_relu3)
    output4 = evaluate_neural_network(input_matrix4, weights4, biases4, use_relu4)
    output5 = evaluate_neural_network(input_matrix5, weights5, biases5, use_relu5)
    # Print the output
    #print("Output:", output)
    #print("Output 2:", output2)
    #print("Output 3:", output3)
    print("Output 4:", output4)
    print("Output 5:", output5)