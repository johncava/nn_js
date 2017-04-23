var numeric = require('numeric')

var input = [  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1]]
var y = [[0],[0],[1],[1]]
var weights1 = numeric.random([3,5])
var weights2 = numeric.random([5,4])
var weights3 = numeric.random([4,1])

var hiddenLayer1
var hiddenLayer2

function sigmoid(x){
    return numeric.div(1,(numeric.add(1,numeric.exp(numeric.neg(x)))))
}

function sigmoid_derivative(x){
    return numeric.mul(x, numeric.sub(1,x)) 
}

function feedforward(){
    hiddenLayer1 = sigmoid(numeric.dotMMbig(input,weights1));
    hiddenLayer2 = sigmoid(numeric.dotMMbig(hiddenLayer1,weights2));
    var output = sigmoid(numeric.dotMMbig(hiddenLayer2, weights3));
    return output;
}

function neural_network(){
    var max_iterations = 60000;
    for (iteration = 0; iteration < max_iterations; iteration++){
        var output = feedforward();
        var outputError = numeric.sub(output,y)

        if (iteration % 10000 === 0){
            console.log(outputError)
        }

        //Backpropagation
        var output_layer_delta = numeric.mul(outputError,sigmoid_derivative(output))

        var hidden_layer2_error = numeric.dotMMbig(output_layer_delta,numeric.transpose(weights3))

        var hidden_layer2_delta = numeric.mul(hidden_layer2_error,sigmoid_derivative(hiddenLayer2))

        var hidden_layer1_error = numeric.dotMMbig(hidden_layer2_delta, numeric.transpose(weights2))

        var hidden_layer1_delta = numeric.mul(hidden_layer1_error,sigmoid_derivative(hiddenLayer1))

        //Gradient Descent 
        weights3 = numeric.sub(weights3, numeric.dotMMbig(numeric.transpose(hiddenLayer2), output_layer_delta))
        weights2 = numeric.sub(weights2, numeric.dotMMbig(numeric.transpose(hiddenLayer1), hidden_layer2_delta))
        weights1 = numeric.sub(weights1, numeric.dotMMbig(numeric.transpose(input),hidden_layer1_delta))
    }
}

neural_network();

