var numeric = require('numeric')

var input = [[1,2,3],
             [4,5,6],
             [7,8,9],
             [10,11,12]]
var y = [0,0,1,1]
var weights1 = numeric.random([5,5])
var weights2 = numeric.random([5,4])
var weights3 = numeric.random([4,6])
var weights4 = numeric.random([6,1])

function sigmoid(x){
    return numeric.div(1,(numeric.add(1,numeric.exp(numeric.neg(x)))))
}

function sigmoid_derivative(x){
    return numeric.mul(x, numeric.sub(1,x)) 
}

function feedforward (){
    var hiddenLayer1 = sigmoid(numeric.dotMMbig(input,weights1))
    var hiddenLayer2 = sigmoid(numeric.dotMMbig(hiddenLayer1,weights2))
    var hiddenLayer3 = sigmoid(numeric.dotMMbig(hiddenLayer2, weights3))
    var output = sigmoid(numeric.dotMMbig(hiddenLayer3, weights4))
    console.log(output)
    return output
}

feedforward();