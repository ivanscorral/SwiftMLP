import SwiftMLP

let xorInputs = [
    MLVector<Float>([0, 0]),
    MLVector<Float>([0, 1]),
    MLVector<Float>([1, 0]),
    MLVector<Float>([1, 1])
]

let xorOutputs = [
    MLVector<Float>([0]),
    MLVector<Float>([1]),
    MLVector<Float>([1]),
    MLVector<Float>([0])
]

let mlp = MultiLayerPerceptron(
    layers: [2, 6, 1], // 2 neurons in input layer, 2 in hidden layer, 1 in output layer
    learningRate: 0.01,
    activationFunctions: [.relu, .relu, .relu, .relu, .relu, .relu], // Only one sigmoid for the hidden layer
    lossFunction: MeanSquaredErrorLoss()
)

// Training loop
for _ in 0..<2000 { // Adjust the number of iterations as needed
    for (input, target) in zip(xorInputs, xorOutputs) {
        mlp.train(input: input, target: target)
    }
}

// Testing
for input in xorInputs {
    let output = mlp.predict(input: input)
    print("Input: \(input) Output: \(output)")
}
