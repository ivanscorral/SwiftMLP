//
//  MultiLayerPerceptron.swift
//
//
//  Created by Ivan Sanchez on 26/11/23.
//

import Foundation


public class MultiLayerPerceptron {
    var layers: [Int]
    var learningRate: Float
    var activationFunctions: [ActivationFunction]
    var lossFunction: LossFunction
    
    var weights: [MLMatrix<Float>]
    var biases: [MLVector<Float>]
    
    public init(layers: [Int], learningRate: Float, activationFunctions: [ActivationFunction], lossFunction: LossFunction) {
        self.layers = layers
        self.learningRate = learningRate
        self.activationFunctions = activationFunctions
        self.lossFunction = lossFunction
        
        // Create a weight matrix fo    r each layer
        weights = []
        biases = []
        
        for layerIndex in 1..<layers.count {
            let previousLayerSize = layers[layerIndex - 1]
            let layerSize = layers[layerIndex]
            
            let weightMatrix = MLMatrix<Float>.init(randomFilledRows: layerSize, columns: previousLayerSize)
            
            weights.append(weightMatrix)
            
            let biasVector = MLVector<Float>.init(repeating: 0.0, count: layers[layerIndex])
            biases.append(biasVector)
        }
    }
    
    
    func forward(input: MLVector<Float>) -> MLVector<Float> {
        var activation = input
        
        for (index, weightMatrix) in weights.enumerated() {
            // Calculate the weighted sum (z) for this layer
            let z: MLVector<Float> = (weightMatrix * activation).adding(biases[index])
            print("z: \(z)")
            // Apply the activation function
            activation = MLVector(z.map { activationFunctions[index].apply($0) })
        }
        return activation
    }
    public func train(input: MLVector<Float>, target: MLVector<Float>) {
        // Forward pass
        var activations = [input]
        var zs = [MLVector<Float>]()
        
        for (index, weightMatrix) in weights.enumerated() {
            let z = weightMatrix * activations.last! + biases[index]
            print("z: \(z)")
            zs.append(z)
            activations.append(MLVector(z.map { activationFunctions[index].apply($0) }))
        }
        
        // Compute the error in the output layer
        let outputError = lossFunction.gradient(target: target, output: activations.last!)
        print("outputError: \(outputError)")
        
        
        // Backward pass
        var delta = outputError
        var nabla_w = [MLMatrix<Float>]()
        var nabla_b = [MLVector<Float>]()
        
        for layer in (1..<layers.count).reversed() {
            let z = zs[layer-1]
            let sp = MLVector(z.map { activationFunctions[layer-1].derivative($0) })
            delta = (delta * sp)
            nabla_b.insert(delta, at: 0)
            nabla_w.insert(delta.outerProduct(activations[layer-1]), at: 0)
            delta = weights[layer-1].transpose() * delta
            print("delta: \(delta)")
        }
        
        // Update weights and biases using gradient descent
        for i in 0..<weights.count {
            weights[i] = weights[i] - nabla_w[i].scaled(by: learningRate)
            biases[i] = biases[i] - nabla_b[i].scaled(by: learningRate)
        }
    }
    
    public func predict(input: MLVector<Float>) -> MLVector<Float> {
        return forward(input: input)
    }
    
    public func getWeights() -> [MLMatrix<Float>] {
        return weights
    }
}
