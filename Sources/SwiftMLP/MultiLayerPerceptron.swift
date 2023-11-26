//
//  MultiLayerPerceptron.swift
//
//
//  Created by Ivan Sanchez on 26/11/23.
//

import Foundation


class MultiLayerPerceptron {
    var inputSize: Int
    var hiddenSize: Int
    var outputSize: Int
    var learningRate: Float
    
    var activationFunction: ActivationFunction
    
    var weightsInputHidden: MLMatrix<Float>
    var weightsHiddenOutput: MLMatrix<Float>
    
    init(inputSize: Int, hiddenSize: Int, outputSize: Int, learningRate: Float, activationFunction: ActivationFunction, weightsInputHidden: MLMatrix<Float>, weightsHiddenOutput: MLMatrix<Float>) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.learningRate = learningRate
        self.activationFunction = activationFunction
        
        self.weightsInputHidden = (0..<inputSize).map { _ in (0..<hiddenSize).map { _ in Float.random(in: -1...1) } }
        self.weightsHiddenOutput = (0..<hiddenSize).map { _ in (0..<outputSize).map { _ in Float.random(in: -1...1) } }
    }
    
    func forward(input: MLVector<Float>) -> MLVector<Float> {
        let hiddenInput = dot(input, weightsInputHidden)
        let hiddenOutput = hiddenInput.map { activationFunction.apply($0) }
        
        let finalInput = dot(hiddenOutput, weightsHiddenOutput)
        let finalOutput = finalInput.map { activationFunction.apply($0) }
        
        return finalOutput
    }
    
    
    func train(input: MLVector<Float>, target: MLVector<Float>) {
          // Forward pass
          let hiddenInput = dot(input, weightsInputHidden)
          let hiddenOutput = hiddenInput.map { activationFunction.apply($0) }
          
          let finalInput = dot(hiddenOutput, weightsHiddenOutput)
          let finalOutput = finalInput.map { activationFunction.apply($0) }
          
          // Backward pass
          let outputErrors = zip(target, finalOutput).map(-)
          let hiddenErrors = dot(outputErrors, transpose(MLMatrix: weightsHiddenOutput))
          
          // Update weights
          for i in 0..<hiddenSize {
              for j in 0..<outputSize {
                  let delta = learningRate * outputErrors[j] * activationFunction.derivative(finalOutput[j])
                  weightsHiddenOutput[i][j] += delta * hiddenOutput[i]
              }
          }
          
          for i in 0..<inputSize {
              for j in 0..<hiddenSize {
                  let delta = learningRate * hiddenErrors[j] * activationFunction.derivative(hiddenOutput[j])
                  weightsInputHidden[i][j] += delta * input[i]
              }
          }
      }
    
    func transpose(MLMatrix: MLMatrix<Float>) -> MLMatrix<Float> {
        guard let firstRow = MLMatrix.first else { return [] }
        return firstRow.indices.map { idx in
            MLMatrix.map { $0[idx] }
        }
    }
    
    func dot(_ vec: MLVector<Float>, _ MLMatrix: MLMatrix<Float>) -> MLVector<Float> {
        return MLMatrix.indices.map { j in
            vec.indices.reduce(0) { sum, i in
                sum + vec[i] * MLMatrix[i][j]
            }
        }
    }
}
