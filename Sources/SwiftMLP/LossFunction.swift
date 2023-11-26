//
//  LossFunction.swift
//
//
//  Created by Ivan Sanchez on 26/11/23.
//

import Foundation


public protocol LossFunction {
    func compute(target: MLVector<Float>, output: MLVector<Float>) -> Float
    func gradient(target: MLVector<Float>, output: MLVector<Float>) -> MLVector<Float>
}


public struct MeanSquaredErrorLoss: LossFunction {

    public init() {}

    public func compute(target: MLVector<Float>, output: MLVector<Float>) -> Float {
        let error = zip(target, output).map { $0 - $1 }
        return (error.map { $0 * $0 }.reduce(0, +)) / Float(error.count)
    }

    public func gradient(target: MLVector<Float>, output: MLVector<Float>) -> MLVector<Float> {
        precondition(target.size == output.size, "`target` and `output` must have the same size")
        return MLVector<Float>(zip(target, output).map { 2 * ($1 - $0) / Float(target.size) })
    }
}
