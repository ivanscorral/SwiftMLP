//
//  ActivationFunction.swift
//
//
//  Created by Ivan Sanchez on 26/11/23.
//

import Foundation
    

enum ActivationFunction {
    case sigmoid
    case relu
    
    func apply(_ x: Float) -> Float {
        switch self {
        case .sigmoid:
            return 1.0 / (1.0 + exp(-x))
        case .relu:
            return max(0.0, x)
        }
    }
    
    func derivative(_ x: Float) -> Float {
        switch self {
        case .sigmoid:
            return x * (1.0 - x)
        case .relu:
            return x > 0.0 ? 1.0 : 0.0
        }
    }
}
