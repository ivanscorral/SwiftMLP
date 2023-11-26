//
//  MLTypes.swift
//
//
//  Created by Ivan Sanchez on 26/11/23.
//

import Foundation


// Matrix

public typealias MLMatrix<T> = [[T]] where T: FloatingPoint

// Generic Vector

public typealias MLVector<T> = [T] where T: FloatingPoint


func matrixVectorMultiply<T>(_ matrix: MLMatrix<T>, _ vector: MLVector<T>) -> MLVector<T> where T: FloatingPoint {
    precondition(matrix[0].count == vector.count, "Matrix and vector must match dimensions")
    precondition(matrix.count > 0, "Matrix must have at least one row")
    return matrix.map { row in
        zip(row, vector).reduce(0, { sum, pair in sum + pair.0 * pair.1 })
    }
}
