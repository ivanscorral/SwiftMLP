//
//  Extensions.swift
//
//
//  Created by Ivan Sanchez on 26/11/23.
//

import Foundation


extension MLVector: Sequence {
    public typealias Element = T
    public typealias Iterator = IndexingIterator<[T]>
    
    public func makeIterator() -> IndexingIterator<[T]> {
        return grid.makeIterator()
    }
}

extension MLVector {
    init(grid: [T], size: Int) {
        self.grid = grid
        self.size = size
    }
}

extension MLVector where T == Float {
    static func +(lhs: MLVector, rhs: MLVector) -> MLVector {
        precondition(lhs.size == rhs.size, "Vector sizes must match.")
        return MLVector(grid: zip(lhs.grid, rhs.grid).map(+), size: lhs.size)
    }
    
    static func -(lhs: MLVector, rhs: MLVector) -> MLVector {
        precondition(lhs.size == rhs.size, "Vector sizes must match.")
        return MLVector(grid: zip(lhs.grid, rhs.grid).map(-), size: lhs.size)
    }
    
    static func *(lhs: MLVector, rhs: MLVector) -> MLVector {
        precondition(lhs.size == rhs.size, "Vector sizes must match.")
        return MLVector(grid: zip(lhs.grid, rhs.grid).map(*), size: lhs.size)
    }
    
    static func *(lhs: MLVector, rhs: MLMatrix<T>) -> MLVector {
        return rhs.transpose() * lhs
    }
    
    func outerProduct(_ other: MLVector) -> MLMatrix<T> {
        return MLMatrix(rows: self.size, columns: other.size) { row, column in
            self.grid[row] * other.grid[column]
        }
    }
    
    func subtracting(_ other: MLVector) -> MLVector {
        return self - other
    }
    
    func adding(_ other: MLVector) -> MLVector {
        return self + other
    }
    
    func scaled(by scalar: T) -> MLVector {
        MLVector(grid: self.grid.map { $0 * scalar }, size: self.size)
    }
    
    func dot(_ other: MLVector) -> Float {
        precondition(self.size == other.size, "Vector sizes must match.")
        return zip(self.grid, other.grid).map(*).reduce(0, +)
    }
    
    func transpose() -> MLMatrix<T> {
        MLMatrix(rows: 1, columns: self.size) { _, column in
            self.grid[column]
        }
    }
}

extension MLMatrix where T == Float {
    
    init(randomFilledRows rows: Int, columns: Int) {
        self.init(rows: rows, columns: columns) { _, _ in  Float.random(in: -1...1)}
    }
    
    static func -(lhs: MLMatrix, rhs: MLMatrix) -> MLMatrix {
        precondition(lhs.rows == rhs.rows && lhs.columns == rhs.columns, "Matrix sizes must match.")
        return MLMatrix(rows: lhs.rows, columns: lhs.columns) { row, column in
            lhs[row, column] - rhs[row, column]
        }
    }
    
    // Calculate the dot product of a matrix and a vector
    static func *(lhs: MLMatrix, rhs: MLVector<T>) -> MLVector<T> {
        precondition(lhs.columns == rhs.size, "Matrix columns and vector size must match.")
        let vectorArray = rhs.grid
        
        let resultArray: [Float] = lhs.grid.map { row in
            // Calculate the dot product of the row and the vector
            zip(row, vectorArray).reduce(0) { sum, pair in sum + pair.0 * pair.1 }
        }
        
        // Convert the resultArray into an MLVector<Float> before returning
        return MLVector(resultArray)
    }
    
    static func *(lhs: MLMatrix, rhs: T) -> MLMatrix {
        return MLMatrix(rows: lhs.rows, columns: lhs.columns) { row, column in
            lhs[row, column] * rhs
        }
    }
    
    func scaled(by scalar: T) -> MLMatrix {
        return self * scalar
    }
    
    
    func transpose() -> MLMatrix<Float> {
        var transposedMatrix = MLMatrix<Float>(rows: self.columns, columns: self.rows)
        for i in 0..<self.rows {
            for j in 0..<self.columns {
                transposedMatrix[j, i] = self[i, j]
            }
        }
        return transposedMatrix
    }
}

extension MLMatrix {
    // Subscript to access/modify a specific element
    public subscript(row: Int, column: Int) -> T {
        get {
            assert(row >= 0 && row < rows && column >= 0 && column < columns, "Index out of range")
            return grid[row][column]
        }
        set {
            assert(row >= 0 && row < rows && column >= 0 && column < columns, "Index out of range")
            grid[row][column] = newValue
        }
    }

    // Subscript to access/modify a whole row
    public subscript(row: Int) -> [T] {
        get {
            assert(row >= 0 && row < rows, "Row index out of range")
            return grid[row]
        }
        set {
            assert(row >= 0 && row < rows, "Row index out of range")
            assert(newValue.count == columns, "New row must have the same number of columns")
            grid[row] = newValue
        }
    }
}
