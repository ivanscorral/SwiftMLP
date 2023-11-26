//
//  MLTypes.swift
//
//
//  Created by Ivan Sanchez on 26/11/23.
//

import Foundation


// Matrix

public struct MLMatrix<T> where T: FloatingPoint {
    var grid: [[T]]
    let rows: Int
    let columns: Int
    
    public init(rows: Int, columns: Int) {
        self.rows = rows
        self.columns = columns
        
        self.grid = (0..<rows).map { _ in
            [T](repeating: 0, count: columns)
        }
    }
    
    public init(rows: Int, columns: Int, fill: (Int, Int) -> T) {
        self.rows = rows
        self.columns = columns
        
        self.grid = (0..<rows).map { row in
            (0..<columns).map { column in
                fill(row, column)
            }
        }
    }
}



// Generic Vector

public struct MLVector<T> where T: FloatingPoint {
    var grid: [T]
    let size: Int
    
    public init(repeating: T, count: Int) {
        self.size = count
        self.grid = [T](repeating: repeating, count: count)
    }
    
    public init(_ array: [T]) {
        self.grid = array
        self.size = array.count
    }
}
