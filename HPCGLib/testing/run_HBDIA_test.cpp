// run_HBDIA.cpp
#include "MatrixLib/HBDIA.hpp"
#include "UtilLib/types.hpp"
#include <iostream>
#include <string>

int main(int argc, char *argv[]){

    // Test 1: Create a small matrix and convert to DIA format internally
    std::vector<int> rows = {0, 0, 1, 1, 1, 2, 2};
    std::vector<int> cols = {0, 2, 0, 1, 2, 1, 2};
    std::vector<DataType> vals = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    
    HBDIA<DataType> matrix1(rows, cols, vals);
    std::cout << "=== Test 1: Small Matrix ===" << std::endl;
    matrix1.printDense();
    
    std::cout << "Before conversion:" << std::endl;
    matrix1.printCOO();
    
    std::cout << "\nIs DIA format? " << (matrix1.isDIAFormat() ? "Yes" : "No") << std::endl;
    
    matrix1.convertToDIAFormat();
    
    std::cout << "\nAfter conversion:" << std::endl;
    matrix1.printDIA();
    
    std::cout << "Is DIA format? " << (matrix1.isDIAFormat() ? "Yes" : "No") << std::endl;
    
    // Test 2: Convert to HBDIA format
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "=== Test 2: HBDIA Conversion (Small Matrix) ===" << std::endl;
    
    // Create a new matrix for HBDIA test
    HBDIA<DataType> matrix2(rows, cols, vals);
    matrix2.printDense();
    
    std::cout << "Before HBDIA conversion:" << std::endl;
    matrix2.printCOO();
    matrix2.convertToDIAFormat();
    matrix2.printDIA(4);
    
    matrix2.convertToHBDIAFormat(4, 2);  // Block width=4, threshold=2
    
    std::cout << "\nAfter HBDIA conversion:" << std::endl;
    matrix2.printHBDIA();
    
    // Test 3: Larger matrix for HBDIA (20x20)
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "=== Test 3: HBDIA with Larger Matrix (20x20) ===" << std::endl;
    
    // Create a 20x20 matrix with various diagonal patterns
    std::vector<int> large_rows, large_cols;
    std::vector<DataType> large_vals;
    
    // Add main diagonal
    for (int i = 0; i < 20; ++i) {
        large_rows.push_back(i);
        large_cols.push_back(i);
        large_vals.push_back(5.0 + i * 0.1);
    }
    
    // Add first super-diagonal (offset +1)
    for (int i = 0; i < 19; ++i) {
        large_rows.push_back(i);
        large_cols.push_back(i + 1);
        large_vals.push_back(2.0 + i * 0.05);
    }
    
    // Add first sub-diagonal (offset -1)
    for (int i = 1; i < 20; ++i) {
        large_rows.push_back(i);
        large_cols.push_back(i - 1);
        large_vals.push_back(1.0 + i * 0.05);
    }
    
    // Add some sparse entries (second super-diagonal with gaps)
    for (int i = 0; i < 18; i += 3) {  // Only every 3rd element
        large_rows.push_back(i);
        large_cols.push_back(i + 2);
        large_vals.push_back(0.5 + i * 0.02);
    }
    
    // Add some random sparse entries
    int sparse_entries[][2] = {{2, 15}, {5, 18}, {10, 3}, {15, 8}, {18, 19}};
    for (auto& entry : sparse_entries) {
        large_rows.push_back(entry[0]);
        large_cols.push_back(entry[1]);
        large_vals.push_back(0.3);
    }
    
    HBDIA<DataType> matrix3(large_rows, large_cols, large_vals);
    matrix3.printDense();
    
    std::cout << "Large matrix in COO:" << std::endl;
    matrix3.printCOO();

    matrix3.convertToDIAFormat();
    std::cout << "Large matrix in DIA:" << std::endl;
    matrix3.printDIA(8);
    
    matrix3.convertToHBDIAFormat(8, 3);  // Block width=8, threshold=5
    std::cout << "\nLarge matrix after HBDIA conversion:" << std::endl;
    matrix3.printHBDIA();
    
    // Test 4: Load larger matrix from file and convert
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "=== Test 4: File Matrix ===" << std::endl;
    HBDIA<DataType> matrix4;
    std::string filename = "/users/nrottstegge/SuiteSparseMatrixCollection/dwt_59/dwt_59.mtx";
    
    if (matrix4.loadMTX(filename)) {
        matrix4.printDense();
        
        std::cout << "\nBefore conversion:" << std::endl;
        matrix4.printCOO();
        
        std::cout << "\nConverting to DIA format..." << std::endl;
        matrix4.convertToDIAFormat();
        
        std::cout << "\nAfter conversion:" << std::endl;
        matrix4.printDIA(8);

        std::cout << "\nConverting to HBDIA format..." << std::endl;
        matrix4.convertToHBDIAFormat(8, 5);  // Block width=8, threshold=5

        std::cout << "\nAfter HBDIA conversion:" << std::endl;
        matrix4.printHBDIA();
    } else {
        std::cout << "Failed to load matrix file!" << std::endl;
    }
    
    return 0;
}