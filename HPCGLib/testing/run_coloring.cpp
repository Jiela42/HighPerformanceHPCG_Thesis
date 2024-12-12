#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <MatrixLib/sparse_CSR_Matrix.hpp>
#include "MatrixLib/banded_Matrix.hpp"
#include <MatrixLib/coloring.cuh>
#include "MatrixLib/generations.hpp"

void write_coloring_to_file(std::vector<int> colors, std::string filename){
   std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    std::cout << "Writing to file: " << filename << std::endl;

    for (const int& value : colors) {
        outfile << value << ","; // Write each value followed by a comma
    }
    outfile << std::endl; // End the line after writing all values

    outfile.close();
    if (!outfile) {
        std::cerr << "Error closing file: " << filename << std::endl;
    }
}

void print_coloring(int nx, int ny, int nz){
    
    // make a matrix
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A_csr = problem.first;
    
    banded_Matrix<double> A_banded;
    A_banded.banded_Matrix_from_sparse_CSR(A_csr);   

    // pass matrix to coloring
    std::vector<int> colors = color_for_forward_pass(A_banded);


    // print colors
    // for (int i = 0; i < colors.size(); i++){
    //     int color_i = colors[i];
    //     std::cout << "Row " << i << " has color " << color_i << std::endl;
    // }

    // write coloring to file
    std::string coloring_folder = "../../../colorings/";
    std::string new_csv_file = coloring_folder + "coloring_" + std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz) + ".csv";
    write_coloring_to_file(colors, new_csv_file);
}


int main() {
    // print_coloring(4, 4, 4);
    // print_coloring(3, 4, 5);
    print_coloring(4, 3, 5);
    print_coloring(5, 4, 3);

    print_coloring(3,5,6);
    print_coloring(5,3,6);
    print_coloring(6,5,3);
    
    // print_coloring(8, 8, 8);
    // print_coloring(16, 16, 16);
    // print_coloring(24, 24, 24);
    // print_coloring(32, 32, 32);
    // print_coloring(64, 64, 64);

    return 0;
}