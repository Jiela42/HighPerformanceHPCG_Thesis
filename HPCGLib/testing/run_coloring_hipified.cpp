#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "MatrixLib/sparse_CSR_Matrix_hipified.hpp"
#include "MatrixLib/striped_Matrix_hipified.hpp"
#include "MatrixLib/coloring_hipified.cuh"
#include "MatrixLib/generations_hipified.hpp"

void print_COR_Format(int nx, int ny, int nz){

    // make a matrix
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A_csr = problem.first;

    striped_Matrix<double>* A_striped = A_csr.get_Striped();
    std::cout << "getting striped matrix" << std::endl;
    // A_striped.striped_Matrix_from_sparse_CSR(A_csr);

    // make coloring
    A_striped->generate_coloring();
    A_striped->print_COR_Format();

}

void write_coloring_to_file(std::vector<local_int_t> colors, std::string filename){
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

    striped_Matrix<double>* A_striped = A_csr.get_Striped();
    std::cout << "getting striped matrix" << std::endl;

    // A_striped.striped_Matrix_from_sparse_CSR(A_csr);

    // pass matrix to coloring
    std::vector<local_int_t> colors = color_for_forward_pass(*A_striped);


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

void compare_backward_forward_coloring(int nx, int ny, int nz){

    // make a matrix
    std::pair<sparse_CSR_Matrix<double>, std::vector<double>> problem = generate_HPCG_Problem(nx, ny, nz);
    sparse_CSR_Matrix<double> A_csr = problem.first;

    striped_Matrix<double>* A_striped = A_csr.get_Striped();
    std::cout << "getting striped matrix" << std::endl;
    // A_striped.striped_Matrix_from_sparse_CSR(A_csr);

    // pass matrix to coloring
    std::vector<local_int_t> forward_colors = color_for_forward_pass(*A_striped);
    std::vector<local_int_t> backward_colors = color_for_backward_pass(*A_striped);

    for(int i = 0; i < forward_colors.size(); i++){
        local_int_t forward_color = forward_colors[i];
        local_int_t backward_color = backward_colors[i];

        local_int_t backward_color_based_on_forward = forward_colors[forward_colors.size()-1-i];
        if (backward_color != backward_color_based_on_forward){
            std::cout << "For nx = " << nx << " ny = " << ny << " nz = " << nz << " Row " << i << " has forward color " << forward_color << " and backward color " << backward_color << std::endl;
        }
    }

    std::cout << "For nx = " << nx << " ny = " << ny << " nz = " << nz << " Backward Coloring can be computed off of forwar coloring" << std::endl;
}

int main() {
    // print_coloring(4, 4, 4);
    // print_coloring(3, 4, 5);
    // print_coloring(4, 3, 5);
    // print_coloring(5, 4, 3);

    // print_coloring(3,5,6);
    // print_coloring(5,3,6);
    // print_coloring(6,5,3);

    // print_coloring(8, 8, 8);
    // print_coloring(16, 16, 16);
    // print_coloring(24, 24, 24);
    // print_coloring(32, 32, 32);
    // print_coloring(64, 64, 64);

    // compare_backward_forward_coloring(4, 4, 4);
    // compare_backward_forward_coloring(3, 4, 5);
    // compare_backward_forward_coloring(4, 3, 5);
    // compare_backward_forward_coloring(5, 4, 3);
    // compare_backward_forward_coloring(3,5,6);
    // compare_backward_forward_coloring(5,3,6);
    // compare_backward_forward_coloring(6,5,3);
    // compare_backward_forward_coloring(8, 8, 8);
    // compare_backward_forward_coloring(16, 16, 16);
    // compare_backward_forward_coloring(24, 24, 24);
    // compare_backward_forward_coloring(32, 32, 32);
    // compare_backward_forward_coloring(64, 64, 64);

    print_COR_Format(4, 4, 4);

    return 0;
}