#include "UtilLib/utils.hpp"

bool double_compare(double a, double b){
    return abs(a - b) < error_tolerance;
}

bool vector_compare(const std::vector<double>& a, const std::vector<double>& b){
    if (a.size() != b.size()){
        std::cout << "Vector sizes do not match" << std::endl;
        return false;
    }
    for (int i = 0; i < a.size(); i++){
        if (not double_compare (a[i], b[i])){
            std::cout << "Error at index " << i << " should be " << a[i] << " but was " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

bool vector_compare(const std::vector<int>& a, const std::vector<int>& b, std::string info){
    bool test_result = true;
    if (a.size() != b.size()){
        std::cout << "Vector sizes do not match for " << info << " size a = " << a.size() << " size b = " << b.size() << std::endl;
         test_result = false;
    }

    int fault_ctr = 0;

    for(int i = 0; i < a.size(); i++){
        if (a[i] != b[i] && fault_ctr < 10){
            std::cout << "Error at index " << i << " should be " << a[i] << " but was " << b[i] << " for " << info << std::endl;
            test_result = false;
            fault_ctr++;
        }
    }
    return test_result;
}