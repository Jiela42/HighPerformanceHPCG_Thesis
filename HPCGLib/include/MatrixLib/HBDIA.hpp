// HBDIA.hpp
#ifndef HBDIA_HPP
#define HBDIA_HPP

#include <string>
#include <vector>

template <typename T>
class HBDIA {
    public:
        HBDIA();
        HBDIA(const std::vector<int>& rowIndices, const std::vector<int>& colIndices, const std::vector<T>& values);
        ~HBDIA();
        bool loadMTX(const std::string& filename);
        void print() const;
        void convertToDIAFormat(bool COOisUnique = false);
        bool isDIAFormat() const;
        bool isCOOFormat() const;
        bool isHBDIAFormat() const;
        void printDIA(int block_width = 0) const;
        void printCOO() const;
        void printHBDIA() const;
        void printDense() const;
        void convertToHBDIAFormat(int blockWidth = 32, int threshold = 16, bool COOisUnique = false);
        
        // Format deletion methods
        void deleteCOOFormat();
        void deleteDIAFormat();
        void deleteHBDIAFormat();
        void deleteMatrix();
        
        // Helper method to remove duplicates from COO format
        void removeCOODuplicates();
        
    private:
        // Coordinate format storage
        std::vector<T> values;
        std::vector<int> rowIndices;
        std::vector<int> colIndices;
        
        // DIA format storage
        std::vector<std::vector<T>> diagonals;
        std::vector<int> offsets;
        
        // HBDIA format storage
        std::vector<T> hbdiaData;                    // Contiguous memory for all blocks
        std::vector<T*> ptrToBlock;                  // Pointers to each block's data
        std::vector<std::vector<int>> offsetsPerBlock; // Offsets for each block
        std::vector<int> cpuRowIndices;              // CPU fallback rows
        std::vector<int> cpuColIndices;              // CPU fallback cols
        std::vector<T> cpuValues;                    // CPU fallback values
        int blockWidth;                              // Block width for HBDIA
        int threshold;                               // Threshold for CPU fallback
        
        // Matrix metadata
        int numRows;
        int numCols;
        int numNonZeros;
        
        // Format flags
        bool hasCOO;   // Flag to track if COO format is available
        bool hasDIA;   // Flag to track if DIA format is available
        bool hasHBDIA; // Flag to track if HBDIA format is available
};

#endif // HBDIA_HPP
