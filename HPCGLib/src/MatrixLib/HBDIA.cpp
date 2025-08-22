// HBDIA.cpp
#include "MatrixLib/HBDIA.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <set>
#include <map>
#include <iomanip> // For std::setw, std::fixed, std::setprecision

template <typename T>
HBDIA<T>::HBDIA() : numRows(0), numCols(0), numNonZeros(0), hasCOO(false), hasDIA(false), hasHBDIA(false) {}

template <typename T>
HBDIA<T>::HBDIA(const std::vector<int>& rowIndices, const std::vector<int>& colIndices, const std::vector<T>& values) 
    : values(values), rowIndices(rowIndices), colIndices(colIndices), hasCOO(true), hasDIA(false), hasHBDIA(false) {
    
    // Validate that all vectors have the same size
    if (rowIndices.size() != colIndices.size() || rowIndices.size() != values.size()) {
        throw std::invalid_argument("Row indices, column indices, and values vectors must have the same size");
    }
    
    numNonZeros = static_cast<int>(values.size());
    
    // Calculate matrix dimensions by finding max indices
    numRows = 0;
    numCols = 0;
    
    for (int row : rowIndices) {
        if (row >= numRows) {
            numRows = row + 1;  // +1 because indices are 0-based
        }
    }
    
    for (int col : colIndices) {
        if (col >= numCols) {
            numCols = col + 1;  // +1 because indices are 0-based
        }
    }
}

template <typename T>
HBDIA<T>::~HBDIA() {}

template <typename T>
bool HBDIA<T>::loadMTX(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    std::string line;
    bool isSymmetric = false;
    
    // Parse header and check for symmetry
    while (std::getline(file, line)) {
        if (line[0] == '%') {
            if (line.find("symmetric") != std::string::npos) {
                isSymmetric = true;
                std::cout << "Detected symmetric matrix - will expand to full matrix" << std::endl;
            }
        } else {
            break; // First non-comment line contains dimensions
        }
    }
    
    // Parse header line (rows, cols, non-zeros)
    std::istringstream headerStream(line);
    headerStream >> numRows >> numCols >> numNonZeros;
    
    // Reserve space for vectors (more space if symmetric)
    int expectedEntries = isSymmetric ? numNonZeros * 2 : numNonZeros;
    values.reserve(expectedEntries);
    rowIndices.reserve(expectedEntries);
    colIndices.reserve(expectedEntries);
    
    // Read matrix entries
    int row, col;
    T value;
    int originalEntries = 0;
    
    while (std::getline(file, line) && originalEntries < numNonZeros) {
        std::istringstream entryStream(line);
        
        // For pattern matrices, there might be no value column
        if (entryStream >> row >> col) {
            if (!(entryStream >> value)) {
                value = T(1); // Default value for pattern matrices
            }
            
            // Convert to 0-based indexing (MTX format is 1-based)
            row -= 1;
            col -= 1;
            
            // Add the original entry
            rowIndices.push_back(row);
            colIndices.push_back(col);
            values.push_back(value);
            originalEntries++;
            
            // If symmetric and not on diagonal, add mirrored entry
            if (isSymmetric && row != col) {
                rowIndices.push_back(col);
                colIndices.push_back(row);
                values.push_back(value);
            }
        }
    }
    
    file.close();
    
    // Update numNonZeros to reflect actual entries stored
    numNonZeros = values.size();
    hasCOO = true;  // Set COO format flag
    
    std::cout << "Loaded " << originalEntries << " entries from file";
    if (isSymmetric) {
        std::cout << ", expanded to " << numNonZeros << " total entries";
    }
    std::cout << std::endl;
    
    return originalEntries > 0;
}

template <typename T>
void HBDIA<T>::removeCOODuplicates() {
    if (!hasCOO) {
        std::cout << "No COO format available for duplicate removal" << std::endl;
        return;
    }
    
    if (values.empty()) {
        return; // Empty matrix
    }
    
    // Step 1: Create map to consolidate duplicates by summing them up
    std::map<std::pair<int, int>, T> uniqueEntries;
    int originalEntries = values.size();
    
    for (size_t i = 0; i < values.size(); ++i) {
        std::pair<int, int> position(rowIndices[i], colIndices[i]);
        uniqueEntries[position] += values[i];
    }
    
    // Step 2: Replace COO vectors with deduplicated data
    values.clear();
    rowIndices.clear();
    colIndices.clear();
    
    values.reserve(uniqueEntries.size());
    rowIndices.reserve(uniqueEntries.size());
    colIndices.reserve(uniqueEntries.size());
    
    for (const auto& entry : uniqueEntries) {
        rowIndices.push_back(entry.first.first);
        colIndices.push_back(entry.first.second);
        values.push_back(entry.second);
    }
    
    // Update numNonZeros to reflect deduplicated count
    numNonZeros = values.size();
    
    std::cout << "Consolidated " << originalEntries << " entries to " << uniqueEntries.size() 
              << " unique entries (removed " << (originalEntries - uniqueEntries.size()) << " duplicates)" << std::endl;
}

template <typename T>
void HBDIA<T>::convertToDIAFormat(bool COOisUnique) {
    if (hasDIA) {
        std::cout << "Matrix is already in DIA format" << std::endl;
        return;
    }
    
    if (!hasCOO) {
        std::cout << "No COO format available for conversion to DIA" << std::endl;
        return;
    }
    
    if (numNonZeros == 0) {
        hasDIA = true;
        return; // Empty matrix
    }
    
    std::cout << "Converting matrix from coordinate format to DIA format..." << std::endl;
    
    // Step 1: Remove duplicates from COO format if not already unique
    if (!COOisUnique) {
        removeCOODuplicates();
    }
    
    // Step 2: Find all unique diagonal offsets
    std::set<int> uniqueOffsets;
    for (size_t i = 0; i < values.size(); ++i) {
        int offset = colIndices[i] - rowIndices[i]; // col - row
        uniqueOffsets.insert(offset);
    }
    
    // Step 3: Convert set to sorted vector of offsets
    offsets.assign(uniqueOffsets.begin(), uniqueOffsets.end());
    int numDiagonals = offsets.size();
    
    // Step 4: Initialize diagonals matrix
    diagonals.resize(numDiagonals);
    
    int diagLength = std::min(numRows, numCols);
    
    for (int d = 0; d < numDiagonals; ++d) {
        int offset = offsets[d];
        
        // Initialize diagonal with zeros
        diagonals[d].assign(diagLength, T(0));
    }
    
    // Step 5: Fill in the diagonal values from COO data
    for (size_t i = 0; i < values.size(); ++i) {
        int row = rowIndices[i];
        int col = colIndices[i];
        int offset = col - row;
        T value = values[i];
        
        // Find which diagonal this belongs to
        auto it = std::lower_bound(offsets.begin(), offsets.end(), offset);
        int diagIndex = std::distance(offsets.begin(), it);
        
        // Calculate position in the diagonal
        int diagPos = col;
        
        // Store the value
        if (diagPos >= 0 && diagPos < static_cast<int>(diagonals[diagIndex].size())) {
            diagonals[diagIndex][diagPos] = value;
        }
    }
    
    // Step 6: Set flag to indicate DIA format is available
    hasDIA = true;
    
    std::cout << "Conversion complete. Matrix now stored in DIA format with " 
              << numDiagonals << " diagonals." << std::endl;
}

template <typename T>
void HBDIA<T>::convertToHBDIAFormat(int blockWidth, int threshold, bool COOisUnique) {
    if (hasHBDIA) {
        std::cout << "Matrix is already in HBDIA format" << std::endl;
        return;
    }
    
    if (!hasCOO) {
        std::cout << "No COO format available for conversion to HBDIA" << std::endl;
        return;
    }
    
    if (values.empty()) {
        std::cout << "Cannot convert empty matrix to HBDIA format" << std::endl;
        return;
    }
    
    this->blockWidth = blockWidth;
    this->threshold = threshold;
    
    std::cout << "Converting to HBDIA format with block_width=" << blockWidth 
              << " and threshold=" << threshold << std::endl;
    
    // Step 1: Remove duplicates from COO format if not already unique
    if (!COOisUnique) {
        removeCOODuplicates();
    }
    
    // Step 2: Calculate maximum number of blocks
    int maxNumBlocks = (numCols + blockWidth - 1) / blockWidth;
    
    // Step 3: Count offsets per block using COO data
    std::vector<std::map<int, int>> offsetCountPerBlock(maxNumBlocks);
    
    for (size_t i = 0; i < values.size(); ++i) {
        int r = rowIndices[i];
        int c = colIndices[i];
        int block = c / blockWidth;
        int offset = c - r;  // col - row: positive for super-diagonals, negative for sub-diagonals
        
        if (block < maxNumBlocks) {
            offsetCountPerBlock[block][offset]++;
        }
    }
    
    // Step 4: Calculate storage requirements and filter offsets
    int storageRequired = 0;
    std::vector<std::vector<int>> validOffsetsPerBlock(maxNumBlocks);
    
    for (int b = 0; b < maxNumBlocks; ++b) {
        for (const auto& pair : offsetCountPerBlock[b]) {
            if (pair.second >= threshold) {
                validOffsetsPerBlock[b].push_back(pair.first);
                storageRequired += blockWidth;
            }
        }
        // Sort offsets for consistent ordering
        std::sort(validOffsetsPerBlock[b].begin(), validOffsetsPerBlock[b].end());
    }
    
    std::cout << "Storage required: " << storageRequired << " elements" << std::endl;
    
    // Step 5: Allocate contiguous memory and set up pointers
    hbdiaData.resize(storageRequired);
    ptrToBlock.resize(maxNumBlocks, nullptr);
    offsetsPerBlock = std::move(validOffsetsPerBlock);
    
    T* dataPtr = hbdiaData.data();
    
    for (int b = 0; b < maxNumBlocks; ++b) {
        if (!offsetsPerBlock[b].empty()) {
            ptrToBlock[b] = dataPtr;
            dataPtr += offsetsPerBlock[b].size() * blockWidth;
        }
    }
    
    // Step 6: Initialize data to zero
    std::fill(hbdiaData.begin(), hbdiaData.end(), T(0));
    
    // Step 7: Fill data and separate CPU fallback entries
    cpuRowIndices.clear();
    cpuColIndices.clear();
    cpuValues.clear();
    
    for (size_t i = 0; i < values.size(); ++i) {
        int r = rowIndices[i];
        int c = colIndices[i];
        T v = values[i];
        
        int block = c / blockWidth;
        int offset = c - r;  // col - row: positive for super-diagonals, negative for sub-diagonals
        int lane = c % blockWidth;
        
        if (block < maxNumBlocks && !offsetsPerBlock[block].empty()) {
            // Find offset index in this block
            auto it = std::find(offsetsPerBlock[block].begin(), offsetsPerBlock[block].end(), offset);
            
            if (it != offsetsPerBlock[block].end()) {
                int offsetIndex = std::distance(offsetsPerBlock[block].begin(), it);
                int dataIndex = offsetIndex * blockWidth + lane;
                ptrToBlock[block][dataIndex] = v;
            } else {
                // Offset not stored in GPU format, use CPU fallback
                cpuRowIndices.push_back(r);
                cpuColIndices.push_back(c);
                cpuValues.push_back(v);
            }
        } else {
            // Block doesn't exist or is empty, use CPU fallback
            cpuRowIndices.push_back(r);
            cpuColIndices.push_back(c);
            cpuValues.push_back(v);
        }
    }
    
    std::cout << "HBDIA conversion complete:" << std::endl;
    std::cout << "  GPU storage: " << (values.size() - cpuValues.size()) << " entries" << std::endl;
    std::cout << "  CPU fallback: " << cpuValues.size() << " entries" << std::endl;
    std::cout << "  Active blocks: " << std::count_if(ptrToBlock.begin(), ptrToBlock.end(), 
                                                      [](T* ptr) { return ptr != nullptr; }) << "/" << maxNumBlocks << std::endl;
    
    // Set flag to indicate HBDIA format is available
    hasHBDIA = true;
}

template <typename T>
bool HBDIA<T>::isDIAFormat() const {
    return hasDIA;
}

template <typename T>
bool HBDIA<T>::isCOOFormat() const {
    return hasCOO;
}

template <typename T>
bool HBDIA<T>::isHBDIAFormat() const {
    return hasHBDIA;
}

template <typename T>
void HBDIA<T>::print() const {
    std::vector<std::string> formats;

    if (hasCOO)   formats.push_back("COO");
    if (hasHBDIA) formats.push_back("HBDIA");
    if (hasDIA)   formats.push_back("DIA");

    std::cout << "Matrix size: " << numRows << " x " << numCols
              << " with " << numNonZeros << " non-zeros" << std::endl;

    if (!formats.empty()) {
        std::cout << "Storage formats: ";
        for (size_t i = 0; i < formats.size(); ++i) {
            std::cout << formats[i];
            if (i + 1 < formats.size()) std::cout << ", ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "No storage format available" << std::endl;
        return;
    }

    // Print dense representation
    printDense();

    // Print all available formats
    if (hasCOO) {
        std::cout << "Printing: COO" << std::endl;
        printCOO();
    }

    if (hasHBDIA) {
        std::cout << "Printing: HBDIA" << std::endl;
        printHBDIA();
    }

    if (hasDIA) {
        std::cout << "Printing: DIA" << std::endl;
        printDIA();
    }
}

template <typename T>
void HBDIA<T>::printCOO() const {
    if (!hasCOO) {
        std::cout << "Matrix is not in COO format or is empty." << std::endl;
        return;
    }
    
    if (values.empty()) {
        std::cout << "Empty matrix" << std::endl;
        return;
    }
    
    std::cout << "\nCOO Format Storage:" << std::endl;
    std::cout << "Number of entries: " << values.size() << std::endl;
    std::cout << "Storage vectors:" << std::endl;
    std::cout << "  rowIndices: " << rowIndices.size() << " elements" << std::endl;
    std::cout << "  colIndices: " << colIndices.size() << " elements" << std::endl;
    std::cout << "  values: " << values.size() << " elements" << std::endl;
    
    // Show first 10 entries
    std::cout << "\nFirst " << std::min(static_cast<size_t>(10), values.size()) << " entries:" << std::endl;
    std::cout << "Row\tCol\tValue" << std::endl;
    std::cout << "---\t---\t-----" << std::endl;
    
    size_t printCount = std::min(static_cast<size_t>(10), values.size());
    for (size_t i = 0; i < printCount; ++i) {
        std::cout << rowIndices[i] << "\t" << colIndices[i] << "\t";
        if (values[i] == T(0)) {
            std::cout << "." << std::endl;
        } else {
            std::cout << std::fixed << std::setprecision(6) << values[i] << std::endl;
        }
    }
    
    if (values.size() > 10) {
        std::cout << "... and " << (values.size() - 10) << " more entries" << std::endl;
    }
    
    std::cout << std::endl;
}

template <typename T>
void HBDIA<T>::printDIA(int block_width) const {
    if (!hasDIA) {
        std::cout << "Matrix is not in DIA format. Call convertToDIAFormat() first." << std::endl;
        return;
    }
    
    if (diagonals.empty()) {
        std::cout << "Empty DIA matrix" << std::endl;
        return;
    }
    
    std::cout << "\nDIA Format Storage:" << std::endl;
    std::cout << "Number of diagonals: " << diagonals.size() << std::endl;
    std::cout << "Matrix size: " << numRows << " x " << numCols << std::endl;
    
    if (block_width > 0) {
        std::cout << "Block width: " << block_width << std::endl;
    }
    
    std::cout << "\nDiagonal storage details:" << std::endl;
    
    // Calculate the maximum width needed for any value to ensure proper alignment
    int maxWidth = 4; // Minimum width for "0.00"
    for (size_t i = 0; i < diagonals.size(); ++i) {
        for (size_t j = 0; j < diagonals[i].size(); ++j) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << diagonals[i][j];
            maxWidth = std::max(maxWidth, static_cast<int>(oss.str().length()));
        }
    }
    
    // Print each diagonal vector
    for (size_t i = 0; i < diagonals.size(); ++i) {
        std::cout << "Diagonal " << i << " (offset " << offsets[i] << ", length " 
                  << diagonals[i].size() << "): \t\t";
        
        if (block_width > 0) {
            // Print in blocked format with separators
            for (size_t j = 0; j < diagonals[i].size(); ++j) {
                if (j > 0 && j % block_width == 0) {
                    std::cout << " | ";
                }
                
                if (diagonals.size() > 20) {
                    std::cout << std::setw(maxWidth) << (diagonals[i][j] != 0 ? "*" : ".");
                } else {
                    if (diagonals[i][j] == T(0)) {
                        std::cout << std::setw(maxWidth) << ".";
                    } else {
                        std::cout << std::setw(maxWidth) << std::fixed << std::setprecision(2) << diagonals[i][j];
                    }
                }
                
                if (j < diagonals[i].size() - 1 && (j + 1) % block_width != 0) {
                    std::cout << " ";
                }
            }
        } else {
            // Print normally without blocking
            for (size_t j = 0; j < diagonals[i].size(); ++j) {
                if (diagonals.size() > 20) {
                    std::cout << std::setw(maxWidth) << (diagonals[i][j] != 0 ? "*" : ".");
                } else {
                    if (diagonals[i][j] == T(0)) {
                        std::cout << std::setw(maxWidth) << ".";
                    } else {
                        std::cout << std::setw(maxWidth) << std::fixed << std::setprecision(2) << diagonals[i][j];
                    }
                }
                if (j < diagonals[i].size() - 1) {
                    std::cout << " ";
                }
            }
        }
        std::cout << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Offset explanation:" << std::endl;
    std::cout << "  offset = 0:  Main diagonal" << std::endl;
    std::cout << "  offset > 0:  Super-diagonals (above main)" << std::endl;
    std::cout << "  offset < 0:  Sub-diagonals (below main)" << std::endl;
}

template <typename T>
void HBDIA<T>::printHBDIA() const {
    if (!hasHBDIA) {
        std::cout << "Matrix is not in HBDIA format. Call convertToHBDIAFormat() first." << std::endl;
        return;
    }
    
    std::cout << "\nHBDIA Format Storage:" << std::endl;
    std::cout << "Block width: " << blockWidth << std::endl;
    std::cout << "Threshold: " << threshold << std::endl;
    std::cout << "Total data size: " << hbdiaData.size() << " elements" << std::endl;
    std::cout << "CPU fallback entries: " << cpuValues.size() << std::endl;

    // Compute max width for aligned printing
    int maxWidth = 4;
    for (size_t b = 0; b < ptrToBlock.size(); ++b) {
        if (ptrToBlock[b] != nullptr && !offsetsPerBlock[b].empty()) {
            for (size_t oi = 0; oi < offsetsPerBlock[b].size() && oi < 3; ++oi) {
                for (int lane = 0; lane < std::min(blockWidth, 8); ++lane) {
                    int dataIndex = oi * blockWidth + lane;
                    T value = ptrToBlock[b][dataIndex];
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2) << value;
                    maxWidth = std::max(maxWidth, static_cast<int>(oss.str().length()));
                }
            }
        }
    }
    
    std::cout << "\nBlock storage details:" << std::endl;
    int activeBlocks = 0;
    for (size_t b = 0; b < ptrToBlock.size(); ++b) {
        if (ptrToBlock[b] != nullptr && !offsetsPerBlock[b].empty()) {
            activeBlocks++;
            std::cout << "\nBlock " << b << " (columns " << b * blockWidth 
                      << "-" << (b + 1) * blockWidth - 1 << "):" << std::endl;
            std::cout << "  Offsets: \t";
            for (int offset : offsetsPerBlock[b]) {
                std::cout << offset << " ";
            }
            std::cout << std::endl;
            
            // Show first few values of each offset in this block
            for (size_t oi = 0; oi < offsetsPerBlock[b].size() && oi < 10; ++oi) {
                std::cout << "  Offset " << offsetsPerBlock[b][oi] << ": \t\t";
                for (int lane = 0; lane < std::min(blockWidth, 8); ++lane) {
                    int dataIndex = oi * blockWidth + lane;
                    T value = ptrToBlock[b][dataIndex];
                    if (value == T(0)) {
                        std::cout << std::setw(maxWidth) << ".";
                    } else {
                        std::cout << std::setw(maxWidth) << std::fixed << std::setprecision(2) << value;
                    }
                    if (lane < std::min(blockWidth, 8) - 1) {
                        std::cout << " ";
                    }
                }
                if (blockWidth > 8) std::cout << "...";
                std::cout << std::endl;
            }
            if (offsetsPerBlock[b].size() > 10) {
                std::cout << "  ... and " << (offsetsPerBlock[b].size() - 3) << " more offsets" << std::endl;
            }
        }
    }
    
    std::cout << "\nSummary: " << activeBlocks << " active blocks out of " << ptrToBlock.size() << std::endl;
    
    if (!cpuValues.empty()) {
        std::cout << "\nCPU fallback entries (first 10):" << std::endl;
        std::cout << "Row\tCol\tValue" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(10), cpuValues.size()); ++i) {
            std::cout << cpuRowIndices[i] << "\t" << cpuColIndices[i] << "\t";
            if (cpuValues[i] == T(0)) {
                std::cout << "." << std::endl;
            } else {
                std::cout << std::fixed << std::setprecision(6) << cpuValues[i] << std::endl;
            }
        }
        if (cpuValues.size() > 10) {
            std::cout << "... and " << (cpuValues.size() - static_cast<size_t>(10)) << " more CPU entries" << std::endl;
        }
    }
}

template <typename T>
void HBDIA<T>::printDense() const {
    if (!hasCOO && !hasDIA && !hasHBDIA) {
        std::cout << "No matrix data available to display." << std::endl;
        return;
    }
    
    std::cout << "\nDense Matrix Visualization (ASCII art):" << std::endl;
    std::cout << "Matrix size: " << numRows << " x " << numCols << std::endl;
    
    if (numRows > 100 || numCols > 100) {
        std::cout << "Matrix too large for ASCII art display (max 100x100). Showing structure only." << std::endl;
        std::cout << "Non-zero density: " << static_cast<double>(numNonZeros) / (numRows * numCols) * 100.0 << "%" << std::endl;
        return;
    }
    
    // Create a dense matrix representation initialized to zero
    std::vector<std::vector<T>> denseMatrix(numRows, std::vector<T>(numCols, T(0)));
    
    // Reconstruct from available format (prioritize order: COO, DIA, HBDIA)
    if (hasCOO) {
        // Reconstruct from COO format
        for (size_t i = 0; i < values.size(); ++i) {
            if (rowIndices[i] >= 0 && rowIndices[i] < numRows && 
                colIndices[i] >= 0 && colIndices[i] < numCols) {
                denseMatrix[rowIndices[i]][colIndices[i]] = values[i];
            }
        }
    } else if (hasDIA) {
        // Reconstruct from DIA format
        for (size_t d = 0; d < diagonals.size(); ++d) {
            int offset = offsets[d];
            for (size_t pos = 0; pos < diagonals[d].size(); ++pos) {
                int row, col;
                if (offset >= 0) {
                    row = static_cast<int>(pos);
                    col = row + offset;
                } else {
                    col = static_cast<int>(pos);
                    row = col - offset;
                }
                
                if (row >= 0 && row < numRows && col >= 0 && col < numCols && diagonals[d][pos] != T(0)) {
                    denseMatrix[row][col] = diagonals[d][pos];
                }
            }
        }
    } else if (hasHBDIA) {
        // Reconstruct from HBDIA format
        for (size_t b = 0; b < ptrToBlock.size(); ++b) {
            if (ptrToBlock[b] != nullptr && !offsetsPerBlock[b].empty()) {
                for (size_t oi = 0; oi < offsetsPerBlock[b].size(); ++oi) {
                    int offset = offsetsPerBlock[b][oi];
                    for (int lane = 0; lane < blockWidth; ++lane) {
                        int col = static_cast<int>(b) * blockWidth + lane;
                        int row = col + offset;
                        
                        if (row >= 0 && row < numRows && col >= 0 && col < numCols) {
                            int dataIndex = oi * blockWidth + lane;
                            T value = ptrToBlock[b][dataIndex];
                            if (value != T(0)) {
                                denseMatrix[row][col] = value;
                            }
                        }
                    }
                }
            }
        }
        
        // Add CPU fallback entries
        for (size_t i = 0; i < cpuValues.size(); ++i) {
            int row = cpuRowIndices[i];
            int col = cpuColIndices[i];
            if (row >= 0 && row < numRows && col >= 0 && col < numCols) {
                denseMatrix[row][col] = cpuValues[i];
            }
        }
    }
    
    // Calculate the maximum width needed for any value to ensure proper alignment
    int maxWidth = 4; // Minimum width for "0.00" and "."
    if (numCols <= 20) {
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                if (denseMatrix[i][j] != T(0)) {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2) << denseMatrix[i][j];
                    maxWidth = std::max(maxWidth, static_cast<int>(oss.str().length()));
                }
            }
        }
    }
    
    // Print column headers for small matrices
    if (numCols <= 20) {
        std::cout << "     ";
        for (int j = 0; j < numCols; ++j) {
            std::cout << std::setw(maxWidth) << j;
            if (j < numCols - 1) std::cout << " ";
        }
        std::cout << std::endl;
    }
    
    // Print the matrix
    for (int i = 0; i < numRows; ++i) {
        if (numCols <= 20) {
            std::cout << std::setw(3) << i << ": ";
        }
        
        for (int j = 0; j < numCols; ++j) {
            if (denseMatrix[i][j] == T(0)) {
                if (numCols <= 20) {
                    std::cout << std::setw(maxWidth) << ".";
                } else {
                    std::cout << ".";
                }
            } else {
                if (numCols <= 20) {
                    std::cout << std::setw(maxWidth) << std::fixed << std::setprecision(2) << denseMatrix[i][j];
                } else {
                    std::cout << "*";
                }
            }
            if (j < numCols - 1 && numCols <= 20) std::cout << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void HBDIA<T>::deleteMatrix() {
    // Clear all format data
    values.clear();
    rowIndices.clear();
    colIndices.clear();
    
    diagonals.clear();
    offsets.clear();
    
    hbdiaData.clear();
    ptrToBlock.clear();
    offsetsPerBlock.clear();
    cpuRowIndices.clear();
    cpuColIndices.clear();
    cpuValues.clear();
    
    // Reset metadata
    numRows = 0;
    numCols = 0;
    numNonZeros = 0;
    blockWidth = 0;
    threshold = 0;
    
    // Reset all flags
    hasCOO = false;
    hasDIA = false;
    hasHBDIA = false;
    
    std::cout << "All matrix data deleted. Object reset to empty state." << std::endl;
}

template <typename T>
void HBDIA<T>::deleteCOOFormat() {
    if (!hasCOO) {
        std::cout << "COO format is not available" << std::endl;
        return;
    }
    
    // Check if this is the last format
    int availableFormats = (hasCOO ? 1 : 0) + (hasDIA ? 1 : 0) + (hasHBDIA ? 1 : 0);
    if (availableFormats <= 1) {
        std::cout << "Cannot delete COO format: it's the last remaining format. Call deleteMatrix() instead." << std::endl;
        return;
    }
    
    // Clear COO data
    values.clear();
    rowIndices.clear();
    colIndices.clear();
    hasCOO = false;
    
    std::cout << "COO format deleted. Memory freed." << std::endl;
}

template <typename T>
void HBDIA<T>::deleteDIAFormat() {
    if (!hasDIA) {
        std::cout << "DIA format is not available" << std::endl;
        return;
    }
    
    // Check if this is the last format
    int availableFormats = (hasCOO ? 1 : 0) + (hasDIA ? 1 : 0) + (hasHBDIA ? 1 : 0);
    if (availableFormats <= 1) {
        std::cout << "Cannot delete DIA format: it's the last remaining format. Call deleteMatrix() instead." << std::endl;
        return;
    }
    
    // Clear DIA data
    diagonals.clear();
    offsets.clear();
    hasDIA = false;
    
    std::cout << "DIA format deleted. Memory freed." << std::endl;
}

template <typename T>
void HBDIA<T>::deleteHBDIAFormat() {
    if (!hasHBDIA) {
        std::cout << "HBDIA format is not available" << std::endl;
        return;
    }
    
    // Check if this is the last format
    int availableFormats = (hasCOO ? 1 : 0) + (hasDIA ? 1 : 0) + (hasHBDIA ? 1 : 0);
    if (availableFormats <= 1) {
        std::cout << "Cannot delete HBDIA format: it's the last remaining format. Call deleteMatrix() instead." << std::endl;
        return;
    }
    
    // Clear HBDIA data
    hbdiaData.clear();
    ptrToBlock.clear();
    offsetsPerBlock.clear();
    cpuRowIndices.clear();
    cpuColIndices.clear();
    cpuValues.clear();
    blockWidth = 0;
    threshold = 0;
    hasHBDIA = false;
    
    std::cout << "HBDIA format deleted. Memory freed." << std::endl;
}

// Explicit template instantiations for common types
template class HBDIA<double>;
template class HBDIA<float>;
template class HBDIA<int>;