
#add_compile_definitions(__HIP_PLATFORM_AMD__)
# Include CUDA directories
#include_directories(${CUDAToolkit_INCLUDE_DIRS})

# specify cuda standard
# set(CMAKE_CUDA_STANDARD 12)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# # Set C++ standard to C++17
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures
# set(CMAKE_CUDA_ARCHITECTURES 70 86)

#enable_language(CUDA)

# Set compile options for CUDA
#target_compile_options(HPCGLib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    # --std=c++12
    # -arch=sm_86
#>)


file(GLOB SOURCES "*.cpp")
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE HIP)

add_library(UtilLib
    ${SOURCES}
)


target_link_libraries(UtilLib  PUBLIC
    HPCGLib
    hipsparse
    hipblas
    #CUDA::cudart
    MPI::MPI_CXX
    #amdhip64
)