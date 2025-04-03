# Minimum CMake version required
file(GLOB SOURCES "**/*.cpp")
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE HIP)

add_library(HPCGLib ${SOURCES}
)
include_directories("/capstor/scratch/cscs/ybudanaz/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/hipsparse-6.2.1-r2wk6eagqbyozhokoioobayvtjtgoilb/include")
target_include_directories(HPCGLib PUBLIC "/capstor/scratch/cscs/ybudanaz/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/hipsparse-6.2.1-r2wk6eagqbyozhokoioobayvtjtgoilb/include")
target_link_directories(HPCGLib PUBLIC "/capstor/scratch/cscs/ybudanaz/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/hipsparse-6.2.1-r2wk6eagqbyozhokoioobayvtjtgoilb/lib")

target_include_directories(HPCGLib PUBLIC ${hipSPARSE_INCLUDE_DIRS})

target_link_libraries(
    HPCGLib  PUBLIC
    TimingLib
    UtilLib
    MatrixLib
    hipsparse
    hipblas
    MPI::MPI_CXX
)
