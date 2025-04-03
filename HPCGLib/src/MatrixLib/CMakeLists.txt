
file(GLOB SOURCES "*.cpp")
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE HIP)

add_library(MatrixLib ${SOURCES})


target_link_libraries(MatrixLib  PUBLIC
    HPCGLib
    UtilLib
    #CUDA::cudart
)