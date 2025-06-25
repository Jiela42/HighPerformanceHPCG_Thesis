cmake_minimum_required(VERSION 3.20)
project(HighPerformanceHPCG_Thesis)

link_directories(/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/nccl-2.22.3-1-4j6h3ffzysukqpqbvriorrzk2lm762dd/lib)

add_subdirectory(HPCGLib)
add_subdirectory(hpcg)