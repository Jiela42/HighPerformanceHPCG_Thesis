
file(GLOB SOURCES "*.cpp")
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE HIP)

add_library(TimingLib ${SOURCES})

target_link_libraries(TimingLib PUBLIC
    UtilLib
)