cmake_minimum_required(VERSION 3.5.1)
project(FABO)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER ${CMAKE_CXX_COMPILER})

include_directories(${PROJECT_SOURCE_DIR})
# Optional: point BOOST_INCLUDE_DIR at a Boost headers directory if needed.
if(DEFINED BOOST_INCLUDE_DIR AND EXISTS "${BOOST_INCLUDE_DIR}")
    include_directories("${BOOST_INCLUDE_DIR}")
endif()

file(GLOB SALT_SRCS salt/*.cpp salt/*/*.cpp salt/base/flute/*.c)

add_library(salt STATIC ${SALT_SRCS})
add_library(fabo_interface STATIC other_methods/interface.cpp)
target_link_libraries(fabo_interface salt)

add_executable(run_fabo examples/run_sota_main.cpp)
target_link_libraries(run_fabo fabo_interface salt)

add_executable(eval_fabo_sweep_worker examples/eval_sota_sweep_worker.cpp)
target_link_libraries(eval_fabo_sweep_worker fabo_interface salt)
