cmake_minimum_required(VERSION 3.10)
project(Example)

include_directories("/home/jorgen/Documents/eigen-3.4.0")

add_executable(runit Ex4c.cpp Neural_Network_1N1.cpp Neural_Network_Settings.cpp)

target_link_libraries(runit /usr/local/lib/libnlopt.so)
