cmake_minimum_required(VERSION 3.20)
project(proofline VERSION 0.1.0 LANGUAGES CXX)

add_executable(proofline
    src/main.cpp
    src/attention_cache.cpp
    src/model.cpp
    src/certificate.cpp
    src/codegen.cpp
)
target_compile_features(proofline PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(proofline PRIVATE /W4 /permissive-)
else()
    target_compile_options(proofline PRIVATE -Wall -Wextra -Wpedantic)
endif()
