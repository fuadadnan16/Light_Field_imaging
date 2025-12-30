cmake_minimum_required(VERSION 3.16)
project(LightFieldStudio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find OpenCV (via vcpkg) ---
find_package(OpenCV REQUIRED)

# --- Sources ---
add_executable(LightFieldStudio
    src/main.cpp
    src/lightfield.cpp
)

target_include_directories(LightFieldStudio PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(LightFieldStudio PRIVATE
    ${OpenCV_LIBS}
)

# Windows-specific: prevent console popup (optional)
if (WIN32)
    target_compile_definitions(LightFieldStudio PRIVATE NOMINMAX)
endif()
