cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(yolo-tracker)

set (OpenCV_DIR "/home/dl/3rdparty/opencv-3.2.0/install/share/OpenCV/")
find_package(OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


set (Torch_DIR "/home/dl/3rdparty/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
include_directories(${Torch_INCLUDE_DIRS})
message("Torch_INCLUDE_DIRS " ${Torch_INCLUDE_DIRS})

set(folder_source
        main.cpp
        src/Darknet.cpp
        src/Tracker.cpp
        src/ObjectTracker.cpp
        src/HungarianAlg.cpp)

set(folder_include  ./include)


SOURCE_GROUP("Source Files" FILES ${folder_source})


include_directories(${folder_include})

add_executable(yolo-tracker  ${folder_source})
target_link_libraries(yolo-tracker "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET yolo-tracker PROPERTY CXX_STANDARD 11)
