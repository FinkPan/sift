cmake_minimum_required(VERSION 2.8)

#项目名
project(SIFT)

#包含include搜索			  
include_directories(${PROJECT_SOURCE_DIR}/include)


find_path(OpenCV_PATH OPENCV $ENV{OPENCV_DIR})
message(${OpenCV_PATH})
include_directories(${OpenCV_PATH}/include)
set(OpencvCV_x64LIB ${OpenCV_PATH}/x64/vc10)
set(OpencvCV_BINARY_DIR ${OpencvCV_x64LIB}/bin)
message(${OpencvCV_x64LIB})
link_directories(${OpencvCV_x64LIB}/lib)

#源文件
set(lib_srcs src/sift.cpp)
#头文件
set(lib_hdrs include/sift.hpp)
#单元测试
set(tool test/main.cpp)

#生成静态库
add_library(sift STATIC ${lib_hdrs} ${lib_srcs})
target_link_libraries(sift opencv_features2d300d.lib
                                             opencv_core300d.lib
                                             opencv_features2d300d.lib
                                             opencv_highgui300d.lib
                                             opencv_nonfree300d.lib
                                             opencv_imgproc300d.lib)
                                             
set(lib_names  opencv_features2d300d.dll
                        opencv_core300d.dll
                        opencv_features2d300d.dll
                        opencv_highgui300d.dll
                        opencv_nonfree300d.dll
                        opencv_imgproc300d.dll)  

#生成可执行文件			  
add_executable(tools ${tool})
#设置依赖的动态库
target_link_libraries(tools sift)
#复制dll到debug文件夹里
foreach(lib_name ${lib_names})
message(${OpencvCV_BINARY_DIR}/${lib_name})
add_custom_command(
          TARGET tools
          POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy "${OpencvCV_BINARY_DIR}/${lib_name}" ${PROJECT_BINARY_DIR}/Debug)                                          

endforeach()
