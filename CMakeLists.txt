cmake_minimum_required(VERSION 2.8)

#��Ŀ��
project(SIFT)

#����include����			  
include_directories(${PROJECT_SOURCE_DIR}/include)


find_path(OpenCV_PATH OPENCV $ENV{OPENCV_DIR})
message(${OpenCV_PATH})
include_directories(${OpenCV_PATH}/include)
set(OpencvCV_x64LIB ${OpenCV_PATH}/x64/vc10)
set(OpencvCV_BINARY_DIR ${OpencvCV_x64LIB}/bin)
message(${OpencvCV_x64LIB})
link_directories(${OpencvCV_x64LIB}/lib)

#Դ�ļ�
set(lib_srcs src/sift.cpp)
#ͷ�ļ�
set(lib_hdrs include/sift.hpp)
#��Ԫ����
set(tool test/main.cpp)

#���ɾ�̬��
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

#���ɿ�ִ���ļ�			  
add_executable(tools ${tool})
#���������Ķ�̬��
target_link_libraries(tools sift)
#����dll��debug�ļ�����
foreach(lib_name ${lib_names})
message(${OpencvCV_BINARY_DIR}/${lib_name})
add_custom_command(
          TARGET tools
          POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy "${OpencvCV_BINARY_DIR}/${lib_name}" ${PROJECT_BINARY_DIR}/Debug)                                          

endforeach()