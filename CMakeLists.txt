cmake_minimum_required(VERSION 3.22)
project(Dylann CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(TORCH_DIR "D:\\Dependencies\\libtorch")
set(TORCHVISION_DIR "D:\\Dependencies\\torchvision")

set(OpenCV_DIR "D:\\Dependencies\\OpenCV_CUDA\\build")
include_directories(${OpenCV_DIR}/include)
include_directories(${TORCHVISION_DIR}/include)
find_package(OpenCV REQUIRED core highgui imgproc)
find_package(Torch REQUIRED PATHS ${TORCH_DIR} NO_DEFAULT_PATH)
find_package(TorchVision REQUIRED PATHS ${TORCHVISION_DIR} NO_DEFAULT_PATH)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${Torch_INCLUDE_DIRS})

add_executable(Dylann main.cu
        dylann/tensor/cuTensorBase.cu
        dylann/tensor/cuTensorBase.cuh

        cudautil/assertion.cu
        cudautil/assertion.cuh

        io/logging/Color.cu
        io/logging/Color.cuh
        io/logging/LogUtils.cu
        io/logging/LogUtils.cuh
        dylann/ops/cuTensorOps.cu
        dylann/ops/cuTensorOps.cuh
        dylann/ops/opRegistry.cuh
        dylann/ops/opRegistry.cu
        dylann/tensor/cuTensor.cu
        dylann/tensor/cuTensor.cuh dylann/tensor/shell.cu dylann/tensor/shell.cuh)

target_link_libraries(Dylann ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${TORCHVISION_DIR}/lib/torchvision.lib)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET Dylann
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:Dylann>)
endif (MSVC)


set_target_properties(Dylann PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 86)
