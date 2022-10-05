cmake_minimum_required(VERSION 3.22)
set(CMAKE_CUDA_ARCHITECTURES 86)

project(Dylann CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(TORCH_DIR "D:\\Dependencies\\libtorch")
set(TORCHVISION_DIR "D:\\Dependencies\\torchvision")

set(OpenCV_DIR "D:\\Dependencies\\OpenCV_CUDA\\build")
set(CUDNN_PATH "D:/Program\ Files/NVIDIA/CUDNN_115")

include_directories(${OpenCV_DIR}/include)

find_package(OpenCV REQUIRED core highgui imgproc)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

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
        dylann/tensor/cuTensor.cu
        dylann/tensor/cuTensor.cuh
        dylann/tensor/shell.cu
        dylann/tensor/shell.cuh
        dylann/ops/cuLinear.cu
        dylann/ops/cuLinear.cuh
        dylann/ops/cuConv.cu
        dylann/ops/cuConv.cuh

        dylann/ops/cuActivation.cu
        dylann/ops/cuActivation.cuh
        dylann/ops/cuReduce.cu
        dylann/ops/cuReduce.cuh
        dylann/serial/Instructions.cu
        dylann/serial/Instructions.cuh
        dylann/ops/cuPool.cu
        dylann/ops/cuPool.cuh
        dylann/ops/cuBatchnorm.cu
        dylann/ops/cuBatchnorm.cuh
        dylann/ops/cuConcat.cu
        dylann/ops/cuConcat.cuh
        dylann/DylannContext.cu
        dylann/DylannContext.cuh
        dylann/ops/cuDropout.cu
        dylann/ops/cuDropout.cuh
        dylann/serial/GradInstructions.cu
        dylann/serial/GradInstructions.cu
        dylann/module/Module.cu
        dylann/module/Module.cuh
        dylann/serial/AutoGrad.cu
        dylann/serial/AutoGrad.cuh
        dylann/optim/Optimizers.cu
        dylann/optim/Optimizers.cuh
        dylann/optim/Loss.cu
        dylann/optim/Loss.cuh
        presets/Resnet.cu presets/Resnet.cuh
        dylann/module/Sequence.cu
        dylann/module/Sequence.cuh
        io/dataset/Data.cu
        io/dataset/Data.cuh
        io/dataset/AugCVInstructions.cu
        io/dataset/AugCVInstructions.cuh
        cudautil/ThreadController.cu
        cudautil/ThreadController.cuh
        io/dataset/DataPipeline.cu
        io/dataset/DataPipeline.cuh
        io/dataset/Dataset.cu
        io/dataset/Dataset.cuh
        io/dataset/AngTensorInstructions.cu
        io/dataset/AngTensorInstructions.cuh
        presets/readFuncs/BuildinReadfuncs.cu
        presets/readFuncs/BuildinReadfuncs.cuh
#        testExec.cu
#        testOptim.cu
        dylann/optim/OptImpl.cu dylann/optim/OptImpl.cuh)

target_link_libraries(Dylann ${OpenCV_LIBS})
include_directories(${CUDNN_PATH}/include)
link_directories(${CUDNN_PATH} ${CUDNN_PATH}/bin ${CUDNN_PATH}/lib/x64 ${CUDNN_PATH}/dec/dll_x64)
target_link_libraries(Dylann cudnn cublas)

set_target_properties(Dylann PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 86)
