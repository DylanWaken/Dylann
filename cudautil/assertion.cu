//
// Created by Dylan on 8/5/2022.
//

#include "assertion.cuh"

void assertCuda(const char *file, int line){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        logFatal(io::LOG_SEG_COMP, "CUDA error at " + string(file) + ": " + to_string(line)
        + ": " + cudaGetErrorString(err));
        std::cerr << "CUDA error at " << file << ":" << line << ": " << cudaGetErrorString(err) << std::endl;
        assert(false);
    }
}