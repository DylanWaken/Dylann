//
// Created by Dylan on 8/6/2022.
//

#include "cuTensor.cuh"

namespace dylann{
    cuTensor cuTensor::operator+=(const cuTensor& A) const {
        add(this->impl, A.impl, 1, 1);
    }
}