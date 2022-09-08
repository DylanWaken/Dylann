//
// Created by Dylan on 9/7/2022.
//

#ifndef DYLANN_GRADINSTRUCTIONS_CUH
#define DYLANN_GRADINSTRUCTIONS_CUH

#include "Instructions.cuh"

#define INS_GRADS_ADD 1000
#define INS_GRADS_SCALE 1001

namespace dylann {
     
     struct ADD_GRADS : public Operation{
     public:
         TENSOR_PTR A;
         TENSOR_PTR B;
         float alpha;
         float beta;
         
         ADD_GRADS(TENSOR_PTR A, TENSOR_PTR B, float alpha, float beta)
                 : Operation(INS_GRADS_ADD, 4), A(A), B(B), alpha(alpha), beta(beta) {}
    
         void run() override;
    
         void encodeParams(unsigned char * file, size_t &offset) override;
    
         size_t getEncodedSize() override {
             return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2 + sizeof(float) * 2;
         }
    
         void print() override {
             cout << "GRADS ADD " << std::hex << A << " " << std::hex
                  << B << " " << std::dec << alpha << " " << beta << endl;
         }
     };
     
     struct SCALE_GRADS : public Operation{
     public:
         TENSOR_PTR A;
         float alpha;
         
         SCALE_GRADS(TENSOR_PTR A, float alpha)
                 : Operation(INS_GRADS_SCALE, 2), A(A), alpha(alpha) {}
    
         void run() override;
    
         void encodeParams(unsigned char * file, size_t &offset) override;
    
         size_t getEncodedSize() override {
             return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) + sizeof(float);
         }
    
         void print() override {
             cout << "GRADS SCALE " << std::hex << A << " " << std::dec << alpha << endl;
         }
     };
     }
    
} // dylann

#endif //DYLANN_GRADINSTRUCTIONS_CUH
