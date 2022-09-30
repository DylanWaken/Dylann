//
// Created by Dylan on 9/21/2022.
//

#include "Sequence.cuh"


namespace dylann {
    unsigned int MAGIC_NUMBER = 0xfade2077;
    
    void dylann::Sequence::toFile(const string &basePath, const string &saveName) {
        ofstream file(basePath + saveName + ".dylann");
        assert(file.is_open());
        
        //construct file header
        auto *header = (unsigned char *) alloca(256 * sizeof(unsigned char));
        memset(header, 0, 256 * sizeof(unsigned char));
        
        //write header info
        memcpy(header, &MAGIC_NUMBER, sizeof(unsigned int));
    }
}
