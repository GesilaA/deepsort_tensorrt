#include <iostream>
#include <NvInfer.h>
#include "deepsortenginegenerator.h"
#include "cuda_runtime_api.h"
#include "logging.h"

using namespace nvinfer1;

static Logger gLogger;

int main(int argc, char** argv) {
    cudaSetDevice(0);
    if (argc < 3) {
        std::cout << "./onnx2engine [input .onnx path] [output .engine path]" << std::endl;
        return -1;
    }
    std::string onnxPath = argv[1];
    std::string enginePath = argv[2];
    DeepSortEngineGenerator* engG = new DeepSortEngineGenerator(&gLogger);
    engG->setFP16(true);
    engG->createEngine(onnxPath, enginePath);
    std::cout << "==============" << std::endl;
    std::cout << "|  SUCCESS!  |" << std::endl;
    std::cout << "==============" << std::endl;
    return 0;
}

    
