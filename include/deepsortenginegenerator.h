//#ifndef DEEPSORT_ENGINE_GENERATOR_H
//#define DEEPSORT_ENGINE_GENERATOR_H

#ifdef _DLL_EXPORTS
#define DLL_API _declspec(dllexport)
#else
#define DLL_API _declspec(dllimport)
#endif

#include <iostream>
#include <NvInfer.h>
#include <NvOnnxParser.h>

using namespace nvinfer1;

const int IMG_HEIGHT = 128;
const int IMG_WIDTH = 64;
const int MAX_BATCH_SIZE = 128;
const std::string INPUT_NAME("input");

class DLL_API DeepSortEngineGenerator {
public:
    DeepSortEngineGenerator(ILogger* gLogger);
    ~DeepSortEngineGenerator();

public:
    void setFP16(bool state);
    void createEngine(std::string onnxPath, std::string enginePath);

private: 
    std::string modelPath, engingPath;
    ILogger* gLogger;  
    bool useFP16; 
};

//#endif
