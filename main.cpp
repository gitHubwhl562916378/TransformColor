#include <iostream>
#include <fstream>
#include <memory>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include "ColorSpace.h"

int main(int argc, char **argv)
{
    std::string file_name = "sources/nv1";
    std::ifstream ifile("sources/nv1", std::ios::binary);
    ifile.seekg(0, std::ios::end);
    int length = ifile.tellg();
    ifile.seekg(0, std::ios::beg);
    std::cout << "file length " << length << std::endl;

    int width = 1920, height = 1080;
    uint8_t *buffer = new uint8_t[length];
    ifile.read(reinterpret_cast<char*>(buffer), length);

    CUdeviceptr device_ptr = 0;
    cuMemAlloc(&device_ptr, length);
    CUDA_MEMCPY2D m = { 0 };
	m.srcMemoryType = CU_MEMORYTYPE_HOST;
	m.srcDevice = (CUdeviceptr)(m.srcHost = buffer);
	m.srcPitch = width;
	m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	m.dstDevice = device_ptr;
	m.dstPitch = width;
	m.WidthInBytes = width;
	m.Height = height * 1.5;
	cuMemcpy2DAsync(&m, 0);
    // cudaMemcpy(device_ptr, buffer, length, cudaMemcpyHostToDevice);

    CUdeviceptr rgba_device = 0;
    cuMemAlloc(&rgba_device, width * height * 4);
    cv::Mat img(height, width, CV_8UC4);
    Nv12ToColor32<BGRA32>(reinterpret_cast<uint8_t*>(device_ptr), width, reinterpret_cast<uint8_t*>(rgba_device), width * 4, width, height, 0);
    cudaMemcpy(reinterpret_cast<void*>(img.data), reinterpret_cast<void*>(rgba_device), width * height * 4 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    cv::imwrite(file_name + ".jpg", img);

    // cuFree(device_ptr);
    // cuFree(rgba_device);
    return 0;
}