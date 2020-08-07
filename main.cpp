#include <iostream>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include "ColorSpace.h"

int main(int argc, char **argv)
{
    std::string file_name="/workspaces/tensorrt/TransformColor/bin/sources/nv0";
    std::ifstream ifile(file_name, std::ios::in | std::ios::binary);
    if(!ifile.is_open()){
        std::cout << "open " << file_name << " failed" << std::endl;
        return -1;
    }
    ifile.seekg(0, std::ios::end);
    int length = ifile.tellg();
    ifile.seekg(0, std::ios::beg);
    std::shared_ptr<uint8_t> buffer(new uint8_t[length]);
    ifile.read(reinterpret_cast<char*>(buffer.get()), length);

    uint8_t *d_nv12ptr = 0;
    cudaMalloc(&d_nv12ptr, length);
    cudaMemcpy(d_nv12ptr, buffer.get(), length, cudaMemcpyHostToDevice);

    int width = 1920, height = 1080;
    uint8_t *d_bgraptr = 0;
    cudaMalloc(&d_bgraptr, width * height * 4 * sizeof(uint8_t));
    Nv12ToColor32<BGRA32>(d_nv12ptr, width, d_bgraptr, width * 4, width, height, 5);
    
    cv::Mat img(height, width, CV_8UC4);
    cudaMemcpy(img.data, d_bgraptr, width * height * 4, cudaMemcpyDeviceToHost);
    cv::imwrite(file_name + ".jpg", img);

    cudaFree(d_nv12ptr);
    cudaFree(d_bgraptr);

    return 0;
}