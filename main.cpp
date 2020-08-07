#include <iostream>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include "ColorSpace.h"

int main(int argc, char **argv)
{
    std::string file_name="sources/nv0";
    std::ifstream ifile(file_name, std::ios::binary);
    ifile.seekg(0, std::ios::end);
    int length = ifile.tellg();
    ifile.seekg(0, std::ios::beg);
    std::shared_ptr<uint8_t> buffer = std::make_shared<uint8_t>(length);
    ifile.read(buffer.get(), length);

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

    return 0;
}