#include <iostream>
#include "opencv2/opencv.hpp"
#include <ctime>
#include "gaussFilter.h"

int main() {
    int ksize = 65;
    float sigma = 10;
    cv::Mat m1 = cv::imread("../1.JPG");
    cv::Mat m2 = cv::Mat::zeros(cv::Size(m1.cols, m1.rows), m1.type());
    cv::Mat m3 = cv::Mat::zeros(cv::Size(m1.cols, m1.rows), m1.type());
    cv::Mat m4 = cv::Mat::zeros(cv::Size(m1.cols, m1.rows), m1.type());
    cv::Mat m5 = cv::Mat::zeros(cv::Size(m1.cols, m1.rows), m1.type());
    cv::Mat m6 = cv::Mat::zeros(cv::Size(m1.cols, m1.rows), m1.type());
    int width = m1.cols;
    int height = m1.rows;

    clock_t t1 = clock();
    for(int i = 0; i < 100; i++) {
        cv::GaussianBlur(m1, m2, cv::Size(ksize, ksize), sigma);
    }
    printf("Time[Gaussian OpenCV] %fs\n", ((double) (clock() - t1)) / CLOCKS_PER_SEC);
    cv::imwrite("../opencv.jpg", m2);

//    clock_t t4 = clock();
//    for(int i = 0; i < 100; i++) {
//        separateGaussianFilter(m1.data, m5.data, height, width, 3, ksize, sigma);
//    }
//    printf("Time[Separate Gaussian] %fs\n", ((double) (clock() - t4)) / CLOCKS_PER_SEC);
//    cv::imwrite("../gauss_sep.jpg", m5);

    clock_t t5 = clock();
    for(int i = 0; i < 100; i++) {
        gaussianFilter_u8_Neon(m1.data, m6.data, height, width, 3, ksize, sigma);
    }
    printf("Time[Separate Gaussian NEON x3] %fs\n", ((double) (clock() - t5)) / CLOCKS_PER_SEC);
    cv::imwrite("../gauss_sep_neon.jpg", m6);

    cv::Mat m1_f32 = cv::Mat::zeros(cv::Size(m1.cols, m1.rows), CV_32FC3);
    m1.copyTo(m3);
    m3.convertTo(m1_f32, CV_32FC3);
    cv::Mat m_res = cv::Mat::zeros(cv::Size(m1_f32.cols, m1_f32.rows), m1_f32.type());

    clock_t t2 = clock();
    for(int i = 0; i < 100; i++){
        gaussianFilter_float_Neon((float*)m1_f32.data, (float*)m_res.data, height, width, 3, ksize, sigma);
    }
    printf("Time[Separate Gaussian NEON F32 x3] %fs\n", ((double) (clock() - t2)) / CLOCKS_PER_SEC);
    cv::imwrite("../gauss_sep_neon_f32.jpg", m1_f32);

//    clock_t t3 = clock();
//    for(int i = 0; i < 100; i++) {
//        GaussianFilter(m1.data, m4.data, height, width, 3, ksize, sigma);
//    }
//    printf("Time[Gaussian Origin] %fs\n", ((double) (clock() - t3)) / CLOCKS_PER_SEC);
//    cv::imwrite("../gauss_ori.jpg", m4);

    return 0;
}
