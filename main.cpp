#include <iostream>
#include "opencv2/opencv.hpp"
#include <ctime>
#include "gaussFilter.h"

int main() {
    int ksize = 5;
    float sigma = 10;
    cv::Mat m1 = cv::imread("../1.JPG");

    cv::Mat m2 = cv::Mat::zeros(cv::Size(m1.cols, m1.rows), m1.type());
    cv::Mat m3 = cv::Mat::zeros(cv::Size(m1.cols, m1.rows), m1.type());
    cv::Mat m4 = cv::Mat::zeros(cv::Size(m1.cols, m1.rows), m1.type());
    cv::Mat m5 = cv::Mat::zeros(cv::Size(m1.cols, m1.rows), m1.type());
    cv::Mat m6 = cv::Mat::zeros(cv::Size(m1.cols, m1.rows), m1.type());
    int width = m1.cols;
    int height = m1.rows;
    int channel = m1.channels();

    ///////////////   OpenCV Test    ///////////////////////
    clock_t t1 = clock();
    cv::GaussianBlur(m1, m2, cv::Size(ksize, ksize), sigma);
    printf("Time[Gaussian OpenCV] %fs\n", ((double) (clock() - t1)) / CLOCKS_PER_SEC);
    cv::imwrite("../opencv.jpg", m2);
    //////////////////////////////////////////////////////////

    //////////////    Separate Gaussian Test   ///////////////
    clock_t t4 = clock();
    separateGaussianFilter(m1.data, m5.data, height, width, 3, ksize, sigma);
    printf("Time[Separate Gaussian] %fs\n", ((double) (clock() - t4)) / CLOCKS_PER_SEC);
    cv::imwrite("../gauss_sep.jpg", m5);
    ///////////////////////////////////////////////////////////

    /////////////    Separate Gaussian NEON U8 Test ///////////
    clock_t t = clock();
    gaussianFilter_u8_Neon(m1.data, m6.data, height, width, channel, ksize, sigma);
    printf("Time[Separate Gaussian NEON x3] %fs\n", ((double) (clock() - t)) / CLOCKS_PER_SEC);
    cv::imwrite("../gauss_sep_neon_u8.jpg", m6);
    ///////////////////////////////////////////////////////////

    /////////////    Separate Gaussian NEON F32 Test  /////////
    cv::Mat m1_f32 = cv::Mat::zeros(cv::Size(m1.cols, m1.rows), CV_32FC3);
    m1.copyTo(m3);
    m3.convertTo(m1_f32, CV_32FC3, 1.0/255.0);
    cv::Mat m_res = cv::Mat::zeros(cv::Size(m1_f32.cols, m1_f32.rows), CV_32FC3);

    float * tmp = (float *) malloc(sizeof(float) * m1_f32.rows * m1_f32.cols * m1_f32.channels());
    for(int i = 0; i < m1_f32.rows; i++){
        for(int j = 0; j < m1_f32.cols; j++){
            tmp[i * m1_f32.cols * 3 + j * 3 + 0] = m1_f32.at<cv::Vec3f>(i, j)[0];
            tmp[i * m1_f32.cols * 3 + j * 3 + 1] = m1_f32.at<cv::Vec3f>(i, j)[1];
            tmp[i * m1_f32.cols * 3 + j * 3 + 2] = m1_f32.at<cv::Vec3f>(i, j)[2];
        }
    }

    clock_t t2 = clock();
    gaussianFilter_float_Neon(tmp, tmp, height, width, 3, ksize, sigma);
    printf("Time[Separate Gaussian NEON F32 x3] %fs\n", ((double) (clock() - t2)) / CLOCKS_PER_SEC);

    for(int i = 0; i < m_res.rows; i++){
        for(int j = 0; j < m_res.cols; j++){
            m_res.at<cv::Vec3f>(i, j)[0] = *tmp;
            tmp++;
            m_res.at<cv::Vec3f>(i, j)[1] = *tmp;
            tmp++;
            m_res.at<cv::Vec3f>(i, j)[2] = *tmp;
            tmp++;
        }

    }

    cv::imwrite("../gauss_sep_neon_f32.jpg", m_res*255);
    //////////////////////////////////////////////////////////////////////////////////////////////////

    //////////////// Original Gaussian Test     /////////////////
//    clock_t t3 = clock();
//    for(int i = 0; i < 1; i++) {
//        GaussianFilter(m1.data, m4.data, height, width, 3, ksize, sigma);
//    }
//    printf("Time[Gaussian Origin] %fs\n", ((double) (clock() - t3)) / CLOCKS_PER_SEC);
//    cv::imwrite("../gauss_ori.jpg", m4);
    //////////////////////////////////////////////////////////////

    return 0;
}
