// File: ${PACKAGE_NAME}
// Brief: 
// Author: Lixiaolong@xa.com
// Date: 2022/6/30
//

#ifndef XAGISPALGORITHM_GAUSSFILTER_H
#define XAGISPALGORITHM_GAUSSFILTER_H

#ifdef __cplusplus
extern "C"{
#endif

typedef unsigned char U8;

/**
 * @brief 高斯滤波浮点C实现
 * @param src
 * @param dst
 * @param row
 * @param col
 * @param channel
 * @param ksize
 * @param sigma
 */
void GaussianFilter(unsigned char *src, unsigned char* dst, int row, int col, int channel, int ksize, float sigma);

/**
 * @brief 核分离高斯滤波浮点C实现
 * @param src
 * @param dst
 * @param row
 * @param col
 * @param channel
 * @param ksize
 * @param sigma
 */
void separateGaussianFilter(unsigned char *src, unsigned char* dst, int row, int col, int channel, int ksize, float sigma);

/**
 * @brief 核分离高斯滤波定点型NEON实现
 * @param src
 * @param dst
 * @param height
 * @param width
 * @param channel
 * @param ksize
 * @param sigma
 */
void gaussianFilter_u8_Neon(U8* src, U8* dst, int height, int width, int channel, int ksize, float sigma);

/**
 * @brief 核分离高斯滤波浮点型NEON实现
 * @param src
 * @param dst
 * @param height
 * @param width
 * @param channel
 * @param ksize
 * @param sigma
 */
void gaussianFilter_float_Neon(float* src, float* dst, int height, int width, int channel, int ksize, float sigma);


#ifdef __cplusplus
}
#endif

#endif //XAGISPALGORITHM_GAUSSFILTER_H
