// File: filter.c
// Brief: 图像高斯滤波
// Author: Lixiaolong@xa.com
// Date: 2022/6/30
//

#include <printf.h>
#include "stdlib.h"
#include "math.h"
#include "gaussFilter.h"
#include "arm_neon.h"
#include "string.h"

#define MAX_KERNEL_SIZE   79
#define XA_U8_SIZE        16
#define XA_F32_SIZE       4
#define PI 3.14159
#define NORM_SHIFT        8


static void getGaussianKernel1D(float *kernel, int ksize, float sigma);

static void getGaussianKernel2D(float *kernel, int ksize, float sigma);

static void generateGaussianKernel(float **kernel, int ksize, float sigma);

static unsigned int getGaussianInt(uint16_t *kernel, int ksize);

static void verticalFilterNeonU8(U8* src, U8* dst, int height, int width, int channel, uint16_t* kernel, int ksize);

static void horizonFilterNeonU8(U8* src, U8* dst, int height, int width, int channel, uint16_t* kernel, int ksize);

static void verticalFilterNeonF32(float* src, float* dst, int height, int width, int channel, float* kernel, int ksize);

static void horizonFilterNeonF32(float* src, float* dst, int height, int width, int channel, float* kernel, int ksize);



static void getGaussianKernel1D(float *kernel, int ksize, float sigma)
{
    if(kernel == NULL){
        kernel = (float *) malloc(sizeof(float ) * ksize);
    }

    int kCenter = ksize / 2;
    float sum = 0;
    for(int i = 0; i < ksize; i++){
        kernel[i] = expf(-(float)(i - kCenter) * (float)(i - kCenter) / (2 * sigma * sigma));
        sum += kernel[i];
    }

    for(int i = 0; i < ksize; i++){
        kernel[i] /= sum;
    }
}

static void getGaussianKernel2D(float *kernel, int ksize, float sigma)
{
    if(kernel == NULL){
        kernel = (float *) malloc(sizeof(float ) * ksize * ksize);
    }
    float flag = 1.0f / (2.0f * PI * sigma * sigma);
    float coef;
    float sum  = 0;
    int center = (ksize - 1) / 2;

    for (int i = 0; i < ksize; i++) {
        float x2 = (float)pow((i - center), 2);
        for (int j = 0; j < ksize; j++) {
            float y2 = (float)pow((j - center), 2);
            coef = (-x2 - y2) / (float)(2.0 * sigma * sigma);
            kernel[i * ksize + j] = flag * expf(coef);
            sum += kernel[i * ksize + j];
        }
    }

    /* Normalize */
    if (sum) {
        for (int i = 0; i < ksize * ksize; i++) {
            kernel[i] /= sum;
        }
    }

}

static void generateGaussianKernel(float **kernel, int ksize, float sigma)
{
    if(kernel == NULL){
        return;
    }

    int center = ksize / 2;
    float x2, y2;
    float sum = 0;
    for(int i = 0; i < ksize; i++){
        x2 = (float)(i - center) * (float)(i - center);
        for(int j = 0; j < ksize; j++){
            y2 = (float)(j - center) * (float)(j - center);
            float g = expf(-(x2 + y2) / (2 * sigma * sigma));
            g /= 2 * PI * sigma * sigma;
            sum += g;
            kernel[i][j] = g;
        }
    }

    for(int i = 0; i < ksize; i++){
        for(int j = 0; j < ksize; j++){
            kernel[i][j] /= sum;
        }
    }
}

static unsigned int getGaussianInt(uint16_t *kernel, int ksize)
{
    if(kernel == NULL){
        return 0;
    }
    int s = 0;

    for(int i = 0; i < ksize / 2 + 1; i++){
        kernel[i] = 2 * (i + 1) - 1;
        s += kernel[i];
    }
    for(int i = ksize / 2 + 1; i < ksize; i++){
        kernel[i] = kernel[ksize - i - 1];
        s += kernel[i];
    }
    return s;
}

static void verticalFilterNeonU8(U8* src, U8* dst, int height, int width, int channel, uint16_t* kernel, int ksize)
{
    int kCenter = ksize / 2;

    unsigned char* in = (unsigned char*) calloc(sizeof(unsigned char), width * (height + ksize) * channel);
    memcpy(in, src, kCenter * width * channel);
    memcpy(in + kCenter * width * channel, src, height * width * channel);
    memcpy(in + (kCenter + height) * width * channel, in + height * width * channel, kCenter * width * channel);

    if(channel == 1){
        for(int i = 0; i < height; i++){
            unsigned char*p_dst = dst + i * width * channel;
            int n = width / XA_U8_SIZE;
            int end = n * (XA_U8_SIZE * channel);
            int count = 0;

            for(int j = 0; j < end; j += XA_U8_SIZE){
                uint16x8_t laccum_u16, haccum_u16;
                laccum_u16 = vmovq_n_u16(0);
                haccum_u16 = vmovq_n_u16(0);

                for(int k = 0; k < ksize; k++){
                    uint8x16_t data = vld1q_u8(in + (i + k) * width * channel + j);
                    uint8x8_t lp_u8 = vget_low_u8( data );
                    uint8x8_t hp_u8 = vget_high_u8( data );

                    uint16x8_t lp_u16 = vmovl_u8( lp_u8 );
                    uint16x8_t hp_u16 = vmovl_u8( hp_u8 );

                    laccum_u16 = vmlaq_n_u16(laccum_u16, lp_u16, kernel[k]);
                    haccum_u16 = vmlaq_n_u16(haccum_u16, hp_u16, kernel[k]);
                }
                laccum_u16 = vshrq_n_u16(laccum_u16, NORM_SHIFT);
                haccum_u16 = vshrq_n_u16(haccum_u16, NORM_SHIFT);
                uint8x8_t laccum_u8 = vmovn_u16( laccum_u16 );
                uint8x8_t haccum_u8 = vmovn_u16( haccum_u16 );
                uint8x16_t accum_u8 = vcombine_u8( laccum_u8, haccum_u8 );
                vst1q_u8(p_dst + j, accum_u8);
                count += 16;
            }

            for(int j = count; j < width; j++){
                U8 s = 0;
                for(int k = 0; k < ksize; k++){
                    s += (in[(i + k) * width * channel + j * channel] * kernel[k]) >> NORM_SHIFT;
                }
                if(s < 0)
                    s = 0;
                if(s > 255)
                    s = 255;
                p_dst[j * channel] = (U8)s;
            }
        }
    }
    else if(channel == 3)
    {
        for(int i = 0; i < height; i++){
            int n = width / XA_U8_SIZE;
            int end = n * (XA_U8_SIZE * channel);
            unsigned char*p_dst = dst + i * width * channel;
            int count = 0;

            for(int j = 0; j < end; j+= XA_U8_SIZE * channel){
                uint16x8_t laccum_u16_r = vmovq_n_u16(0);
                uint16x8_t haccum_u16_r = vmovq_n_u16(0);

                uint16x8_t laccum_u16_g = vmovq_n_u16(0);
                uint16x8_t haccum_u16_g = vmovq_n_u16(0);

                uint16x8_t laccum_u16_b = vmovq_n_u16(0);
                uint16x8_t haccum_u16_b = vmovq_n_u16(0);

                uint8x8_t lp_u8;
                uint8x8_t hp_u8;
                uint16x8_t lp_u16;
                uint16x8_t hp_u16;

                for(int k = 0; k < ksize; k++){
                    uint8x16x3_t data = vld3q_u8(in + (i + k) * width * channel + j);

                    // r
                    lp_u8 = vget_low_u8( data.val[0]);
                    hp_u8 = vget_high_u8( data.val[0]);

                    lp_u16 = vmovl_u8( lp_u8 );
                    hp_u16 = vmovl_u8( hp_u8 );

                    laccum_u16_r = vmlaq_n_u16(laccum_u16_r, lp_u16, kernel[k]);
                    haccum_u16_r = vmlaq_n_u16(haccum_u16_r, hp_u16, kernel[k]);

                    // g
                    lp_u8 = vget_low_u8( data.val[1]);
                    hp_u8 = vget_high_u8( data.val[1]);

                    lp_u16 = vmovl_u8( lp_u8 );
                    hp_u16 = vmovl_u8( hp_u8 );

                    laccum_u16_g = vmlaq_n_u16(laccum_u16_g, lp_u16, kernel[k]);
                    haccum_u16_g = vmlaq_n_u16(haccum_u16_g, hp_u16, kernel[k]);

                    // b
                    lp_u8 = vget_low_u8( data.val[2]);
                    hp_u8 = vget_high_u8( data.val[2]);

                    lp_u16 = vmovl_u8( lp_u8 );
                    hp_u16 = vmovl_u8( hp_u8 );

                    laccum_u16_b = vmlaq_n_u16(laccum_u16_b, lp_u16, kernel[k]);
                    haccum_u16_b = vmlaq_n_u16(haccum_u16_b, hp_u16, kernel[k]);
                }

                laccum_u16_r = vshrq_n_u16(laccum_u16_r, NORM_SHIFT);
                haccum_u16_r = vshrq_n_u16(haccum_u16_r, NORM_SHIFT);
                uint8x8_t laccum_u8_r = vmovn_u16( laccum_u16_r );
                uint8x8_t haccum_u8_r = vmovn_u16( haccum_u16_r );
                uint8x16_t accum_u8_r = vcombine_u8( laccum_u8_r, haccum_u8_r);

                laccum_u16_g = vshrq_n_u16(laccum_u16_g, NORM_SHIFT);
                haccum_u16_g = vshrq_n_u16(haccum_u16_g, NORM_SHIFT);
                uint8x8_t laccum_u8_g = vmovn_u16( laccum_u16_g );
                uint8x8_t haccum_u8_g = vmovn_u16( haccum_u16_g );
                uint8x16_t accum_u8_g = vcombine_u8( laccum_u8_g, haccum_u8_g);

                laccum_u16_b = vshrq_n_u16(laccum_u16_b, NORM_SHIFT);
                haccum_u16_b = vshrq_n_u16(haccum_u16_b, NORM_SHIFT);
                uint8x8_t laccum_u8_b = vmovn_u16( laccum_u16_b );
                uint8x8_t haccum_u8_b = vmovn_u16( haccum_u16_b );
                uint8x16_t accum_u8_b = vcombine_u8( laccum_u8_b, haccum_u8_b);

                uint8x16x3_t res;
                res.val[0] = accum_u8_r;
                res.val[1] = accum_u8_g;
                res.val[2] = accum_u8_b;

                vst3q_u8(p_dst + j, res);

                count += 48;
            }
            // 处理剩余部分
            for (int j = count / 3; j < width; j++) {
                U8 s[3] = {0};
                for (int k = 0; k < ksize; k++) {
                    s[0] += (in[(i + k) * width * channel + j * channel + 0] * kernel[k]) >> NORM_SHIFT;
                    s[1] += (in[(i + k) * width * channel + j * channel + 1] * kernel[k]) >> NORM_SHIFT;
                    s[2] += (in[(i + k) * width * channel + j * channel + 2] * kernel[k]) >> NORM_SHIFT;
                }
                for (int m = 0; m < channel; m++) {
                    if (s[m] < 0) {
                        s[m] = 0;
                    }
                    if (s[m] > 255) {
                        s[m] = 255;
                    }
                }

                p_dst[j * channel + 0] = (U8) s[0];
                p_dst[j * channel + 1] = (U8) s[1];
                p_dst[j * channel + 2] = (U8) s[2];
            }
        }
    } else{
        free(in);
        return;
    }
    free(in);
}

static void horizonFilterNeonU8(U8* src, U8* dst, int height, int width, int channel, uint16_t* kernel, int ksize)
{
    int kCenter = ksize / 2;

    if (channel == 1){
        for(int i = 0; i < height; i++){
            U8* in = (U8*) calloc((width + ksize), sizeof(U8));
            memcpy(in, src + i * width * channel, kCenter * channel);
            memcpy(in + kCenter * channel, src + i * width * channel, width * channel);
            memcpy(in + (kCenter + width) * channel, in + width * channel, kCenter * channel);

            unsigned char* p_dst = dst + i * width;
            int n = width / XA_U8_SIZE;
            int end = n * (XA_U8_SIZE * channel);
            int count = 0;

            for(int j = 0; j < end; j+=XA_U8_SIZE){
                uint16x8_t laccum_u16, haccum_u16;
                laccum_u16 = vmovq_n_u16(0);
                haccum_u16 = vmovq_n_u16(0);

                for(int k = 0; k < ksize; k++){
                    uint8x16_t data = vld1q_u8(in + j + k);
                    uint8x8_t lp_u8 = vget_low_u8( data );
                    uint8x8_t hp_u8 = vget_high_u8( data );

                    uint16x8_t lp_u16 = vmovl_u8( lp_u8 );
                    uint16x8_t hp_u16 = vmovl_u8( hp_u8 );

                    laccum_u16 = vmlaq_n_u16(laccum_u16, lp_u16, kernel[k]);
                    haccum_u16 = vmlaq_n_u16(haccum_u16, hp_u16, kernel[k]);
                }
                laccum_u16 = vshrq_n_u16(laccum_u16, NORM_SHIFT);
                haccum_u16 = vshrq_n_u16(haccum_u16, NORM_SHIFT);
                uint8x8_t laccum_u8 = vmovn_u16( laccum_u16 );
                uint8x8_t haccum_u8 = vmovn_u16( haccum_u16 );
                uint8x16_t accum_u8 = vcombine_u8( laccum_u8, haccum_u8 );
                vst1q_u8(p_dst + j, accum_u8);
                count += XA_U8_SIZE;
            }

            for(int j = count / channel; j < width; j++){
                U8 s = 0;
                for(int k = 0; k < ksize; k++){
                    s += (*(in + (j + k) * channel) * kernel[k]) >> NORM_SHIFT;
                }
                if(s < 0)
                    s = 0;
                if(s > 255)
                    s = 255;

                p_dst[j * channel] = (U8)s;
            }

            free(in);
        }
    }
    else if(channel == 3)
    {
        for(int i = 0; i < height; i++){
            U8* in = (U8*) calloc((width + ksize) * channel, sizeof(U8));
            memcpy(in, src + i * width * channel, kCenter * channel);
            memcpy(in + kCenter * channel, src + i * width * channel, width * channel);
            memcpy(in + (kCenter + width) * channel, in + width * channel, kCenter * channel);

            U8* p_dst = dst + i * width * channel;
            int n = width / XA_U8_SIZE;
            int end = n * (XA_U8_SIZE * channel);
            int count = 0;

            for(int j = 0; j < end; j += XA_U8_SIZE * channel){
                uint16x8_t laccum_u16_r = vmovq_n_u16(0);
                uint16x8_t haccum_u16_r = vmovq_n_u16(0);

                uint16x8_t laccum_u16_g = vmovq_n_u16(0);
                uint16x8_t haccum_u16_g = vmovq_n_u16(0);

                uint16x8_t laccum_u16_b = vmovq_n_u16(0);
                uint16x8_t haccum_u16_b = vmovq_n_u16(0);

                uint8x8_t lp_u8;
                uint8x8_t hp_u8;
                uint16x8_t lp_u16;
                uint16x8_t hp_u16;

                for(int k = 0; k < ksize; k++){
                    uint8x16x3_t data = vld3q_u8(in + j + k * channel);

                    // r
                    lp_u8 = vget_low_u8( data.val[0]);
                    hp_u8 = vget_high_u8( data.val[0]);

                    lp_u16 = vmovl_u8( lp_u8 );
                    hp_u16 = vmovl_u8( hp_u8 );

                    laccum_u16_r = vmlaq_n_u16(laccum_u16_r, lp_u16, kernel[k]);
                    haccum_u16_r = vmlaq_n_u16(haccum_u16_r, hp_u16, kernel[k]);

                    // g
                    lp_u8 = vget_low_u8( data.val[1]);
                    hp_u8 = vget_high_u8( data.val[1]);

                    lp_u16 = vmovl_u8( lp_u8 );
                    hp_u16 = vmovl_u8( hp_u8 );

                    laccum_u16_g = vmlaq_n_u16(laccum_u16_g, lp_u16, kernel[k]);
                    haccum_u16_g = vmlaq_n_u16(haccum_u16_g, hp_u16, kernel[k]);

                    // b
                    lp_u8 = vget_low_u8( data.val[2]);
                    hp_u8 = vget_high_u8( data.val[2]);

                    lp_u16 = vmovl_u8( lp_u8 );
                    hp_u16 = vmovl_u8( hp_u8 );

                    laccum_u16_b = vmlaq_n_u16(laccum_u16_b, lp_u16, kernel[k]);
                    haccum_u16_b = vmlaq_n_u16(haccum_u16_b, hp_u16, kernel[k]);
                }

                laccum_u16_r = vshrq_n_u16(laccum_u16_r, NORM_SHIFT);
                haccum_u16_r = vshrq_n_u16(haccum_u16_r, NORM_SHIFT);
                uint8x8_t laccum_u8_r = vmovn_u16( laccum_u16_r );
                uint8x8_t haccum_u8_r = vmovn_u16( haccum_u16_r );
                uint8x16_t accum_u8_r = vcombine_u8( laccum_u8_r, haccum_u8_r);

                laccum_u16_g = vshrq_n_u16(laccum_u16_g, NORM_SHIFT);
                haccum_u16_g = vshrq_n_u16(haccum_u16_g, NORM_SHIFT);
                uint8x8_t laccum_u8_g = vmovn_u16( laccum_u16_g );
                uint8x8_t haccum_u8_g = vmovn_u16( haccum_u16_g );
                uint8x16_t accum_u8_g = vcombine_u8( laccum_u8_g, haccum_u8_g);

                laccum_u16_b = vshrq_n_u16(laccum_u16_b, NORM_SHIFT);
                haccum_u16_b = vshrq_n_u16(haccum_u16_b, NORM_SHIFT);
                uint8x8_t laccum_u8_b = vmovn_u16( laccum_u16_b );
                uint8x8_t haccum_u8_b = vmovn_u16( haccum_u16_b );
                uint8x16_t accum_u8_b = vcombine_u8( laccum_u8_b, haccum_u8_b);

                uint8x16x3_t res;
                res.val[0] = accum_u8_r;
                res.val[1] = accum_u8_g;
                res.val[2] = accum_u8_b;

                vst3q_u8(p_dst + j, res);
                count += 48;
            }
            // 处理剩余部分
            for (int j = count / 3; j < width; j++) {
                int s[3] = {0};
                for (int k = 0; k < ksize; k++) {
                    s[0] += (*(in + (j + k) * channel + 0) * kernel[k]) >> NORM_SHIFT;
                    s[1] += (*(in + (j + k) * channel + 1) * kernel[k]) >> NORM_SHIFT;
                    s[2] += (*(in + (j + k) * channel + 2) * kernel[k]) >> NORM_SHIFT;
                }
                for (int m = 0; m < channel; m++) {
                    if (s[m] < 0) {
                        s[m] = 0;
                    }
                    if (s[m] > 255) {
                        s[m] = 255;
                    }
                }

                p_dst[j * channel + 0] = (U8) s[0];
                p_dst[j * channel + 1] = (U8) s[1];
                p_dst[j * channel + 2] = (U8) s[2];
            }

            free(in);
        }
    } else{
        return;
    }
}

void gaussianFilter_u8_Neon(U8* src, U8* dst, int height, int width, int channel, int ksize, float sigma)
{
    if(ksize > MAX_KERNEL_SIZE){
        return;
    }
#if 1

    uint16_t *weight = (uint16_t*) malloc(sizeof(uint16_t) * ksize);
    uint16_t sum = getGaussianInt(weight, ksize);
    float weightsNorm = (float)(1 << NORM_SHIFT);
    float iwsum = weightsNorm / (float)sum;
    for(int i = 0; i < ksize; i++){
        weight[i] = (uint16_t)((float)weight[i] * iwsum + 0.5);
    }

    verticalFilterNeonU8(src, dst, height, width, channel, weight, ksize);
    horizonFilterNeonU8(dst, dst, height, width, channel, weight, ksize);

    free(weight);
#endif
}

void GaussianFilter(unsigned char *src, unsigned char* dst, int row, int col, int channel, int ksize, float sigma)
{
    float **kernel = (float **) malloc(sizeof(float*) * ksize);
    for(int i = 0; i < ksize; i++){
        kernel[i] = (float *) malloc(sizeof(float) * ksize);
    }

    generateGaussianKernel(kernel, ksize, sigma);

    int border = ksize / 2;

    for(int i = border; i < row - border; i++){
        for(int j = border; j < col - border; j++){
            float s[3] = {0};
            for(int a = -border; a <= border; a++){
                for(int b = -border; b <= border; b++){
                    if(channel == 1){
                        s[0] += kernel[a + border][b + border] * (float)src[(i + a) * col + j + b];
                    } else{
                        s[0] += kernel[a + border][b + border] * (float)src[(i + a) * col * channel + (j + b) * channel + 0];
                        s[1] += kernel[a + border][b + border] * (float)src[(i + a) * col * channel + (j + b) * channel + 1];
                        s[2] += kernel[a + border][b + border] * (float)src[(i + a) * col * channel + (j + b) * channel + 2];
                    }
                }
            }

            for(int k = 0; k < channel; k++){
                if(s[k] < 0)
                    s[k] = 0;
                if(s[k] > 255)
                    s[k] = 255;
            }

            if(channel == 1){
                dst[i * col + j] = (unsigned char)s[0];
            } else{
                dst[i * col * channel + j * channel + 0] = (unsigned char)s[0];
                dst[i * col * channel + j * channel + 1] = (unsigned char)s[1];
                dst[i * col * channel + j * channel + 2] = (unsigned char)s[2];
            }
        }
    }

    for(int i = 0; i < ksize; i++){
        free(kernel[i]);
    }
    free(kernel);
}

void separateGaussianFilter(unsigned char *src, unsigned char* dst, int row, int col, int channel, int ksize, float sigma)
{
    float *matrix = (float *)malloc(sizeof(float) * ksize);
    float sum = 0;
    int origin = ksize / 2;
    for(int i = 0; i < ksize; i++){
        float g = expf(-(i - origin) * (i - origin) / (2 * sigma * sigma));
        g /= (2 * PI * sigma);
        sum += g;
        matrix[i] = g;
    }

    for(int i = 0; i < ksize; i++){
        matrix[i] /= sum;
    }

    int border = ksize / 2;

    unsigned char* tmp = (unsigned char*) calloc(row * col * channel, sizeof(unsigned char));

    for(int i = border; i < row - border; i++){
        for(int j = border; j < col - border; j++){
            float s[3] = {0};
            for(int k = -border; k <= border; k++){
                if(channel == 1){
                    s[0] += matrix[border + k] * (float)src[i * col + j + k];
                } else{
                    s[0] += matrix[border + k] * (float)src[i * col * channel + (j + k) * channel + 0];
                    s[1] += matrix[border + k] * (float)src[i * col * channel + (j + k) * channel + 1];
                    s[2] += matrix[border + k] * (float)src[i * col * channel + (j + k) * channel + 2];
                }
            }

            for(int k = 0; k < channel; k++){
                if(s[k] < 0)
                    s[k] = 0;
                if(s[k] > 255)
                    s[k] = 255;
            }

            if(channel == 1){
                tmp[i * col + j] = (unsigned char)s[0];
            } else{
                tmp[i * col * channel + j * channel + 0] = (unsigned char)s[0];
                tmp[i * col * channel + j * channel + 1] = (unsigned char)s[1];
                tmp[i * col * channel + j * channel + 2] = (unsigned char)s[2];
            }
        }
    }


    for(int i = border; i < row - border; i++){
        for(int j = border; j < col - border; j++){
            float s[3] = {0};
            for(int k = -border; k <= border; k++){
                if(channel == 1){
                    s[0] += matrix[border + k] * (float)tmp[(i + k) * col + j];
                } else{
                    s[0] += matrix[border + k] * (float)tmp[(i + k) * col * channel + j * channel + 0];
                    s[1] += matrix[border + k] * (float)tmp[(i + k) * col * channel + j * channel + 1];
                    s[2] += matrix[border + k] * (float)tmp[(i + k) * col * channel + j * channel + 2];
                }
            }

            for(int k = 0; k < channel; k++){
                if(s[k] < 0)
                    s[k] = 0;
                if(s[k] > 255)
                    s[k] = 255;
            }

            if(channel == 1){
                dst[i * col + j] = (unsigned char)s[0];
            } else{
                dst[i * col * channel + j * channel + 0] = (unsigned char)s[0];
                dst[i * col * channel + j * channel + 1] = (unsigned char)s[1];
                dst[i * col * channel + j * channel + 2] = (unsigned char)s[2];
            }

        }
    }

    free(matrix);
    free(tmp);
}

static void verticalFilterNeonF32(float* src, float* dst, int height, int width, int channel, float* kernel, int ksize)
{
    int kCenter = ksize / 2;

    float* in = (float*) malloc(sizeof(float) * width * (height + ksize) * channel);
    memcpy(in, src, kCenter * width * channel * sizeof(float));
    memcpy(in + kCenter * width * channel, src, height * width * channel * sizeof(float));
    memcpy(in + (kCenter + height) * width * channel, in + height * width * channel, kCenter * width * channel * sizeof(float));

    if(channel == 1){
        for(int i = 0; i < height; i++){
            float* p_dst = dst + i * width * channel;
            int n = width / XA_F32_SIZE;
            int end = n * (XA_F32_SIZE * channel);
            int count = 0;

            for(int j = 0; j < end; j += XA_F32_SIZE){
                float32x4_t accum_f32;
                accum_f32 = vmovq_n_f32(0);

                for(int k = 0; k < ksize; k++){
                    float32x4_t data = vld1q_f32(in + (i + k) * width * channel + j);

                    float32x4_t w = vdupq_n_f32(kernel[k]);

                    accum_f32 = vaddq_f32(accum_f32, vmulq_f32(data, w));
                }

                vst1q_f32(p_dst + j, accum_f32);
                count += XA_F32_SIZE;
            }

            for(int j = count; j < width; j++){
                float s = 0;
                for(int k = 0; k < ksize; k++){
                    s += in[(i + k) * width * channel + j * channel] * kernel[k];
                }

                p_dst[j * channel] = s;
            }
        }
    }else if(channel == 3){
        for(int i = 0; i < height; i++){
            int n = width / XA_F32_SIZE;
            int end = n * XA_F32_SIZE * channel;
            float *p_dst = dst + i * width * channel;
            int count = 0;

            for(int j = 0; j < end; j+=XA_F32_SIZE * channel){
                float32x4_t accum_f32_r, accum_f32_g, accum_f32_b;
                accum_f32_r = vmovq_n_f32(0);
                accum_f32_g = vmovq_n_f32(0);
                accum_f32_b = vmovq_n_f32(0);

                for(int k = 0; k < ksize; k++){
                    float32x4x3_t data = vld3q_f32(in + (i + k) * width * channel + j);

                    // r
                    accum_f32_r = vmlaq_n_f32(accum_f32_r, data.val[0], kernel[k]);
                    // g
                    accum_f32_g = vmlaq_n_f32(accum_f32_g, data.val[1], kernel[k]);
                    // b
                    accum_f32_b = vmlaq_n_f32(accum_f32_b, data.val[2], kernel[k]);
                }

                float32x4x3_t res;
                res.val[0] = accum_f32_r;
                res.val[1] = accum_f32_g;
                res.val[2] = accum_f32_b;

                vst3q_f32(p_dst + j, res);
                count += XA_F32_SIZE * channel;
            }

            for(int j = count / channel; j < width; j++){
                float s[3] = {0};
                for(int k = 0; k < ksize; k++){
                    s[0] += in[(i + k) * width * channel + j * channel + 0] * kernel[k];
                    s[1] += in[(i + k) * width * channel + j * channel + 1] * kernel[k];
                    s[2] += in[(i + k) * width * channel + j * channel + 2] * kernel[k];
                }

                p_dst[j * channel + 0] = s[0];
                p_dst[j * channel + 1] = s[1];
                p_dst[j * channel + 2] = s[2];
            }
        }
    } else{
        free(in);
        return;
    }

    free(in);
}

static void horizonFilterNeonF32(float* src, float* dst, int height, int width, int channel, float* kernel, int ksize)
{
    int kCenter = ksize / 2;
    if(channel == 1){
        for(int i = 0; i < height; i++){
            float * in = (float*) calloc((width + ksize),  sizeof(float));
            memcpy(in, src + i * width, kCenter * sizeof(float));
            memcpy(in + kCenter, src + i * width, width * sizeof(float));
            memcpy(in + width + kCenter, src + i * width + width - kCenter, kCenter * sizeof(float));

            float *p_dst = dst + i * width;
            int n = width / XA_F32_SIZE;
            int end = n * XA_F32_SIZE * channel;
            int count = 0;

            for(int j = 0; j < end; j+=XA_F32_SIZE){
                float32x4_t accum_f32;
                accum_f32 = vmovq_n_f32(0);

                for(int k = 0; k < ksize; k++){
                    float32x4_t data = vld1q_f32(in + j + k);
                    accum_f32 = vmlaq_n_f32(accum_f32, data, kernel[k]);
                }

                vst1q_f32(p_dst + j, accum_f32);
                count += XA_F32_SIZE;
            }

            for(int j = count / channel; j < width; j++){
                float s = 0;
                for(int k = 0; k < ksize; k++){
                    s += *(in + (j + k) * channel) * kernel[k];
                }
                p_dst[j * channel] = s;
            }
            free(in);
        }
    } else if(channel == 3){
        for(int i = 0; i < height; i++){
            float * in = (float*) malloc((width + ksize) * channel * sizeof(float));
            memcpy(in, src + i * width * channel, kCenter * channel * sizeof(float));
            memcpy(in + kCenter * channel, src + i * width * channel, width * channel * sizeof(float));
            memcpy(in + (width + kCenter) * channel, src + i * width * channel + (width - kCenter) * channel, kCenter * channel *
                                                                                                              sizeof(float ));

            float *p_dst = dst + i * width * channel;
            int n = width / XA_F32_SIZE;
            int end = n * (XA_F32_SIZE * channel);
            int count = 0;

            for(int j = 0; j < end; j +=XA_F32_SIZE*channel){
                float32x4_t accum_f32_r, accum_f32_g, accum_f32_b;
                accum_f32_r = vmovq_n_f32(0);
                accum_f32_g = vmovq_n_f32(0);
                accum_f32_b = vmovq_n_f32(0);

                for(int k = 0; k < ksize; k++){
                    float32x4x3_t data = vld3q_f32(in + j + k * channel);
                    // r
                    accum_f32_r = vmlaq_n_f32(accum_f32_r, data.val[0], kernel[k]);
                    // g
                    accum_f32_g = vmlaq_n_f32(accum_f32_g, data.val[1], kernel[k]);
                    // b
                    accum_f32_b = vmlaq_n_f32(accum_f32_b, data.val[2], kernel[k]);
                }

                float32x4x3_t res;
                res.val[0] = accum_f32_r;
                res.val[1] = accum_f32_g;
                res.val[2] = accum_f32_b;
                vst3q_f32(p_dst + j, res);
                count += XA_F32_SIZE * channel;
            }

            for(int j = count / 3; j < width; j++){
                float s[3] = {0};
                for(int k = 0; k < ksize; k++){
                    s[0] += *(in + (j + k) * channel + 0) * kernel[k];
                    s[1] += *(in + (j + k) * channel + 1) * kernel[k];
                    s[2] += *(in + (j + k) * channel + 2) * kernel[k];
                }
            }
            free(in);
        }
    } else{
        return;
    }

}

void gaussianFilter_float_Neon(float* src, float* dst, int height, int width, int channel, int ksize, float sigma)
{
    if(ksize > MAX_KERNEL_SIZE){
        return;
    }

    float *kernel = (float *) malloc(sizeof(float ) * ksize);
    getGaussianKernel1D(kernel, ksize, sigma);

    verticalFilterNeonF32(src, dst, height, width, channel, kernel, ksize);
    horizonFilterNeonF32(dst, dst, height, width, channel, kernel, ksize);

    free(kernel);
}
