// File: filter.c
// Brief: 图像高斯滤波
// Author: Lixiaolong@xa.com
// Date: 2022/6/30
//
#include <strings.h>
#include "stdlib.h"
#include "math.h"
#include "gaussFilter.h"
#include "arm_neon.h"

#define MAX_KERNEL_SIZE   79
#define XA_U8_SIZE        16
#define XA_F32_SIZE       4
#define PI 3.14159


static void getGaussianKernel1D(float *kernel, int ksize, float sigma);

static void getGaussianKernel2D(float *kernel, int ksize, float sigma);

static void generateGaussianKernel(float **kernel, int ksize, float sigma);

static void verticalFilterNeonU8(U8* src, U8* dst, int height, int width, int channel, float* kernel, int ksize);

static void horizonFilterNeonU8(U8* src, U8* dst, int height, int width, int channel, float* kernel, int ksize);

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

static void verticalFilterNeonU8(U8* src, U8* dst, int height, int width, int channel, float* kernel, int ksize)
{
    int kCenter = ksize / 2;

    unsigned char* in = (unsigned char*) calloc(sizeof(unsigned char), width * (height + ksize) * channel);
    memcpy(in + kCenter * width * channel, src, height * width * channel);

    if(channel == 1){
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j += XA_U8_SIZE){
                uint16x8_t laccum_u16, haccum_u16;
                laccum_u16 = vmovq_n_u16(0);
                haccum_u16 = vmovq_n_u16(0);

                for(int k = 0; k < ksize; k++){
                    uint8x16_t data = vld1q_u8(in + (i + k) * width * channel + j);
                    uint8x8_t lp_u8 = vget_low_u8( data );
                    uint8x8_t hp_u8 = vget_high_u8( data );

                    uint16x8_t lp_u16 = vmovl_u8( lp_u8 );
                    uint16x8_t hp_u16 = vmovl_u8( hp_u8 );

                    float16x8_t w = vdupq_n_f16(kernel[k]);

                    laccum_u16 = vaddq_u16(laccum_u16, vcvtq_u16_f16(vmulq_f16(vcvtq_f16_u16(lp_u16), w)));
                    haccum_u16 = vaddq_u16(haccum_u16, vcvtq_u16_f16(vmulq_f16(vcvtq_f16_u16(hp_u16), w)));
                }

                uint8x8_t laccum_u8 = vmovn_u16( laccum_u16 );
                uint8x8_t haccum_u8 = vmovn_u16( haccum_u16 );
                uint8x16_t accum_u8 = vcombine_u8( laccum_u8, haccum_u8 );
                vst1q_u8(dst, accum_u8);
                dst += XA_U8_SIZE;
            }
        }
    } else if(channel == 3){
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width * channel; j+= XA_U8_SIZE * channel){
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

                    float16x8_t w = vdupq_n_f16(kernel[k]);

                    // r
                    lp_u8 = vget_low_u8( data.val[0]);
                    hp_u8 = vget_high_u8( data.val[0]);

                    lp_u16 = vmovl_u8( lp_u8 );
                    hp_u16 = vmovl_u8( hp_u8 );

                    laccum_u16_r = vaddq_u16(laccum_u16_r, vcvtq_u16_f16(vmulq_f16(vcvtq_f16_u16(lp_u16), w)));
                    haccum_u16_r = vaddq_u16(haccum_u16_r, vcvtq_u16_f16(vmulq_f16(vcvtq_f16_u16(hp_u16), w)));

                    // g
                    lp_u8 = vget_low_u8( data.val[1]);
                    hp_u8 = vget_high_u8( data.val[1]);

                    lp_u16 = vmovl_u8( lp_u8 );
                    hp_u16 = vmovl_u8( hp_u8 );

                    laccum_u16_g = vaddq_u16(laccum_u16_g, vcvtq_u16_f16(vmulq_f16(vcvtq_f16_u16(lp_u16), w)));
                    haccum_u16_g = vaddq_u16(haccum_u16_g, vcvtq_u16_f16(vmulq_f16(vcvtq_f16_u16(hp_u16), w)));

                    // b
                    lp_u8 = vget_low_u8( data.val[2]);
                    hp_u8 = vget_high_u8( data.val[2]);

                    lp_u16 = vmovl_u8( lp_u8 );
                    hp_u16 = vmovl_u8( hp_u8 );

                    laccum_u16_b = vaddq_u16(laccum_u16_b, vcvtq_u16_f16(vmulq_f16(vcvtq_f16_u16(lp_u16), w)));
                    haccum_u16_b = vaddq_u16(haccum_u16_b, vcvtq_u16_f16(vmulq_f16(vcvtq_f16_u16(hp_u16), w)));
                }
                uint8x8_t laccum_u8_r = vmovn_u16( laccum_u16_r );
                uint8x8_t haccum_u8_r = vmovn_u16( haccum_u16_r );
                uint8x16_t accum_u8_r = vcombine_u8( laccum_u8_r, haccum_u8_r);

                uint8x8_t laccum_u8_g = vmovn_u16( laccum_u16_g );
                uint8x8_t haccum_u8_g = vmovn_u16( haccum_u16_g );
                uint8x16_t accum_u8_g = vcombine_u8( laccum_u8_g, haccum_u8_g);

                uint8x8_t laccum_u8_b = vmovn_u16( laccum_u16_b );
                uint8x8_t haccum_u8_b = vmovn_u16( haccum_u16_b );
                uint8x16_t accum_u8_b = vcombine_u8( laccum_u8_b, haccum_u8_b);

                uint8x16x3_t res;
                res.val[0] = accum_u8_r;
                res.val[1] = accum_u8_g;
                res.val[2] = accum_u8_b;

                vst3q_u8(dst, res);
                dst += XA_U8_SIZE * channel;
            }
        }
    } else{
        free(in);
        return;
    }
    free(in);
}

static void horizonFilterNeonU8(U8* src, U8* dst, int height, int width, int channel, float* kernel, int ksize)
{
    int kCenter = ksize / 2;
    if (channel == 1){
        for(int i = 0; i < height; i++){
            unsigned char* in = (unsigned char*) calloc((width + ksize), sizeof(unsigned char));
            memcpy(in + kCenter, src + i * width, width);

            unsigned char* p_dst = dst + i * width;

            for(int j = 0; j < width; j+=XA_U8_SIZE){

                uint16x8_t laccum_u16, haccum_u16;
                laccum_u16 = vmovq_n_u16(0);
                haccum_u16 = vmovq_n_u16(0);

                for(int k = 0; k < ksize; k++){
                    uint8x16_t data = vld1q_u8(in + j + k);
                    uint8x8_t lp_u8 = vget_low_u8( data );
                    uint8x8_t hp_u8 = vget_high_u8( data );

                    uint16x8_t lp_u16 = vmovl_u8( lp_u8 );
                    uint16x8_t hp_u16 = vmovl_u8( hp_u8 );

                    float16x8_t w = vdupq_n_f16(kernel[k]);

                    laccum_u16 = vaddq_u16(laccum_u16, vcvtq_u16_f16(vmulq_f16(vcvtq_f16_u16(lp_u16), w)));
                    haccum_u16 = vaddq_u16(haccum_u16, vcvtq_u16_f16(vmulq_f16(vcvtq_f16_u16(hp_u16), w)));

                }
                uint8x8_t laccum_u8 = vmovn_u16( laccum_u16 );
                uint8x8_t haccum_u8 = vmovn_u16( haccum_u16 );
                uint8x16_t accum_u8 = vcombine_u8( laccum_u8, haccum_u8 );
                vst1q_u8(p_dst, accum_u8);
                p_dst += XA_U8_SIZE;
            }

            free(in);
        }
    } else if(channel == 3){
        for(int i = 0; i < height; i++){
            U8* in = (U8*) calloc((width + ksize) * channel, sizeof(U8));
            memcpy(in + kCenter * channel, src + i * width * channel, width * channel);

            U8* p_dst = dst + i * width * channel;

            for(int j = 0; j < width * channel; j += XA_U8_SIZE * channel){
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

                    float16x8_t w = vdupq_n_f16(kernel[k]);
                    // r
                    lp_u8 = vget_low_u8( data.val[0]);
                    hp_u8 = vget_high_u8( data.val[0]);

                    lp_u16 = vmovl_u8( lp_u8 );
                    hp_u16 = vmovl_u8( hp_u8 );

                    laccum_u16_r = vaddq_u16(laccum_u16_r, vcvtq_u16_f16(vmulq_f16(vcvtq_f16_u16(lp_u16), w)));
                    haccum_u16_r = vaddq_u16(haccum_u16_r, vcvtq_u16_f16(vmulq_f16(vcvtq_f16_u16(hp_u16), w)));

                    // g
                    lp_u8 = vget_low_u8( data.val[1]);
                    hp_u8 = vget_high_u8( data.val[1]);

                    lp_u16 = vmovl_u8( lp_u8 );
                    hp_u16 = vmovl_u8( hp_u8 );

                    laccum_u16_g = vaddq_u16(laccum_u16_g, vcvtq_u16_f16(vmulq_f16(vcvtq_f16_u16(lp_u16), w)));
                    haccum_u16_g = vaddq_u16(haccum_u16_g, vcvtq_u16_f16(vmulq_f16(vcvtq_f16_u16(hp_u16), w)));

                    // b
                    lp_u8 = vget_low_u8( data.val[2]);
                    hp_u8 = vget_high_u8( data.val[2]);

                    lp_u16 = vmovl_u8( lp_u8 );
                    hp_u16 = vmovl_u8( hp_u8 );

                    laccum_u16_b = vaddq_u16(laccum_u16_b, vcvtq_u16_f16(vmulq_f16(vcvtq_f16_u16(lp_u16), w)));
                    haccum_u16_b = vaddq_u16(haccum_u16_b, vcvtq_u16_f16(vmulq_f16(vcvtq_f16_u16(hp_u16), w)));
                }

                uint8x8_t laccum_u8_r = vmovn_u16( laccum_u16_r );
                uint8x8_t haccum_u8_r = vmovn_u16( haccum_u16_r );
                uint8x16_t accum_u8_r = vcombine_u8( laccum_u8_r, haccum_u8_r);

                uint8x8_t laccum_u8_g = vmovn_u16( laccum_u16_g );
                uint8x8_t haccum_u8_g = vmovn_u16( haccum_u16_g );
                uint8x16_t accum_u8_g = vcombine_u8( laccum_u8_g, haccum_u8_g);

                uint8x8_t laccum_u8_b = vmovn_u16( laccum_u16_b );
                uint8x8_t haccum_u8_b = vmovn_u16( haccum_u16_b );
                uint8x16_t accum_u8_b = vcombine_u8( laccum_u8_b, haccum_u8_b);

                uint8x16x3_t res;
                res.val[0] = accum_u8_r;
                res.val[1] = accum_u8_g;
                res.val[2] = accum_u8_b;

                vst3q_u8(p_dst, res);
                p_dst += XA_U8_SIZE * channel;
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
    float *kernel = (float *) malloc(sizeof(float) * ksize);
    getGaussianKernel1D(kernel, ksize, sigma);

    U8* temp = (U8*) malloc(sizeof(U8) * height * width * channel);
    verticalFilterNeonU8(src, temp, height, width, channel, kernel, ksize);
    horizonFilterNeonU8(temp, dst, height, width, channel, kernel, ksize);

    free(kernel);
    free(temp);
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

    float* in = (float*) calloc(sizeof(float), width * (height + ksize) * channel);
    memcpy(in, src, kCenter * width * channel * sizeof(float));
    memcpy(in + kCenter * width * channel, src, height * width * channel * sizeof(float));
    memcpy(in + (kCenter + height) * width * channel, in + height * width * channel, kCenter * width * channel * sizeof(float));

    if(channel == 1){
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j += 4){
                float32x4_t accum_f32;
                accum_f32 = vmovq_n_f32(0);

                for(int k = 0; k < ksize; k++){
                    float32x4_t data = vld1q_f32(in + (i + k) * width + j);

                    float32x4_t w = vdupq_n_f32(kernel[k]);

                    accum_f32 = vaddq_f32(accum_f32, vmulq_f32(data, w));
                }

                vst1q_f32(dst, accum_f32);
                dst += 4;
            }
        }
    }else if(channel == 3){
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width * channel; j+=12){
                float32x4_t accum_f32_r, accum_f32_g, accum_f32_b;
                accum_f32_r = vmovq_n_f32(0);
                accum_f32_g = vmovq_n_f32(0);
                accum_f32_b = vmovq_n_f32(0);

                for(int k = 0; k < ksize; k++){
                    float32x4x3_t data = vld3q_f32(in + (i + k) * width * channel + j);

                    float32x4_t w = vdupq_n_f32(kernel[k]);

                    // r
                    accum_f32_r = vaddq_f32(accum_f32_r, vmulq_f32(data.val[0], w));
                    // g
                    accum_f32_g = vaddq_f32(accum_f32_g, vmulq_f32(data.val[1], w));
                    // b
                    accum_f32_b = vaddq_f32(accum_f32_b, vmulq_f32(data.val[2], w));
                }

                float32x4x3_t res;
                res.val[0] = accum_f32_r;
                res.val[1] = accum_f32_g;
                res.val[2] = accum_f32_b;

                vst3q_f32(dst, res);
                dst += 12;
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

            for(int j = 0; j < width; j+=4){
                float32x4_t accum_f32;
                accum_f32 = vmovq_n_f32(0);

                for(int k = 0; k < ksize; k++){
                    float32x4_t data = vld1q_f32(in + j + k);
                    float32x4_t w = vdupq_n_f32(kernel[k]);

                    accum_f32 = vaddq_f32(accum_f32, vmulq_f32(data, w));
                }

                vst1q_f32(p_dst, accum_f32);
                p_dst += 4;
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

            for(int j = 0; j < width * channel; j +=12){
                float32x4_t accum_f32_r, accum_f32_g, accum_f32_b;
                accum_f32_r = vmovq_n_f32(0);
                accum_f32_g = vmovq_n_f32(0);
                accum_f32_b = vmovq_n_f32(0);

                for(int k = 0; k < ksize; k++){
                    float32x4x3_t data = vld3q_f32(in + j + k * channel);

                    float32x4_t w = vdupq_n_f32(kernel[k]);

                    // r
                    accum_f32_r = vaddq_f32(accum_f32_r, vmulq_f32(data.val[0], w));
                    // g
                    accum_f32_g = vaddq_f32(accum_f32_g, vmulq_f32(data.val[1], w));
                    // b
                    accum_f32_b = vaddq_f32(accum_f32_b, vmulq_f32(data.val[2], w));
                }

                float32x4x3_t res;
                res.val[0] = accum_f32_r;
                res.val[1] = accum_f32_g;
                res.val[2] = accum_f32_b;
                vst3q_f32(p_dst, res);
                p_dst += 12;
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

    float *temp = (float*) malloc(sizeof(float) * height * width * channel);
    verticalFilterNeonF32(src, temp, height, width, channel, kernel, ksize);
    horizonFilterNeonF32(temp, dst, height, width, channel, kernel, ksize);

    free(kernel);
    free(temp);
}
