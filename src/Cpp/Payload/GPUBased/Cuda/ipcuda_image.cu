/**
  * @file ipcuda_image.h
  * @author Anh Dan <danda@viettel.com.vn/anhdan.do@gmail.com>
  * @version 1.0
  * @date 25 Jan 2019
  *
  * @section LICENSE
  *
  * @section DESCRIPTION
  *
  * This file contains the declarations of CUDA functions that perform
  * image color conversions between I420, Gray, RGB spaces; get image
  * patch from a rotated ROI on a larger image and histogram functions
  *
  * @see ipcuda_image.cu
  */

#include "ipcuda_image.h"
#include "ip_utils.h"


/*=================================================
 *
 *  I420 to Gray conversion device function
 *
 *===============================================*/

__global__ void cuda_i420ToGray( unsigned char *d_i420, unsigned char *d_gray, int width, int height )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int numel = width * height;

    if( index < numel )
    {
        d_gray[index] = d_i420[index];
    }
}


/**
 * @brief dev_i420ToGray
 */
cudaError_t gpu_i420ToGray( unsigned char *d_i420, unsigned char *d_gray, int width, int height )
{
    if( (d_i420 == NULL) || (d_gray == NULL) )
    {
        LOG_MSG( "! ERROR: %s:%d: err = %d: NULL device pointer\n", __FUNCTION__, __LINE__, cudaErrorIllegalAddress );
        return cudaErrorIllegalAddress;
    }

    int     numel = width * height;
    cuda_i420ToGray<<<(numel+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>( d_i420, d_gray, width, height );
    cudaDeviceSynchronize();

    return cudaSuccess;
}


/*=================================================
 *
 *  I420 to RGB conversion device function
 *
 *===============================================*/
__global__ void cuda_i420ToRGB( const unsigned char *d_i420, unsigned char *d_rgb, int width, int height,
                                int roi_x, int roi_y, int roi_w, int roi_h )
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int numel  = roi_w * roi_h;
    int imgLen = width * height;

    //
    //===== Convert i420 color format to RGBA of current pixel
    //
    if( index < numel )
    {
        int     row = index / roi_w,
                col = index - row * roi_w;

        int     imgCol = col + roi_x,
                imgRow = row + roi_y;

        int     imgIndex = imgRow * width + imgCol;

        int     yidx = imgIndex,
                uidx = (imgRow >> 1) * (width >> 1) + (imgCol >> 1) + imgLen,
                vidx = uidx + (imgLen >> 2),
                rgbIdx = index * 3;

        // i420 to rgba
        int     y = (int)d_i420[yidx] - 16,
                u = (int)d_i420[uidx] - 128,
                v = (int)d_i420[vidx] - 128;

        int     r = (298*y + 409*u + 128) >> 8,
                g = (298*y - 100*u - 208*v + 128) >> 8,
                b = (298*y + 516*v + 128) >> 8;

        d_rgb[rgbIdx]   = (r < 0)? 0 : ((r > 255)? 255 : (unsigned char)r);
        d_rgb[rgbIdx+1] = (g < 0)? 0 : ((g > 255)? 255 : (unsigned char)g);
        d_rgb[rgbIdx+2] = (b < 0)? 0 : ((b > 255)? 255 : (unsigned char)b);
    }
}


/**
 * @brief dev_i420ToRGB
 */
cudaError_t gpu_i420ToRGB( const unsigned char *d_i420, unsigned char *d_rgb, int width, int height,
                           int roi_x, int roi_y, int roi_w, int roi_h )
{
    if( d_i420 == NULL || d_rgb == NULL )
        return cudaErrorInvalidDevicePointer;

    if( (roi_x < 0) || (roi_x + roi_w > width) )
        return cudaErrorInvalidValue;

    if( (roi_y < 0) || (roi_y + roi_h > height) )
        return cudaErrorInvalidValue;

    int numel = roi_w * roi_h;
    cuda_i420ToRGB<<<(numel+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>( d_i420, d_rgb, width, height, roi_x, roi_y, roi_w, roi_h );
    cudaDeviceSynchronize();

    return cudaSuccess;
}


/*=================================================
 *
 *  I420 to RGBA conversion device function
 *
 *===============================================*/
__global__ void cuda_i420ToRGBA( const unsigned char *d_i420, unsigned char *d_rgba, int width, int height,
                                 int roi_x, int roi_y, int roi_w, int roi_h )
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int numel  = roi_w * roi_h;
    int imgLen = width * height;

    //
    //===== Convert i420 color format to RGBA of current pixel
    //
    if( index < numel )
    {
        int     row = index / roi_w,
                col = index - row * roi_w;

        int     imgCol = col + roi_x,
                imgRow = row + roi_y;

        int     imgIndex = imgRow * width + imgCol;

        int     yidx = imgIndex,
                uidx = (imgRow >> 1) * (width >> 1) + (imgCol >> 1) + imgLen,
                vidx = uidx + (imgLen >> 2),
                rgbIdx = index * 4;

        // i420 to rgba
        int     y = (int)d_i420[yidx] - 16,
                u = (int)d_i420[uidx] - 128,
                v = (int)d_i420[vidx] - 128;

        int     r = (298*y + 409*u + 128) >> 8,
                g = (298*y - 100*u - 208*v + 128) >> 8,
                b = (298*y + 516*v + 128) >> 8;

        d_rgba[rgbIdx]   = (r < 0)? 0 : ((r > 255)? 255 : (unsigned char)r);
        d_rgba[rgbIdx+1] = (g < 0)? 0 : ((g > 255)? 255 : (unsigned char)g);
        d_rgba[rgbIdx+2] = (b < 0)? 0 : ((b > 255)? 255 : (unsigned char)b);
        d_rgba[rgbIdx+3] = 255;
    }
}


/**
 * @brief dev_i420ToRGB
 */
cudaError_t gpu_i420ToRGBA( const unsigned char *d_i420, unsigned char *d_rgba, int width, int height,
                           int roi_x, int roi_y, int roi_w, int roi_h )
{
    if( d_i420 == NULL || d_rgba == NULL )
        return cudaErrorInvalidDevicePointer;

    if( (roi_x < 0) || (roi_x + roi_w > width) )
        return cudaErrorInvalidValue;

    if( (roi_y < 0) || (roi_y + roi_h > height) )
        return cudaErrorInvalidValue;

    int numel = roi_w * roi_h;
    cuda_i420ToRGBA<<<(numel+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>( d_i420, d_rgba, width, height, roi_x, roi_y, roi_w, roi_h );
    cudaDeviceSynchronize();

    return cudaSuccess;
}


/*=================================================
 *
 *  RGB to I420 conversion device function
 *
 *===============================================*/
__global__ void cuda_rgbToI420( const unsigned char *d_rgb, unsigned char *d_i420, int width, int height )
{
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int len = width * height;

    //
    //===== Convert RGB color format to I420 of current pixel
    //
    if( idx < len )
    {
        int row    = idx / width,
            col    = idx % width,
            rgbIdx = idx * 3;

        int r = d_rgb[rgbIdx],
            g = d_rgb[rgbIdx + 1],
            b = d_rgb[rgbIdx + 2];

        // Compute Y component
        int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        d_i420[idx] = (y < 0) ? 0 : ((r > 255) ? 255 : (unsigned char)y);

        // Copute U & V components
        if( (row & 0x01) && (col & 0x01) )
        {
            // Compute average color of 4 pixels sharing the same UV component
            int rgbIdx1 = rgbIdx - width * 3 - 3,
                rgbIdx2 = rgbIdx1 + 3,
                rgbIdx3 = rgbIdx - 3;

            int avgR = (r + (int)d_rgb[rgbIdx1] + (int)d_rgb[rgbIdx2] + (int)d_rgb[rgbIdx3]) >> 2,
                avgG = (g + (int)d_rgb[rgbIdx1+1] + (int)d_rgb[rgbIdx2+1] + (int)d_rgb[rgbIdx3+1]) >> 2,
                avgB = (b + (int)d_rgb[rgbIdx1+2] + (int)d_rgb[rgbIdx2+2] + (int)d_rgb[rgbIdx3+2]) >> 2;

            // Compute U, V indice and values
            int uidx = (row >> 1) * (width >> 1) + (col >> 1) + len,
                vidx = uidx + (len >> 2);

            int v = ((-38 * avgR - 74 * avgG + 112 * avgB + 128) >> 8) + 128,
                u = ((112 * avgR - 94 * avgG - 18 * avgB + 128) >> 8) + 128;

            d_i420[uidx] = (u < 0) ? 0 : ((u > 255) ? 255 : (unsigned char)u);
            d_i420[vidx] = (v < 0) ? 0 : ((v > 255) ? 255 : (unsigned char)v);
        }
    }
}


/**
 * @brief dev_i420ToRGB
 */
cudaError_t gpu_rgbToI420( const unsigned char *d_rgb, unsigned char *d_i420, int width, int height )
{
    if( d_i420 == NULL || d_rgb == NULL )
        return cudaErrorInvalidDevicePointer;

    int numel = width * height;
    cuda_rgbToI420<<<(numel+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>( d_rgb, d_i420, width, height );
    cudaDeviceSynchronize();

    return cudaSuccess;
}


/*=================================================
 *
 *  RGBA to I420 conversion device function
 *
 *===============================================*/
__global__ void cuda_rgbaToI420( const unsigned char *d_rgba, unsigned char *d_i420, int width, int height )
{
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int len = width * height;

    //
    //===== Convert RGB color format to I420 of current pixel
    //
    if( idx < len )
    {
        int row    = idx / width,
            col    = idx % width,
            rgbIdx = idx * 4;

        int r = d_rgba[rgbIdx],
            g = d_rgba[rgbIdx + 1],
            b = d_rgba[rgbIdx + 2];

        // Compute Y component
        int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        d_i420[idx] = (y < 0) ? 0 : ((r > 255) ? 255 : (unsigned char)y);

        // Copute U & V components
        if( (row & 0x01) && (col & 0x01) )
        {
            // Compute average color of 4 pixels sharing the same UV component
            int rgbIdx1 = rgbIdx - width * 4 - 4,
                rgbIdx2 = rgbIdx1 + 4,
                rgbIdx3 = rgbIdx - 4;

            int avgR = (r + (int)d_rgba[rgbIdx1] + (int)d_rgba[rgbIdx2] + (int)d_rgba[rgbIdx3]) >> 2,
                avgG = (g + (int)d_rgba[rgbIdx1+1] + (int)d_rgba[rgbIdx2+1] + (int)d_rgba[rgbIdx3+1]) >> 2,
                avgB = (b + (int)d_rgba[rgbIdx1+2] + (int)d_rgba[rgbIdx2+2] + (int)d_rgba[rgbIdx3+2]) >> 2;

            // Compute U, V indice and values
            int uidx = (row >> 1) * (width >> 1) + (col >> 1) + len,
                vidx = uidx + (len >> 2);

            int v = ((-38 * avgR - 74 * avgG + 112 * avgB + 128) >> 8) + 128,
                u = ((112 * avgR - 94 * avgG - 18 * avgB + 128) >> 8) + 128;

            d_i420[uidx] = (u < 0) ? 0 : ((u > 255) ? 255 : (unsigned char)u);
            d_i420[vidx] = (v < 0) ? 0 : ((v > 255) ? 255 : (unsigned char)v);
        }
    }
}


/**
 * @brief gpu_rgbaToI420
 */
cudaError_t gpu_rgbaToI420( const unsigned char *d_rgba, unsigned char *d_i420, int width, int height )
{
    if( d_i420 == NULL || d_rgba == NULL )
        return cudaErrorInvalidDevicePointer;

    int numel = width * height;
    cuda_rgbaToI420<<<(numel+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>( d_rgba, d_i420, width, height );
    cudaDeviceSynchronize();

    return cudaSuccess;
}


/*=================================================
 *
 *  UYVY to I420 conversion
 *
 *===============================================*/
__global__ void cuda_uyvy2I420( const unsigned char *d_uyvy, unsigned char *d_i420, const int _width, const int _height )
{
    int len = _width * _height;
    int procIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if( procIdx < len )
    {
        int row = procIdx / _width,
            col = procIdx % _width;

        // Retrieve Y component
        int srcYIdx = ((row * _width) + col) * 2 + 1;
        d_i420[procIdx] = d_uyvy[srcYIdx];

        // Retrieve U, V component
        if( (row & 0x01) && (col & 0x01) )
        {
            int dstUIdx = (row >> 1) * (_width >> 1) + (col >> 1) + len;
            int dstVIdx = dstUIdx + (len >> 2);

            int srcUIdx1 = srcYIdx - 3,
                srcVIdx1 = srcYIdx - 1;
            int srcUIdx2 = srcUIdx1 - (_width << 1),
                srcVIdx2 = srcVIdx1 - (_width << 1);

            d_i420[dstUIdx] = (unsigned char)(((int)d_uyvy[srcUIdx1] + (int)d_uyvy[srcUIdx2]) >> 1);
            d_i420[dstVIdx] = (unsigned char)(((int)d_uyvy[srcVIdx1] + (int)d_uyvy[srcVIdx2]) >> 1);
        }
    }
}


/**
 * @brief gpu_uyvy2I420
 */
cudaError_t gpu_uyvy2I420( const unsigned char *d_uyvy, unsigned char *d_i420, const int _width, const int _height )
{
    if( d_uyvy == NULL || d_i420 == NULL )
    {
        printf( "!Error: %s:%d: Invalid device pointers\n", __FUNCTION__, __LINE__ );
        return cudaErrorInvalidDevicePointer;
    }

    if( (_width & 0x1) || (_height & 0x1) )
    {
        printf( "!Error: %s:%d: Invalid image dimension\n", __FUNCTION__, __LINE__ );
        return cudaErrorInvalidValue;
    }

    int len = _width * _height;
    int threadPerBlk = THREAD_PER_BLOCK * 1;
    cuda_uyvy2I420<<<(len+threadPerBlk-1) / threadPerBlk, threadPerBlk>>>( d_uyvy, d_i420, _width, _height );
    cudaDeviceSynchronize();

    return cudaSuccess;
}


/*=================================================
 *
 *  UYVY to I420 conversion
 *
 *===============================================*/
__global__ void cuda_yuyv2I420( const unsigned char *d_yuyv, unsigned char *d_i420, const int _width, const int _height )
{
    int len = _width * _height;
    int procIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if( procIdx < len )
    {
        int row = procIdx / _width,
            col = procIdx % _width;

        // Retrieve Y component
        int srcYIdx = ((row * _width) + col) * 2;
        d_i420[procIdx] = d_yuyv[srcYIdx];

        // Retrieve U, V component
        if( (row & 0x01) && (col & 0x01) )
        {
            int dstUIdx = (row >> 1) * (_width >> 1) + (col >> 1) + len;
            int dstVIdx = dstUIdx + (len >> 2);

            int srcUIdx1 = srcYIdx + 1,
                srcVIdx1 = srcYIdx + 3;
            int srcUIdx2 = srcUIdx1 - (_width << 1),
                srcVIdx2 = srcVIdx1 - (_width << 1);

            d_i420[dstUIdx] = (unsigned char)(((int)d_yuyv[srcUIdx1] + (int)d_yuyv[srcUIdx2]) >> 1);
            d_i420[dstVIdx] = (unsigned char)(((int)d_yuyv[srcVIdx1] + (int)d_yuyv[srcVIdx2]) >> 1);
        }
    }
}


/**
 * @brief gpu_uyvy2I420
 */
cudaError_t gpu_yuyv2I420( const unsigned char *d_yuyv, unsigned char *d_i420, const int _width, const int _height )
{
    if( d_yuyv == NULL || d_i420 == NULL )
    {
        printf( "!Error: %s:%d: Invalid device pointers\n", __FUNCTION__, __LINE__ );
        return cudaErrorInvalidDevicePointer;
    }

    if( (_width & 0x1) || (_height & 0x1) )
    {
        printf( "!Error: %s:%d: Invalid image dimension\n", __FUNCTION__, __LINE__ );
        return cudaErrorInvalidValue;
    }

    int len = _width * _height;
    int threadPerBlk = THREAD_PER_BLOCK * 1;
    cuda_yuyv2I420<<<(len+threadPerBlk-1) / threadPerBlk, threadPerBlk>>>( d_yuyv, d_i420, _width, _height );
    cudaDeviceSynchronize();

    return cudaSuccess;
}


/*=================================================
 *
 *  UYVY to I420 conversion and resize
 *
 *===============================================*/
__global__ void cuda_uyvy4K2I4204K2K( const unsigned char *d_uyvy4K, unsigned char *d_i4204K, unsigned char *d_i4202K,
                                      const int _width4K, const int _height4K, const int _width2K, const int _height2K )
{
    int len4K = _width4K * _height4K;
    int len2K = _width2K * _height2K;
    int procIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if( procIdx < len4K )
    {
        int row = procIdx / _width4K,
            col = procIdx % _width4K;

        // Retrieve Y component
        int srcYIdx = ((row * _width4K) + col) * 2 + 1;
        d_i4204K[procIdx] = d_uyvy4K[srcYIdx];

        // Retrieve U, V component
        if( (row & 0x01) && (col & 0x01) )
        {
            // Convert to I420
            int dstUIdx = (row >> 1) * (_width4K >> 1) + (col >> 1) + len4K;
            int dstVIdx = dstUIdx + (len4K >> 2);

            int srcUIdx1 = srcYIdx - 3,
                srcVIdx1 = srcYIdx - 1;
            int srcUIdx2 = srcUIdx1 - (_width4K << 1),
                srcVIdx2 = srcVIdx1 - (_width4K << 1);

            d_i4204K[dstUIdx] = (unsigned char)(((int)d_uyvy4K[srcUIdx1] + (int)d_uyvy4K[srcUIdx2]) >> 1);
            d_i4204K[dstVIdx] = (unsigned char)(((int)d_uyvy4K[srcVIdx1] + (int)d_uyvy4K[srcVIdx2]) >> 1);

            // Resize to 2K
            int col2K = col >> 1,
                row2K = row >> 1;
            int yIdx2K = row2K * _width2K + col2K,
                uIdx2K = (row2K >>1) * (_width2K >> 1) + (col2K >> 1) + len2K;
            int vIdx2K = uIdx2K + (len2K >> 2);

            d_i4202K[yIdx2K] = d_i4204K[procIdx];
            d_i4202K[uIdx2K] = d_i4204K[dstUIdx];
            d_i4202K[vIdx2K] = d_i4204K[dstVIdx];
        }
    }
}


/**
 * @brief gpu_uyvy2I420
 */
cudaError_t gpu_uyvy4K2I4204K2K( const unsigned char *d_uyvy4K, unsigned char *d_i4204K, unsigned char *d_i4202K )
{
    if( d_uyvy4K == NULL || d_i4204K == NULL || d_i4202K == NULL )
    {
        printf( "!Error: %s:%d: Invalid device pointers\n", __FUNCTION__, __LINE__ );
        return cudaErrorInvalidDevicePointer;
    }

    int _width4K  = 3840,
        _height4K = 2160,
        _width2K  = 1920,
        _height2K = 1080;
    int len = _width4K * _height4K;
    int threadPerBlk = THREAD_PER_BLOCK * 1;
    cuda_uyvy4K2I4204K2K<<<(len+threadPerBlk-1) / threadPerBlk, threadPerBlk>>>( d_uyvy4K, d_i4204K, d_i4202K, _width4K, _height4K, _width2K, _height2K );
    cudaDeviceSynchronize();

    return cudaSuccess;
}



/*=================================================
 *
 *  Convert a gray image to float matrix
 *
 *===============================================*/
__global__ void cuda_im2float( unsigned char *_ucharIm, float *_floatIm, const int _rows, const int _cols )
{
    int len = _rows * _cols;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < len )
    {
        _floatIm[idx] = (float)_ucharIm[idx] / 255.0;
    }
}


/**
 * @brief gpu_im2float
 */
cudaError_t gpu_im2float( unsigned char *_ucharIm, float *_floatIm, const int _rows, const int _cols )
{
    if( _ucharIm == NULL || _floatIm == NULL )
        return cudaErrorInvalidDevicePointer;

    if( (_cols <= 0) || (_rows <= 0) )
        return cudaErrorInvalidValue;

    int len = _rows * _cols;
    cuda_im2float<<<(len+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>( _ucharIm, _floatIm, _rows, _cols );
    cudaDeviceSynchronize();

    return cudaSuccess;
}


/*=================================================
 *
 *  I420 Image Affine Warping
 *
 *===============================================*/
__global__ void cuda_invWarpI420( const unsigned char *d_src, unsigned char *d_dst, const float *d_T, const int _width, const int _height )
{
    int len = _width * _height;
    int procIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if( procIdx < len )
    {
        // Compute index of pixel in d_src that is corresponding to current pixel in d_dst
        int row  = procIdx / _width,
            col  = procIdx % _width;
        float x  = (float)col + 0.5,
              y  = (float)row + 0.5;
        float invX = d_T[0] * x + d_T[3] * y + d_T[6],
              invY = d_T[1] * x + d_T[4] * y + d_T[7];

        // Compute index of 4 nearest neighbours of th imaginary source pixel
        int row0 = (invY >= 0) ? (int)invY : ((int)invY - 1),
            col0 = (invX >= 0) ? (int)invX : ((int)invX - 1);

        float dx = invX - (float)col0 - 0.5,
              dy = invY - (float)row0 - 0.5;
        int dR   = (dy < 0) ? -1 : 1,
            dC   = (dx < 0) ? -1 : 1;

        int row1 = row0,
            col1 = col0 + dC,
            row2 = row0 + dR,
            col2 = col0 + dC,
            row3 = row0 + dR,
            col3 = col0;

        // Compute bilinear weights
        float alphaX = 1.0 - fabsf(dx),
              alphaY = 1.0 - fabsf(dy);
        float betaX  = 1.0 - alphaX,
              betaY  = 1.0 - alphaY;
        float w0 = alphaX * alphaY,
              w1 = betaX * alphaY,
              w2 = betaX * betaY,
              w3 = alphaX * betaY;
//        if( betaX < 0 || betaX > 1 || betaY < 0 || betaY > 1 )
//        {
//            printf( "betaX = %f  -  betaY = %f\n", betaX, betaY );
//        }

        // Perform bilinear interpolation
        float yBi = 0.0, uBi = 0.0, vBi = 0.0;
        int yIdx, uIdx, vIdx;
        bool exist = true;
        if( isInside( col0, row0, _width, _height) )
        {
            yIdx = row0 * _width + col0;
            uIdx = (row0 >> 1) * (_width >> 1) + (col0 >> 1) + len;
            vIdx = uIdx + (len >> 2);

            yBi += w0 * (float)d_src[yIdx];
            uBi += w0 * (float)d_src[uIdx];
            vBi += w0 * (float)d_src[vIdx];
        }
        else
            exist = false;

        if( isInside( col1, row1, _width, _height) )
        {
            yIdx = row1 * _width + col1;
            uIdx = (row1 >> 1) * (_width >> 1) + (col1 >> 1) + len;
            vIdx = uIdx + (len >> 2);

            yBi += w1 * (float)d_src[yIdx];
            uBi += w1 * (float)d_src[uIdx];
            vBi += w1 * (float)d_src[vIdx];
        }
        else
            exist = false;

        if( isInside( col2, row2, _width, _height) )
        {
            yIdx = row2 * _width + col2;
            uIdx = (row2 >> 1) * (_width >> 1) + (col2 >> 1) + len;
            vIdx = uIdx + (len >> 2);

            yBi += w2 * (float)d_src[yIdx];
            uBi += w2 * (float)d_src[uIdx];
            vBi += w2 * (float)d_src[vIdx];
        }
        else
            exist = false;

        if( isInside( col3, row3, _width, _height) )
        {
            yIdx = row3 * _width + col3;
            uIdx = (row3 >> 1) * (_width >> 1) + (col3 >> 1) + len;
            vIdx = uIdx + (len >> 2);

            yBi += w3 * (float)d_src[yIdx];
            uBi += w3 * (float)d_src[uIdx];
            vBi += w3 * (float)d_src[vIdx];
        }

        else
            exist = false;

        yIdx = row * _width + col;
        if( exist )
            d_dst[yIdx] = (unsigned char)(yBi + 0.5);
        else
            d_dst[yIdx] = 16;
        if( (row & 0x1) && (col & 0x1) )
        {
            uIdx = (row >> 1) * (_width >> 1) + (col >> 1) + len;
            vIdx = uIdx + (len >> 2);
            if( exist )
            {
                d_dst[uIdx] = (unsigned char)(uBi + 0.5);
                d_dst[vIdx] = (unsigned char)(vBi + 0.5);
            }
            else
            {
                d_dst[uIdx] = 128;
                d_dst[vIdx] = 128;
            }
        }
    }
}


/**
 * @brief gpu_warpI420
 */
cudaError_t gpu_invWarpI420( const unsigned char *d_src, unsigned char *d_dst, const float *d_T, const int _width, const int _height )
{
    if( d_src == NULL || d_dst == NULL )
    {
        printf( "!Error: %s:%d: Invalid device pointers\n", __FUNCTION__, __LINE__ );
        return cudaErrorInvalidDevicePointer;
    }

    if( (_width < 0) || (_height < 0) )
    {
        printf( "!Error: %s:%d: Invalid image dimension\n", __FUNCTION__, __LINE__ );
        return cudaErrorInvalidValue;
    }

    int numel = _width * _height;
    cuda_invWarpI420<<<(numel+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>( d_src, d_dst, d_T, _width, _height );
    cudaDeviceSynchronize();

    return cudaSuccess;
}


/*=================================================
 *
 *  I420 Image Affine Warping
 *
 *===============================================*/
__global__ void cuda_invWarpI420_V2( const unsigned char *d_src, unsigned char *d_dst, const float *d_T,
                                     const int _srcW, const int _srcH, const int _dstW, const int _dstH )
{
    int dstLen = _dstW * _dstH;
    int srcLen = _srcW * _srcH;
    int procIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if( procIdx < dstLen )
    {
        // Compute index of pixel in d_src that is corresponding to current pixel in d_dst
        int row  = procIdx / _dstW,
            col  = procIdx % _dstW;
        float x  = (float)col + 0.5,
              y  = (float)row + 0.5;
        float invX = d_T[0] * x + d_T[3] * y + d_T[6],
              invY = d_T[1] * x + d_T[4] * y + d_T[7];

        // Compute index of 4 nearest neighbours of th imaginary source pixel
        int row0 = (invY >= 0) ? (int)invY : ((int)invY - 1),
            col0 = (invX >= 0) ? (int)invX : ((int)invX - 1);

        float dx = invX - (float)col0 - 0.5,
              dy = invY - (float)row0 - 0.5;
        int dR   = (dy < 0) ? -1 : 1,
            dC   = (dx < 0) ? -1 : 1;

        int row1 = row0,
            col1 = col0 + dC,
            row2 = row0 + dR,
            col2 = col0 + dC,
            row3 = row0 + dR,
            col3 = col0;

        // Compute bilinear weights
        float alphaX = 1.0 - fabsf(dx),
              alphaY = 1.0 - fabsf(dy);
        float betaX  = 1.0 - alphaX,
              betaY  = 1.0 - alphaY;
        float w0 = alphaX * alphaY,
              w1 = betaX * alphaY,
              w2 = betaX * betaY,
              w3 = alphaX * betaY;
//        if( betaX < 0 || betaX > 1 || betaY < 0 || betaY > 1 )
//        {
//            printf( "betaX = %f  -  betaY = %f\n", betaX, betaY );
//        }

        // Perform bilinear interpolation
        float yBi = 0.0, uBi = 0.0, vBi = 0.0;
        int yIdx, uIdx, vIdx;
        bool exist = true;
        if( isInside( col0, row0, _srcW, _srcH) )
        {
            yIdx = row0 * _srcW + col0;
            uIdx = (row0 >> 1) * (_srcW >> 1) + (col0 >> 1) + srcLen;
            vIdx = uIdx + (srcLen >> 2);

            yBi += w0 * (float)d_src[yIdx];
            uBi += w0 * (float)d_src[uIdx];
            vBi += w0 * (float)d_src[vIdx];
        }
        else
            exist = false;

        if( isInside( col1, row1, _srcW, _srcH) )
        {
            yIdx = row1 * _srcW + col1;
            uIdx = (row1 >> 1) * (_srcW >> 1) + (col1 >> 1) + srcLen;
            vIdx = uIdx + (srcLen >> 2);

            yBi += w1 * (float)d_src[yIdx];
            uBi += w1 * (float)d_src[uIdx];
            vBi += w1 * (float)d_src[vIdx];
        }
        else
            exist = false;

        if( isInside( col2, row2, _srcW, _srcH) )
        {
            yIdx = row2 * _srcW + col2;
            uIdx = (row2 >> 1) * (_srcW >> 1) + (col2 >> 1) + srcLen;
            vIdx = uIdx + (srcLen >> 2);

            yBi += w2 * (float)d_src[yIdx];
            uBi += w2 * (float)d_src[uIdx];
            vBi += w2 * (float)d_src[vIdx];
        }
        else
            exist = false;

        if( isInside( col3, row3, _srcW, _srcH) )
        {
            yIdx = row3 * _srcW + col3;
            uIdx = (row3 >> 1) * (_srcW >> 1) + (col3 >> 1) + srcLen;
            vIdx = uIdx + (srcLen >> 2);

            yBi += w3 * (float)d_src[yIdx];
            uBi += w3 * (float)d_src[uIdx];
            vBi += w3 * (float)d_src[vIdx];
        }

        else
            exist = false;

        yIdx = row * _dstW + col;
        if( exist )
            d_dst[yIdx] = (unsigned char)(yBi + 0.5);
        else
            d_dst[yIdx] = 16;
        if( (row & 0x1) && (col & 0x1) )
        {
            uIdx = (row >> 1) * (_dstW >> 1) + (col >> 1) + dstLen;
            vIdx = uIdx + (dstLen >> 2);
            if( exist )
            {
                d_dst[uIdx] = (unsigned char)(uBi + 0.5);
                d_dst[vIdx] = (unsigned char)(vBi + 0.5);
            }
            else
            {
                d_dst[uIdx] = 128;
                d_dst[vIdx] = 128;
            }
        }
    }
}


/**
 * @brief gpu_warpI420_V2
 */
cudaError_t gpu_invWarpI420_V2( const unsigned char *d_src, unsigned char *d_dst, const float *d_T,
                               const int _srcW, const int _srcH, const int _dstW, const int _dstH )
{
    if( d_src == NULL || d_dst == NULL )
    {
        printf( "!Error: %s:%d: Invalid device pointers\n", __FUNCTION__, __LINE__ );
        return cudaErrorInvalidDevicePointer;
    }

    if( (_srcW < 0) || (_srcH < 0) )
    {
        printf( "!Error: %s:%d: Invalid image dimension\n", __FUNCTION__, __LINE__ );
        return cudaErrorInvalidValue;
    }

    if( (_dstW < 0) || (_dstH < 0) )
    {
        printf( "!Error: %s:%d: Invalid image dimension\n", __FUNCTION__, __LINE__ );
        return cudaErrorInvalidValue;
    }

    int numel = _dstW * _dstH;
    cuda_invWarpI420_V2<<<(numel+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>( d_src, d_dst, d_T, _srcW, _srcH, _dstW, _dstH );
    cudaDeviceSynchronize();

    return cudaSuccess;
}


/*=================================================
 *
 *  I420 Image Affine Warping
 *
 *===============================================*/
__global__ void cuda_i420Resize( const unsigned char *d_i420Src, unsigned char *d_i420Dst,
                                 const int _srcW, const int _srcH, const int _dstW, const int _dstH )
{
    int srcLen = _srcW * _srcH;
    int dstLen = _dstW * _dstH;
    int procIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if( procIdx < dstLen )
    {
        int dstR = procIdx / _dstW,
            dstC = procIdx % _dstW;

        float srcX = (float)(dstC * _srcW) / (float)_dstW,
              srcY = (float)(dstR * _srcH) / (float)_dstH;

        // Compute index of 4 nearest neighbours of th imaginary source pixel
        int row0 = (srcY >= 0) ? (int)srcY : ((int)srcY - 1),
            col0 = (srcX >= 0) ? (int)srcX : ((int)srcX - 1);

        float dx = srcX - (float)col0 - 0.5,
              dy = srcY - (float)row0 - 0.5;
        int dR   = (dy < 0) ? -1 : 1,
            dC   = (dx < 0) ? -1 : 1;

        int row1 = row0,
            col1 = col0 + dC,
            row2 = row0 + dR,
            col2 = col0 + dC,
            row3 = row0 + dR,
            col3 = col0;

        // Compute bilinear weights
        float alphaX = 1.0 - fabsf(dx),
              alphaY = 1.0 - fabsf(dy);
        float betaX  = 1.0 - alphaX,
              betaY  = 1.0 - alphaY;
        float w0 = alphaX * alphaY,
              w1 = betaX * alphaY,
              w2 = betaX * betaY,
              w3 = alphaX * betaY;

        // Perform bilinear interpolation
        float yBi = 0.0, uBi = 0.0, vBi = 0.0;
        int yIdx, uIdx, vIdx;
        if( isInside( col0, row0, _srcW, _srcH) )
        {
            yIdx = row0 * _srcW + col0;
            uIdx = (row0 >> 1) * (_srcW >> 1) + (col0 >> 1) + srcLen;
            vIdx = uIdx + (srcLen >> 2);

            yBi += w0 * (float)d_i420Src[yIdx];
            uBi += w0 * (float)d_i420Src[uIdx];
            vBi += w0 * (float)d_i420Src[vIdx];
        }

        if( isInside( col1, row1, _srcW, _srcH) )
        {
            yIdx = row1 * _srcW + col1;
            uIdx = (row1 >> 1) * (_srcW >> 1) + (col1 >> 1) + srcLen;
            vIdx = uIdx + (srcLen >> 2);

            yBi += w1 * (float)d_i420Src[yIdx];
            uBi += w1 * (float)d_i420Src[uIdx];
            vBi += w1 * (float)d_i420Src[vIdx];
        }

        if( isInside( col2, row2, _srcW, _srcH) )
        {
            yIdx = row2 * _srcW + col2;
            uIdx = (row2 >> 1) * (_srcW >> 1) + (col2 >> 1) + srcLen;
            vIdx = uIdx + (srcLen >> 2);

            yBi += w2 * (float)d_i420Src[yIdx];
            uBi += w2 * (float)d_i420Src[uIdx];
            vBi += w2 * (float)d_i420Src[vIdx];
        }

        if( isInside( col3, row3, _srcW, _srcH) )
        {
            yIdx = row3 * _srcW + col3;
            uIdx = (row3 >> 1) * (_srcW >> 1) + (col3 >> 1) + srcLen;
            vIdx = uIdx + (srcLen >> 2);

            yBi += w3 * (float)d_i420Src[yIdx];
            uBi += w3 * (float)d_i420Src[uIdx];
            vBi += w3 * (float)d_i420Src[vIdx];
        }

        yIdx = procIdx;
        d_i420Dst[yIdx] = (unsigned char)(yBi + 0.5);
        if( (dstR & 0x1) && (dstC & 0x1) )
        {
            uIdx = (dstR >> 1) * (_dstW >> 1) + (dstC >> 1) + dstLen;
            vIdx = uIdx + (dstLen >> 2);
            d_i420Dst[uIdx] = (unsigned char)(uBi + 0.5);
            d_i420Dst[vIdx] = (unsigned char)(vBi + 0.5);
        }
    }
}


/**
 * @brief gpu_i420Resize
 */
cudaError_t gpu_i420Resize( const unsigned char *d_i420Src, unsigned char *d_i420Dst,
                            const int _srcW, const int _srcH, const int _dstW, const int _dstH )
{
    if( d_i420Src == NULL || d_i420Dst == NULL )
    {
        printf( "!Error: %s:%d: Invalid device pointers\n", __FUNCTION__, __LINE__ );
        return cudaErrorInvalidDevicePointer;
    }

    if( (_srcW < 0) || (_srcH < 0) || (_dstW < 0) || (_dstH < 0) )
    {
        printf( "!Error: %s:%d: Invalid image dimension\n", __FUNCTION__, __LINE__ );
        return cudaErrorInvalidValue;
    }

    int numel = _dstW * _dstH;
    cuda_i420Resize<<<(numel+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>( d_i420Src, d_i420Dst, _srcW, _srcH, _dstW, _dstH );
    cudaDeviceSynchronize();

    return cudaSuccess;
}

/*=================================================
 *
 *  Extract an image patch bounded by a rotated
 *  rectangle from a gray image
 *
 *===============================================*/
__global__ void cuda_getRotatedGrayPatchFromI420( unsigned char *_i420, unsigned char *_patch, const int _iW, const int _iH,
                                                  const int _pW, const int _pH, const float _pCenterX, const float _pCenterY, const float _angle, const float _scale )
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    int len = _pW * _pH;
    if (threadId < len)
    {
        int pR = threadId / _pW,
            pC = threadId % _pW;
        float pX = (float)pC - (float)_pW / 2.0f,
              pY = (float)pR - (float)_pH / 2.0f;

        float cosa  = cosf( _angle ),
              sina  = sinf( _angle );

        float iX    = _scale * (cosa * pX + sina * pY) + _pCenterX,
              iY    = _scale * (-sina * pX + cosa * pY) + _pCenterY;

        int   iR    = (int)(iY + 0.5),
              iC    = (int)(iX + 0.5);

        if( (iR >= 0) && (iR < _iH) && (iC >= 0) && (iC < _iW) )
        {
            int pId = threadId,
                iId = iR * _iW + iC;
            _patch[pId] = _i420[iId];
        }
        else
        {
            int pId = threadId;
            _patch[pId] = 0;
        }
    }
}



/**
 * @brief gpu_getRotatedGrayPatchFromGray
 */
cudaError_t gpu_getRotatedGrayPatchFromI420(unsigned char *_img, unsigned char *_patch, const int _iW, const int _iH,
                                            const cv::RotatedRect &_rotRect, const float _scale )
{
    if( (_img == NULL) || (_patch == NULL) )
    {
        LOG_MSG( "! ERROR: %s:%d: err = %d: NULL device pointer\n", __FUNCTION__, __LINE__, cudaErrorIllegalAddress );
        return cudaErrorIllegalAddress;
    }

    if( (_iW <= 0) || (_iH <= 0) || (_rotRect.size.width <= 0) || (_rotRect.size.height <= 0) )
    {
        LOG_MSG( "! ERROR: %s:%d: err = %d: Invalid image or patch dimension\n", __FUNCTION__, __LINE__, cudaErrorInvalidValue );
        return cudaErrorInvalidValue;
    }

    if( _scale <= 0 )
    {
        LOG_MSG( "! ERROR: %s:%d: err = %d: Invalid scale value\n", __FUNCTION__, __LINE__, cudaErrorInvalidValue );
        return cudaErrorInvalidValue;
    }

    int len = (int)_rotRect.size.width * (int)_rotRect.size.height;
    cuda_getRotatedGrayPatchFromI420<<<(len+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>( _img, _patch, _iW, _iH,
                                                                                                         (int)_rotRect.size.width, (int)_rotRect.size.height,
                                                                                                         _rotRect.center.x, _rotRect.center.y, _rotRect.angle, _scale);
    cudaDeviceSynchronize();

    return cudaSuccess;
}


/*=================================================
 *
 *  Extract an image patch bounded by a rotated
 *  rectangle from a I420 image and convert it
 *  to RGB format
 *
 *===============================================*/
__global__ void cuda_getRotatedRGBPatchFromI420( unsigned char *_i420, unsigned char *_rgbPatch, const int _iW, const int _iH, const int _pW, const int _pH,
                                                 const float _pCenterX, const float _pCenterY, const float _angle, const float _scale )
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    int len = _pW * _pH;
    if (threadId < len)
    {
        int pR = threadId / _pW,
            pC = threadId % _pW;
        float pX = (float)pC - (float)_pW / 2.0f,
              pY = (float)pR - (float)_pH / 2.0f;

        float cosa  = cosf( _angle ),
              sina  = sinf( _angle );

        float iX    = _scale * (cosa * pX + sina * pY) + _pCenterX,
              iY    = _scale * (-sina * pX + cosa * pY) + _pCenterY;

        int   iR    = (int)(iY + 0.5),
              iC    = (int)(iX + 0.5);

        if( (iR >= 0) && (iR < _iH) && (iC >= 0) && (iC < _iW) )
        {
            int pId = threadId * 3,
                yId = iR * _iW + iC,
                uId = (iR >> 1) * (_iW >> 1) + (iC >> 1) + _iW * _iH,
                vId = uId + ((_iW * _iH) >> 2);

            int y   = (int)_i420[yId] - 16,
                u   = (int)_i420[uId] - 128,
                v   = (int)_i420[vId] - 128;

            int r   = (298*y + 409*u + 128) >> 8,
                g   = (298*y - 100*u - 208*v + 128) >> 8,
                b   = (298*y + 516*v + 128) >> 8;

            _rgbPatch[pId++] = (r < 0)? 0 : ((r > 255)? 255 : (unsigned char)r);
            _rgbPatch[pId++] = (g < 0)? 0 : ((g > 255)? 255 : (unsigned char)g);
            _rgbPatch[pId]   = (b < 0)? 0 : ((b > 255)? 255 : (unsigned char)b);
        }
        else
        {
            int pId = threadId * 3;
            _rgbPatch[pId++] = 0;
            _rgbPatch[pId++] = 0;
            _rgbPatch[pId]   = 0;
        }
    }
}


/**
 * @brief gpu_getRotatedRGBPatchFromI420
 */
cudaError_t gpu_getRotatedRGBPatchFromI420(unsigned char *_i420, unsigned char *_rgbPatch, const int _iW, const int _iH,
                                           const cv::RotatedRect &_rotRect, const float _scale )
{
    if( (_i420 == NULL) || (_rgbPatch == NULL) )
    {
        LOG_MSG( "! ERROR: %s:%d: err = %d: NULL device pointer\n", __FUNCTION__, __LINE__, cudaErrorIllegalAddress );
        return cudaErrorIllegalAddress;
    }

    if( (_iW <= 0) || (_iH <= 0) || (_rotRect.size.width <= 0) || (_rotRect.size.height <= 0) )
    {
        LOG_MSG( "! ERROR: %s:%d: err = %d: Invalid image or patch dimension\n", __FUNCTION__, __LINE__, cudaErrorInvalidValue );
        return cudaErrorInvalidValue;
    }

    if( _scale <= 0 )
    {
        LOG_MSG( "! ERROR: %s:%d: err = %d: Invalid scale value\n", __FUNCTION__, __LINE__, cudaErrorInvalidValue );
        return cudaErrorInvalidValue;
    }

    float invScale = 1.0 / _scale;
    int len = (int)_rotRect.size.width * (int)_rotRect.size.height;
    cuda_getRotatedRGBPatchFromI420<<<(len+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>( _i420, _rgbPatch, _iW, _iH,
                                                                                                        (int)_rotRect.size.width, (int)_rotRect.size.height,
                                                                                                        _rotRect.center.x, _rotRect.center.y, _rotRect.angle, invScale);
    cudaDeviceSynchronize();

    return cudaSuccess;
}



/*=================================================
 *
 *  Copy RGB small patch to a bigger RGB image
 *
 *===============================================*/
__global__ void cuda_copyRoi( unsigned char *_fromImg, unsigned char *_toImg, int _depth,
                              int _fx, int _fy, int _fWidth, int _fHeight, int _tWidth, int _tHeight )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int fStride = _depth * _fWidth;
    int len = fStride * _fHeight;
    if( index < len )
    {
        int fR = index / fStride,
            fC = (index % fStride) / _depth,
            fP = (index % fStride) %_depth;

        int tR = fR + _fy,
            tC = fC + _fx;

        if( (tR >= 0) && (tR < _tHeight) && (tC >= 0) && (tC < _tWidth) )
        {
            int tIndex = (tR * _tWidth + tC) * _depth + fP;
            _toImg[tIndex] = _fromImg[index];
        }
    }
}


/**
 * @brief gpu_copyRoi
 */
cudaError_t gpu_copyRoi( unsigned char *_fromImg, unsigned char *_toImg, const int _depth,
                         const int _fx, int _fy, int _fWidth, int _fHeight, int _tWidth, int _tHeight )
{
    if( (_fromImg == NULL) || (_toImg == NULL) )
    {
        return cudaErrorInvalidDevicePointer;
    }

    if( (_depth < 1) || (_depth > 4) )
    {
        return cudaErrorInvalidValue;
    }

    if( (_fWidth <= 0) || (_fHeight <= 0) || (_tWidth <= 0) || (_tHeight <= 0) )
    {
        return cudaErrorInvalidValue;
    }

    int len = _fWidth * _fHeight * _depth;

    cuda_copyRoi<<<(len + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>( _fromImg, _toImg, _depth, _fx, _fy,
                                                                                         _fWidth, _fHeight, _tWidth, _tHeight );
    cudaDeviceSynchronize();

    return cudaSuccess;
}

/*=================================================
 *
 *  Set values of pixels in a specific region of
 *  an I420 image to values of another I420 image
 *  whose size is equal to the size of the above
 *  region
 *
 *===============================================*/
__global__ void cuda_setI420ROIData( const unsigned char *d_i420Src, unsigned char *d_i420Dst, const int _x, const int _y,
                                     const int _srcWidth, const int _srcHeight, const int _dstWidth, const int _dstHeight )
{
    int srcLen = _srcWidth * _srcHeight;
    int dstLen = _dstWidth * _dstHeight;
    int procIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if( procIdx < srcLen )
    {
        int srcRow = procIdx / _srcWidth,
            srcCol = procIdx % _srcWidth;
        int dstRow = srcRow + _y,
            dstCol = srcCol + _x;

        if( isInside( dstCol, dstRow, _dstWidth, _dstHeight) )
        {
            int srcYIdx = srcRow * _srcWidth + srcCol,
                dstYIdx = dstRow * _dstWidth + dstCol;
            d_i420Dst[dstYIdx] = d_i420Src[srcYIdx];

            if( (srcRow & 0x1) && (srcCol &0x1) )
            {
                int srcUIdx = (srcRow >> 1) * (_srcWidth >> 1) + (srcCol >> 1) + srcLen,
                    dstUIdx = (dstRow >> 1) * (_dstWidth >> 1) + (dstCol >> 1) + dstLen;
                d_i420Dst[dstUIdx] = d_i420Src[srcUIdx];

                int srcVIdx = srcUIdx + (srcLen >> 2),
                    dstVIdx = dstUIdx + (dstLen >> 2);
                d_i420Dst[dstVIdx] = d_i420Src[srcVIdx];
            }

        }
    }
}


/**
 * @brief gpu_setI420ROIData
 */
cudaError_t gpu_setI420ROIData( const unsigned char *d_i420Src, unsigned char *d_i420Dst, const int _x, const int _y,
                                const int _srcWidth, const int _srcHeight, const int _dstWidth, const int _dstHeight )
{
    if( d_i420Src == NULL || d_i420Dst == NULL )
    {
        printf( "!Error: %s:%d: Invalid device pointers\n", __FUNCTION__, __LINE__ );
        return cudaErrorInvalidDevicePointer;
    }

    if( (_x & 0x01) || (_y & 0x01) )
    {
        printf( "!Error: %s:%d: Top-left corner x/y coordinate have to be even\n", __FUNCTION__, __LINE__ );
        return cudaErrorInvalidValue;
    }

    if( (_srcWidth < 0) || (_srcHeight < 0) || (_dstWidth < 0) || (_dstHeight < 0) )
    {
        printf( "!Error: %s:%d: Invalid image dimension\n", __FUNCTION__, __LINE__ );
        return cudaErrorInvalidValue;
    }

    int len = _srcWidth * _srcHeight;
    cuda_setI420ROIData<<<(len+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>( d_i420Src, d_i420Dst, _x, _y, _srcWidth,
                                                                                            _srcHeight, _dstWidth, _dstHeight );
    cudaDeviceSynchronize();

    return cudaSuccess;
}


/*=================================================
 *
 *  Compute histogram of an image
 *
 *===============================================*/
__global__ void cuda_histogram( const unsigned char *_img, const float *_hann, float *_hist, const int _width, const int _height, const int _depth, const int _binNum )
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = _width * _depth;
    int len = _height * stride;

    if( id < len )
    {
        int ix = (id % stride) / _depth,
            iy = id / stride;
        int hannId = iy * _width + ix;
        float additive = _hann[hannId];
        int  binId = _img[id] / (255 / _binNum);
        if( binId >= _binNum )
        {
            binId = _binNum - 1;
        }

        int frameId = id % _depth;
        binId = frameId * _binNum + binId;
        atomicAdd( &_hist[binId], additive );
    }
}


__global__ void cuda_histNormalization( float *_arr, float *divider, const int len )
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if( id < len )
    {
        atomicAdd( divider, _arr[id] );
    }

    __syncthreads();

    if( id < len )
    {
        if( *divider )
        {
            _arr[id] /= *divider;
        }
    }
}


/**
 * @brief gpu_histogram
 */
cudaError_t gpu_histogram( const unsigned char *_img, const float *_hann, float *_hist,
                           const int _width, const int _height, const int _depth, const int _binNum )
{
    if( (_img == NULL) || (_hann == NULL) || (_hist == NULL) )
    {
        LOG_MSG( "! ERROR: %s:%d: err = %d: NULL device pointer\n", __FUNCTION__, __LINE__, cudaErrorIllegalAddress );
        return cudaErrorIllegalAddress;
    }

    if( (_width <= 0) || (_height <= 0) || (_depth <= 0) )
    {
        LOG_MSG( "! ERROR: %s:%d: err = %d: Invalid image dimension or format\n", __FUNCTION__, __LINE__, cudaErrorInvalidValue );
        return cudaErrorInvalidValue;
    }

    if( _binNum <= 0 )
    {
        LOG_MSG( "! ERROR: %s:%d: err = %d: Invalid number of histogram bins\n", __FUNCTION__, __LINE__, cudaErrorInvalidValue );
        return cudaErrorInvalidValue;
    }

    // Refresh histogram array
    cudaError_t err = cudaMemset( _hist, 0, _depth * _binNum * sizeof( float ) );
    if( err != cudaSuccess )
    {
        LOG_MSG( "! ERROR: %s:%d: err = %d: Failed to set value for device memory\n", __FUNCTION__, __LINE__, err );
        return err;
    }

    // Compute histogram
    int len = _width * _height * _depth;
    cuda_histogram<<<(len+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>( _img, _hann, _hist, _width, _height, _depth, _binNum );
    cudaDeviceSynchronize();

    float *sum;
    err = cudaMalloc( (void**)&sum, sizeof( float ) );
    if( err != cudaSuccess )
    {
        LOG_MSG( "! ERROR: %s:%d: err = %d: Failed to allocate device memory\n", __FUNCTION__, __LINE__, err );
        return err;
    }
    err = cudaMemset( sum, 0, sizeof( float ) );
    if( err != cudaSuccess )
    {
        cudaFree( sum );
        LOG_MSG( "! ERROR: %s:%d: err = %d: Failed to set value for a memory region\n", __FUNCTION__, __LINE__, err );
        return err;
    }

    // Normalize histogram
    len  = _binNum * _depth;
    cuda_histNormalization<<<(len+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>( _hist, sum, len );
    cudaDeviceSynchronize();
    cudaFree( sum );

    return cudaSuccess;
}


/*=================================================
 *
 * Computes similarity score between two histogram
 * arrays using Bhattacharyya Similarity Coefficient
 *
 *===============================================*/
__global__ void cuda_histSimilarity( const float *_hist1, const float *_hist2, float *_bhatSim, const int _len )
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if( id < _len )
    {
        float q = sqrtf( _hist1[id] * _hist2[id] );
        atomicAdd( _bhatSim, q );
    }

    __syncthreads();

    if( id == 0 )
    {
        *_bhatSim = acosf( *_bhatSim ) * 180.0f / 3.141592654f;
    }
}


/**
 * @brief gpu_histSimilarity
 */
cudaError_t gpu_histSimilarity( const float *_hist1, const float *_hist2, float *_score, const int _len )
{
    if( (_hist1 == NULL) || (_hist2 == NULL) || (_score == NULL) )
    {
        LOG_MSG( "! ERROR: %s:%d: err = %d: NULL pointer(s)\n", __FUNCTION__, __LINE__, cudaErrorIllegalAddress );
        return cudaErrorIllegalAddress;
    }

    float *bhatSim;
    cudaError_t err = cudaMalloc( (void**)&bhatSim, sizeof( float ) );
    if( err != cudaSuccess )
    {
        LOG_MSG( "! ERROR: %s:%d: err = %d: Failed to allocate device memory\n", __FUNCTION__, __LINE__, err );
        return err;
    }
    err = cudaMemset( bhatSim, 0, sizeof( float ) );
    if( err != cudaSuccess )
    {
        cudaFree( bhatSim );
        LOG_MSG( "! ERROR: %s:%d: err = %d: Failed to set value for a memory region\n", __FUNCTION__, __LINE__, err );
        return err;
    }

    cuda_histSimilarity<<<(_len+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>( _hist1, _hist2, bhatSim, _len );
    cudaDeviceSynchronize();

    err = cudaMemcpy( _score, bhatSim, sizeof( float ), cudaMemcpyDeviceToHost );
    usleep(50);
    if( err != cudaSuccess )
        LOG_MSG( "! ERROR: %s:%d: err = %d: Failed to copy data from device to host\n", __FUNCTION__, __LINE__, err );
    cudaFree( bhatSim );

    return err;
}
