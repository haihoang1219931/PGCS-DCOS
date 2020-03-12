#ifndef FIXEDMEMORY_H
#define FIXEDMEMORY_H

#ifdef USE_VIDEO_GPU
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#endif
#include <mutex>
#include <memory>

#define MAX_NUM_PINNED_IMAGE 60

class FixedMemory
{
    public:
        explicit FixedMemory(size_t _maxSize, size_t _imgSize) : m_maxSize(_maxSize), m_imgSize(_imgSize)
        {
            m_head = 0;
#ifdef USE_VIDEO_GPU
            for (int i = 0; i < m_maxSize; i++) {
                //                cudaMalloc((void **) & (m_d_buffer[i]), m_imgSize);
                //                m_h_buffer[i] = (unsigned char *)malloc(m_imgSize);
                cudaHostAlloc((void **) & (m_h_buffer[i]), m_imgSize, cudaHostAllocMapped);
                cudaHostGetDevicePointer((void **)&m_d_buffer[i], (void *)m_h_buffer[i], 0);

            }
#endif
#ifdef USE_VIDEO_CPU
            for (int i = 0; i < m_maxSize; i++) {
                m_h_buffer[i] = (unsigned char *)malloc(m_imgSize);
            }
        }
#endif

        unsigned char *getHeadHost()
        {
            std::lock_guard<std::mutex> lock(m_mtx);
            return m_h_buffer[m_head];
        }

        unsigned char *getHeadDevice()
        {
            std::lock_guard<std::mutex> lock(m_mtx);
            return m_d_buffer[m_head];
        }

        void notifyAddOne()
        {
            std::lock_guard<std::mutex> lock(m_mtx);
            m_head = (m_head + 1) % m_maxSize;
        }

        size_t getCapacity()
        {
            return m_maxSize;
        }

    private:
        std::mutex m_mtx;
        size_t m_head;
        const size_t m_maxSize;
        const size_t m_imgSize;
        unsigned char *m_h_buffer[MAX_NUM_PINNED_IMAGE];
        unsigned char *m_d_buffer[MAX_NUM_PINNED_IMAGE];
};


#endif // FIXEDMEMORY_H
