#ifndef FIXHOSTMEMORY_H
#define FIXHOSTMEMORY_H

#include <mutex>
#include <memory>

#define MAX_NUM_PINNED_IMAGE 60

class FixedHostMemory
{
    public:
        explicit FixedHostMemory(size_t _maxSize, size_t _imgSize) : m_maxSize(_maxSize), m_imgSize(_imgSize)
        {
            m_head = 0;

            for (int i = 0; i < m_maxSize; i++) {
                m_h_buffer[i] = (unsigned char *)malloc(m_imgSize);
            }
        }

        unsigned char *getHeadHost()
        {
            std::lock_guard<std::mutex> lock(m_mtx);
            return m_h_buffer[m_head];
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
};

#endif // FIXHOSTMEMORY_H
