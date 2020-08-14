#ifndef ROLLBUFFER_H
#define ROLLBUFFER_H

#include <cstdio>
#include <memory>
#include <mutex>
#include <vector>

#include "../Packet/Common_type.h"

using namespace Eye;

template <class T> class RollBuffer {
public:
    explicit RollBuffer() {}
    explicit RollBuffer(size_t _size) : m_maxSize(_size) {
        m_buff.reserve(_size);
    }
    ~RollBuffer() {}

    void reset() {
        std::unique_lock<std::mutex> locker(m_mtx);
        m_buff.shrink_to_fit();
        m_buff.clear();
    }
    bool empty() const { return m_buff.size() == 0; }
    bool full() const { return m_buff.size() == m_maxSize; }

    size_t capacity() const { return m_maxSize; }
    size_t size() const { return m_buff.size(); }

    T& at(const index_type _pos){
        if ((_pos >= 0) && (_pos < m_buff.size())) {
            return m_buff[_pos];
        }else{
            return m_temp;
        }
    }

    void add(const T _item) {
        std::unique_lock<std::mutex> locker(m_mtx);
        if (m_buff.size() < m_maxSize) {
            m_buff.push_back(_item);
        } else if (m_buff.size() == m_maxSize) {
            m_buff.front().release();
            m_buff.erase(m_buff.begin());
            m_buff.push_back(_item);
        }
//        m_temp = _item;
    }
    T& last() {
        std::unique_lock<std::mutex> locker(m_mtx);
        if (m_buff.size() > 0) {
            return m_buff.back();
        }else{
            return m_temp;
        }
    }

    T& getElementById(const index_type &_id) {
        std::unique_lock<std::mutex> locker(m_mtx);
        index_type size = m_buff.size();
        if (size == 0){
            return m_temp;
        }
        for (int i = m_buff.size() - 1; i >= 0; i--) {
            if (m_buff.at(i).getIndex() == _id) {
                return m_buff.at(i);
            } else if ((m_buff.at(i).getIndex() < _id) &&
                       (m_buff.at(i).getIndex() != 0)) {
                break;
            }
        }
        return m_temp;
    }

private:
    std::mutex m_mtx;
    std::vector<T> m_buff;
    T m_temp;
    const size_t m_maxSize;
};

#endif // ROLLBUFFER_H
