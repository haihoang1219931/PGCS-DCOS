#ifndef RollBufferH
#define RollBufferH

#include <cstdio>
#include <memory>
#include <mutex>
#include <vector>
#define MAX_BUFFER 150
typedef int index_type ;
template <class T> class RollBuffer {
public:
    explicit RollBuffer() {}
    explicit RollBuffer(size_t _size) : m_maxSize(_size) {

    }
    ~RollBuffer() {}

    void reset() {
        std::unique_lock<std::mutex> locker(m_mtx);
        m_size = 0;
        m_lastID = -1;
    }
    bool empty() const { return m_size == 0; }
    bool full() const { return m_size == m_maxSize; }

    size_t capacity() const { return m_maxSize; }
    size_t size() const { return m_size; }

    T at(const index_type _pos) const {
        T res;
        if(_pos >= 0 && _pos < m_maxSize){
            res = this->m_buff[_pos].second;
        }
        return res;
    }

    void add(const T _item) {
        m_lastID++;
        if (m_size < m_maxSize) {
            this->m_buff[this->m_size] = std::make_pair(this->m_size,_item);
            this->m_size++;
        }
        else if (m_size == m_maxSize){
            for(int i=0; i< m_size; i++){
                if(this->m_buff[i].first == m_lastID-1){
                    if(i == m_size-1){
                        m_buff[0].first = m_lastID;
                        m_buff[0].second = _item;
                    }else{
                        m_buff[i+1].first = m_lastID;
                        m_buff[i+1].second = _item;
                    }
                    break;
                }
            }
        }

    }
    T last() {
        std::unique_lock<std::mutex> locker(m_mtx);
        T res;
        for(int i=0; i< m_size; i++){
            if(m_buff[i].first == m_lastID){
                res = m_buff[i].second;
            }
        }
        return res;
    }

    T getElementById(const index_type &_id) {
        T res;
        for(int i=0; i< m_size; i++){
            if(m_buff[i].first == _id){
                res = m_buff[i].second;
                break;
            }
        }
        return res;
    }

public:
    std::mutex m_mtx;
    std::pair<int,T> m_buff[MAX_BUFFER];
    int m_maxSize;
    int m_size = 0;
    int m_lastID = -1;
};

#endif // RollBufferH
