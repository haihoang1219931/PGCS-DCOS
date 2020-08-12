#ifndef ROLL_BUFFER_H
#define ROLL_BUFFER_H

#include "../Packet/Common_type.h"
#include <condition_variable>
#include <mutex>

using namespace Eye;

template <class T> class RollBuffer
{
    public:
        RollBuffer() {}
        RollBuffer(const index_type _size)
        {
            m_size = _size;
        }

        ~RollBuffer() {}
        void setSize(const index_type _size)
        {
            m_size = _size;
        }
        void add(T _item)
        {
            std::unique_lock<std::mutex> locker(m_mtx);

            if (m_buff.size() < m_size)
            {
                m_buff.push_back(_item);
            }
            else if (m_buff.size() == m_size)
            {
                m_buff.erase(m_buff.begin());
                m_buff.push_back(_item);
            }
            else if (m_buff.size() > m_size)
            {
                printf("ERROR: Add function failed!");
            }
        }

        T first()
        {
            T res;

            if (m_buff.size() > 0)
            {
                res = m_buff.front();
            }

            return res;
        }

        T last()
        {
            std::unique_lock<std::mutex> locker(m_mtx);
            T res;

            if (m_buff.size() > 0)
            {
                res = m_buff.back();
            }

            return res;
        }
        /*  get element by position
         * position range [0, m_size - 1]
         * */
        T at(const index_type _pos) const
        {
            T res;

            if ((_pos >= 0) && (_pos < m_size))
            {
                res = m_buff[_pos];
            }

            return res;
        }

        /* get a subvector from a specificed pos to the end */
        std::vector<T> retrieve(index_type _begin)
        {
            std::vector<T> res;

            if (m_buff.size() > (_begin + 1))
            {
                res = std::vector<T>(m_buff.begin() + _begin, m_buff.end());
            }

            return res;
        }

        length_type size()
        {
            return m_buff.size();
        }
        // pos range [0..m_size-1]
        std::vector<T> retrieveData(index_type _id)
        {
            std::unique_lock<std::mutex> locker(m_mtx);
            std::vector<T> res;
            index_type size = m_buff.size();
            index_type pos = size - 1;
            index_type currentID;

            for (pos; pos >= 0; pos--)
            {
                currentID = m_buff.at(pos).getIndex();

                if (currentID < _id)
                {
                    // TODO: retrieve view motion from this pos to the end of roll buffer
                    res = std::vector<T>(m_buff.begin() + pos + 1, m_buff.end());
                    break;
                }
            }

            return res;
        }

        T getElementById(index_type _id)
        {
            std::unique_lock<std::mutex> locker(m_mtx);
            T res;
            index_type size = m_buff.size();

            if (size == 0)
            {
                return res;
            }

            for (int i = m_buff.size() - 1; i >= 0; i--)
            {
                if (m_buff.at(i).getIndex() == _id)
                {
                    res = m_buff.at(i);
                    break;
                }
                else if ((m_buff.at(i).getIndex() < _id) &&
                         (m_buff.at(i).getIndex() != 0))
                {
                    break;
                }
            }

            return res;
        }

    protected:
        std::vector<T> m_buff;
        length_type m_size;
        std::mutex m_mtx;
        std::condition_variable m_cv;
        bool m_flag;
};

#endif // ROLL_BUFFER_H
