#ifndef MATRIX_H
#define MATRIX_H

#include "Common_type.h"
#include "Vector.h"
#include "utils.h"


namespace math
{
	template<typename T>
	class Matrix_
	{
	public:
		//! Default constructor. Creates an empty matrix.
		inline Matrix_();

		//! Creates an \c m by \c n matrix of default instances of \c T.
		inline Matrix_(V_INT_16 m, V_INT_16 n);

		//! Creates an \c m by \c n matrix of copies of \c a.
		inline Matrix_(V_INT_16 m, V_INT_16 n, const T a);

		//! Creates an \c m by \c n matrix from an array of instances of \c T.
		inline Matrix_(V_INT_16 m, V_INT_16 n, const T* v);

		//! Copy constructor. Performs a deep copy.
		inline Matrix_(const Matrix_<T>& M);

		//! Destructor.
        inline ~Matrix_();

		//@}

		//! \name Member access functions.

		inline T& get(V_INT_16 i, V_INT_16 j);
		//@{

		//! Returns the element <tt>(i,j)</tt>.
		inline T& operator()(V_INT_16 i, V_INT_16 j);

		//! Returns the element <tt>(i,j)</tt>, \c const version.
		inline const T& operator()(V_INT_16 i, V_INT_16 j) const;


		//! Returns the row <tt>(i)</tt>.

		inline Vector_<T> row(V_INT_16 i) const;

		//! Returns the cols <tt>(i)</tt>.

		inline Vector_<T> col(V_INT_16 i)const;

		//! Returns Transpose<tt>(i)</tt>.

		inline Matrix_<T> trans();

		//! Convolution of matrix

		inline Matrix_<T> conv(Matrix_<T> & kernel);

		//! LPFilter

		inline Matrix_<T>& LPF(V_INT_16 i, V_INT_16 j);

		//! HPFilter

		inline Matrix_<T>& HPF(V_INT_16 i, V_INT_16 j);

		//! Resizes the matrix. Resulting matrix contents are undefined.
		inline void resize(V_INT_16 m, V_INT_16 n);

		//! Assigns a copy of \c a to all elements of the matrix.
		inline Matrix_<T>& operator=(const T& a);

		//! Copy assignment operator. Performs a deep copy.
		inline Matrix_<T>& operator=(const Matrix_& M);

		//! Copies a C-style array of instances of \c T in an \c m by \c n matrix.
		inline void assign(V_INT_16 _row_, V_INT_16 _col_, const T* v);

		//! Constant-time swap function between two matrices.
		inline void swap(Matrix_<T>& M);

		//! Constan_time

		inline T* data();

		//! assign cols and rows by vector

		inline void assigncol(V_INT_16 col, Vector_<T> &v);
		inline void assignrow(V_INT_16 row, Vector_<T> &v);

		//! multi two matrix

		Matrix_<T> operator*(const Matrix_& M);

		//! multi with vector
		Vector_<T> operator*(const Vector_<T> &v);

		//!multi with const
		Matrix_<T> operator*(const T &v);

		//! REturns the matrix by div real value
		inline Matrix_<T> operator/(const T &div);

		//! REturns the matrix by div real value
		inline Matrix_<T> operator/(Matrix_<T>& M);

		//! REturns the matrix sum
		inline Matrix_<T> operator+(const Matrix_& M);

		//! REturns the matrix sub
		inline Matrix_<T> operator-(const Matrix_& M);

		//! difference matrix
		inline bool operator!=(const Matrix_&M);

		//! equal matrix
		inline bool operator==(const Matrix_&M);

		// Invert of matrix
		inline Matrix_<T> inv();

		//Convert to

		template <typename T2>
		inline void convertTo(Matrix_<T2> &M2);

		//! \submatrix
		//@{

		static  Matrix_<T> Ones(V_INT_16 size);
		static Matrix_<T> Eyes(V_INT_16 size);

		Matrix_<T> sub(V_INT_16 i, V_INT_16 j);

		//EyePhoenix: Add to byte

		inline std::vector<byte> toByte();
		inline Matrix_<T>* parse(V_INT_16 _row, V_INT_16 _col, std::vector<byte> _data, index_type _index=0);
		inline Matrix_<T>* parse(V_INT_16 _row, V_INT_16 _col, byte* _data, index_type _index = 0);
		inline V_INT_16 b_size()
		{
			return rows*cols*sizeof(T);
		}
		friend inline V_INT_16 b_size(Matrix_<T> &_mat)
		{
			return _mat.rows*_mat.cols*sizeof(T);
		}
	protected:
		//! Array of pointers to rows of \a Mimpl_.

		//! In fact, \a vimpl_ is such that 
		std::vector<T*> _vimpl;
		std::vector<T> _Mimpl;    //!< Underlying vector implementation.
		T** _data;
		//! Helper function to initialize matrix.
		inline void init(V_INT_16 _row_, V_INT_16 _col_);
		inline Matrix_<T> rrow(V_INT_16 i);
	public:
		V_INT_16 rows;             //!< Number of rows of matrix.
		V_INT_16 cols;             //!< Number of columns of matrix.
	public:
		friend inline std::ostream& operator<<(std::ostream &os, Matrix_<T> &_m)
		{
			V_UINT_16 _rows = _m.rows;
			V_UINT_16 _cols = _m.cols;

			if (_rows == 0 || _cols == 0) return os;
			os << "[";
			for (V_UINT_16 i = 0; i < _rows; i++)
			{
				for (V_UINT_16 j = 0; j < _cols; j++)
				{
					os << _m(i, j);
					os << "  ";
				}
				if (i != _rows - 1) os << "," << endl;
			}
			os << "]" << endl;
			return os;
		}

		friend inline void setIdentity(Matrix_<T> &_mat, T _value)
		{
			Matrix_<T> tmp(_mat.rows, _mat.cols);
			_mat.swap(tmp);
            V_INT_16 min_val = min(_mat.rows, _mat.cols);

			T* ptr;
            for (V_INT_16 i = 0; i < min_val; i++)
			{
				ptr = _mat._vimpl[i];
				ptr += i;
				*ptr = _value;
			}
		}
		friend inline void setGeometry(Matrix_<T> &_mat, T _rad, int axis)
		{
			_mat.resize(3, 3);
			T _sin = sin(_rad);
			T _cos = cos(_rad);
			T* ptr;
			switch (axis)
			{
			case 1:
				ptr = _mat._vimpl[0];
				*ptr = 1; *(ptr + 1) = 0; *(ptr + 2) = 0;
				ptr = _mat._vimpl[1];
				*ptr = 0; *(ptr + 1) = _cos; *(ptr + 2) = -_sin;
				ptr = _mat._vimpl[2];
                *ptr = 0; *(ptr + 1) = _sin; *(ptr + 2) = _cos;
				break;
			case 2:
				ptr = _mat._vimpl[0];
				*ptr = _cos; *(ptr + 1) = 0; *(ptr + 2) = -_sin;
				ptr = _mat._vimpl[1];
				*ptr = 0; *(ptr + 1) = 1; *(ptr + 2) = 0;
				ptr = _mat._vimpl[2];
				*ptr = _sin; *(ptr + 1) = 0; *(ptr + 2) = _cos;
				break;
			case 3:
				ptr = _mat._vimpl[0];
				*ptr = _cos; *(ptr + 1) = -_sin; *(ptr + 2) = 0;
				ptr = _mat._vimpl[1];
				*ptr = _sin; *(ptr + 1) = _cos; *(ptr + 2) = 0;
				ptr = _mat._vimpl[2];
				*ptr = 0; *(ptr + 1) = 0; *(ptr + 2) = 1;
				break;
			default:
				break;
			}

		}

	};

	typedef Matrix_<data_type> Matrix;

	/*
	**************************** Other function ************************************
	*/

	template <typename T>
	T determinant(Matrix_<T> &mat)
	{
        assert(mat.cols == mat.rows && mat.cols >= 1);
		T _det = 0;
        V_INT_16 size = mat.cols;

		Matrix_<T> sub(size - 1, size - 1);
		if (size == 1)
		{
			_det = mat(0, 0);
		}
		else if (size == 2)
		{
			_det = mat(0, 0)*mat(1, 1) - mat(0, 1)*mat(1, 0);
		}
		else {
			for (V_INT_16 i = 0; i < size; i++)
			{
				sub = mat.sub(0, i);
				if (i % 2 == 0)
				{
					_det += mat(0, i)*determinant(sub);
				}
				else{
					_det -= mat(0, i)*determinant(sub);
				}

			}

		}
		return _det;
	}
	/*
	------- Implementation-------------
	*/

	template<typename T>
	inline Matrix_<T>::Matrix_()
	{
		init(0, 0);
	}

	template<typename T>
	inline Matrix_<T>::Matrix_(V_INT_16 m, V_INT_16 n) :_Mimpl(m*n)
	{
		init(m, n);
	}


	template<typename T>
	inline Matrix_<T>::Matrix_(V_INT_16 m, V_INT_16 n, const T a) :_Mimpl(m*n, a)
	{
		init(m, n);
	}

	template<typename T>
	inline Matrix_<T>::Matrix_(V_INT_16 m, V_INT_16 n, const T* v) :_Mimpl(v, v + m*n)
	{
		init(m, n);
	}

	template<typename T>
	inline Matrix_<T>::Matrix_(const Matrix_<T>& M) :_Mimpl(M._Mimpl)
	{
		init(M.rows, M.cols);
	}

	template<typename T>
	inline Matrix_<T>::~Matrix_()
    {
	}

	template<typename T>
	inline T& Matrix_<T>::operator()(V_INT_16 i, V_INT_16 j)
	{
		assert(i < rows&&i >= 0 && j < cols&&j >= 0);
		T* ptr = _vimpl[i];
		ptr += j;
		return *ptr;
	}

	template<typename T>
	inline T& Matrix_<T>::get(V_INT_16 i, V_INT_16 j)
	{
		assert(i < rows&&i >= 0 && j < cols&&j >= 0);
		T* ptr = &_Mimpl[i*rows + j];
		return *ptr;
	}

	template<typename T>
	inline  const T& Matrix_<T>::operator()(V_INT_16 i, V_INT_16 j) const
	{
		assert(i < rows&&i >= 0 && j < cols&&j >= 0);
		T* ptr = _vimpl[i];
		ptr += j;
		return *ptr;
	}
	template<typename T>
	inline void Matrix_<T>::init(V_INT_16 _row_, V_INT_16 _col_)
	{

		if (_row_ != 0 && _col_ != 0) {

			// non-empty matrix
			_vimpl.resize(_row_);
			T* ptr = &_Mimpl[0];
			T** M = &_vimpl[0];
			T** end = M + _row_;

			while (M != end) {
				*M++ = ptr;
				ptr += _col_;
			}

			_data = &_vimpl[0];
			rows = _row_;
			cols = _col_;
		}
		else {
			// empty matrix
			_data = 0;
			rows = 0;
			cols = 0;
		}
	}

	template<typename T>
	inline Matrix_<T>& Matrix_<T>::operator=(const T& a)
	{
		T* ptr = &_Mimpl[0];
		const T* end = ptr + _Mimpl.size();

		while (ptr != end) {
			*ptr++ = a;
		}
		return *this;
	}

	template<typename T>
	inline Matrix_<T>& Matrix_<T>::operator=(const Matrix_& M)
	{
		Matrix_<T> temp(M);
		swap(temp);
		return *this;
	}
	template<typename T>
	inline void Matrix_<T>::swap(Matrix_<T>& M) {
		_vimpl.swap(M._vimpl);
		_Mimpl.swap(M._Mimpl);
		std::swap(_data, M._data);
		std::swap(rows, M.rows);
		std::swap(cols, M.cols);
	}

	template<typename T>
	inline T* Matrix_<T>::data()
	{
		return _vimpl[0];
	}
	template<typename T>
	inline void Matrix_<T>::assign(V_INT_16 _row_, V_INT_16 _col_, const T* v)
	{
		Matrix_<T> temp(_row_, _col_, v);
		swap(temp);
	}

	template<typename T>
	inline void Matrix_<T>::resize(V_INT_16 _row_, V_INT_16 _col_)
	{
		if (rows == _row_ && cols == _col_) {
			return;
		}

		_Mimpl.resize(_row_*_col_);
		init(_row_, _col_);
	}

	template<typename T>
	inline Vector_<T> Matrix_<T>::row(V_INT_16 i)const
	{
		assert(i < rows&&i >= 0);
		T* _v = _vimpl[i];
		Vector_<T> _vrow(cols, _v);
		return _vrow;
	}

	template<typename T>
	inline Vector_<T> Matrix_<T>::col(V_INT_16 i)const
	{
        assert(i < cols&&i >= 0);
        Vector_<T> res(rows);
        for (V_INT_16 j = 0; j < rows; j++)
        {
            res(j) = *(_vimpl[j] + i);
        }
        return res;

//        T* _v = (T*)malloc(rows*sizeof(T));

//		for (V_INT_16 j = 0; j < rows; j++)
//		{
//            *(_v + j) = *(_vimpl[j] + i);
//		}

//        return Vector_<T>(rows, _v);
	}

	template<typename T>
	inline Matrix_<T> Matrix_<T>::trans()
	{
		assert(cols > 0 && rows > 0);
        Matrix_<T> res(cols, rows);
        for (V_INT_16 i = 0; i < cols; i++)
        {
            for (V_INT_16 j = 0; j < rows; j++)
            {
                res(i, j) = *(_vimpl[j] + i);
            }
        }
        return res;

//		T* _data = (T*)malloc(rows*cols*sizeof(T));
//		for (V_INT_16 i = 0; i < cols; i++)
//		{
//			for (V_INT_16 j = 0; j < rows; j++)
//			{
//				*(_data + i*rows + j) = *(_vimpl[j] + i);
//			}
//		}
//		return Matrix_<T>(cols, rows, _data);
	}

	template<typename T>
	void Matrix_<T>::assigncol(V_INT_16 col, Vector_<T> &v)
	{
		assert(col >= 0 && col < cols);
		if (v.size() != rows)
		{
			v.resize(rows);
		}
		T* ptr;
		for (V_INT_16 i = 0; i < rows; i++)
		{
			ptr = _vimpl[i];
			ptr += col;
			*ptr = v(i);
		}
	}
	template<typename T>
	inline void Matrix_<T>::assignrow(V_INT_16 row, Vector_<T> &v)
	{
		assert(row >= 0 && row < rows);
		if (v.size() != cols)
		{
			v.resize(cols);
		}
		T* ptr = _vimpl[row];
		for (int i = 0; i < cols; i++)
		{
			*ptr = v(i);
			ptr++;
		}
	}


	template<typename T>
	Matrix_<T> Matrix_<T>:: operator*(const Matrix_ &M)
	{
		assert(M.rows == cols);
		assert(M.rows != 0 && M.cols != 0 && rows != 0 && cols != 0);
		Matrix_<T> _mat(rows, M.cols);
		Vector_<T> _vrow(cols);
		Vector_<T> _vcol(cols);
		T value = 0;
		for (V_UINT_16 i = 0; i < rows; i++)
		{
			_vrow = (*this).row(i);
			for (V_UINT_16 j = 0; j < M.cols; j++)
			{
				_vcol = M.col(j);
				(_mat)(i, j) = _vrow.dotProduct(_vcol);
			}
		}
		return _mat;
	}

	template<typename T>
	Vector_<T> Matrix_<T>:: operator*(const Vector_<T> &_v)
	{
		assert(_v.size() == cols);
		Vector_<T> _vresult(rows);
		Vector_<T> _vrow(cols);
		for (V_INT_16 i = 0; i < rows; i++)
		{
			_vrow = (*this).row(i);
			_vresult(i) = _vrow.dotProduct(_v);
		}
		return _vresult;
	}

	template<typename T>
	Matrix_<T> Matrix_<T>::operator*(const T &v)
	{
		Matrix_<T> _mat(rows, cols);
		for (V_UINT_16 i = 0; i < rows; i++)
		{
			for (V_UINT_16 j = 0; j < cols; j++)
			{
				_mat(i, j) = (*this)(i, j)*v;
			}
		}
		return _mat;
	}

	template<typename T>
	Matrix_<T> Matrix_<T>::sub(V_INT_16 i, V_INT_16 j)
	{
		assert(i >= 0 && i <= rows&&j >= 0 && j <= cols);
		assert(rows > 0 && cols > 0);
		Matrix_<T> _rrow = (*this).rrow(i);
		Matrix_<T> _rrow_t = _rrow.trans();
		Matrix_<T> _rcol_t = _rrow_t.rrow(j);
		return _rcol_t.trans();
	}

	template<typename T>
	inline Matrix_<T> Matrix_<T>::rrow(V_INT_16 i)
	{
		if (i == rows) return (*this);
		assert(i >= 0 && i <= rows&&rows > 0);
		std::vector<T*> _vimpl_ = (*this)._vimpl;
		_vimpl_.erase(_vimpl_.begin() + i);
		std::vector<T> _Mimpl_;
		V_UINT_16 index = 0;
		while (index < rows - 1)
		{
			for (V_UINT_16 lo = 0; lo < cols; lo++)
			{
				_Mimpl_.push_back(*(_vimpl_[index] + lo));
			}
			index++;
		}
		return Matrix_<T>(rows - 1, cols, &_Mimpl_[0]);
	}

	template<typename T>
	inline Matrix_<T> Matrix_<T>::operator/(const T& div)
	{
		assert(div != 0);
		T div_inv = 1 / div;
		return (*this)*div_inv;
	}
	template<typename T>
	inline Matrix_<T> Matrix_<T>::operator/(Matrix_<T>& M)
	{
		assert(rows == cols);
		assert(M.rows == M.cols&&rows == M.rows);
		return (*this)*M.inv();
	}

	template<typename T>
	inline Matrix_<T> Matrix_<T>::operator+(const Matrix_& M)
	{
		assert(M.rows == rows&&M.cols == cols);

		Matrix_<T> _mat(rows, cols);
		for (V_UINT_16 i = 0; i < _mat.rows; i++)
		{
			for (V_UINT_16 j = 0; j < _mat.cols; j++)
			{
				_mat(i, j) = (*this)(i, j) + M(i, j);
			}
		}
		return _mat;
	}
	template<typename T>
	inline Matrix_<T> Matrix_<T>::operator-(const Matrix_& M)
	{
		assert(M.rows == rows&&M.cols == cols);
		Matrix_<T> _mat(rows, cols);
		for (V_UINT_16 i = 0; i < _mat.rows; i++)
		{
			for (V_UINT_16 j = 0; j < _mat.cols; j++)
			{
				_mat(i, j) = (*this)(i, j) - M(i, j);
			}
		}
		return _mat;
	}

	template<typename T>
	inline bool Matrix_<T>::operator!=(const Matrix_<T>& M)
	{
		T dif = 0;
		for (V_INT_16 i = 0; i < _Mimpl.size(); i++)
		{
			dif = (_Mimpl[i] - M._Mimpl[i])* (_Mimpl[i] - M._Mimpl[i]);
		}
		if (dif == 0) return false;
		return true;
	}

	template<typename T>
	inline bool Matrix_<T>::operator==(const Matrix_&M)
	{
		T dif = 0;
		for (V_INT_16 i = 0; i < _Mimpl.size(); i++)
		{
			dif = (_Mimpl[i] - M._Mimpl[i])* (_Mimpl[i] - M._Mimpl[i]);
		}
		if (dif == 0) return true;
		return false;
	}


	template<typename T>
	inline Matrix_<T> Matrix_<T>::inv()
	{
		assert(rows >= 2 && cols >= 2);
		T _det = determinant(*this);
		if (_det == 0) return Matrix_();

		Matrix_<T> _inv(rows, cols);
		Matrix_<T> _trans(cols, rows);
		_trans = (*this).trans();

		Matrix_<T> sub(rows - 1, cols - 1);
		int sign = 1;

		for (V_UINT_16 i = 0; i < rows; i++)
		{
			for (V_UINT_16 j = 0; j < cols; j++)
			{
				sub = _trans.sub(i, j);
				if ((i + j) % 2 == 0) sign = 1;
				else sign = -1;
				_inv(i, j) = determinant(sub) / _det / sign;
			}
		}
		return _inv;
	}

	template<typename T> template <typename T2>
	inline void Matrix_<T>::convertTo(Matrix_<T2> &M2)
	{
		M2.resize(rows, cols);
		for (V_INT_16 i = 0; i < rows; i++)
		{
			for (V_INT_16 j = 0; j < cols; j++)
			{
				M2(i, j) = static_cast<T2>((*this)(i, j));
			}
		}
	}

	template<typename T>
	Matrix_<T> Matrix_<T>::Ones(V_INT_16 size)
	{
		Matrix_<T> _mat(size, size, (T)1);
		return _mat;
	}

	template<typename T>
	Matrix_<T> Matrix_<T>::Eyes(V_INT_16 size)
	{
		Matrix_<T> _mat(size, size);
		for (V_INT_16 i = 0; i < size; i++)
		{
			_mat(i, i) = 1;
		}
		return _mat;
	}

	template<typename T>
	inline Matrix_<T> Matrix_<T>::conv(Matrix_<T> & kernel)
	{
		assert(kernel.cols == kernel.rows&&kernel.cols >= 2);
		assert(kernel.cols % 2 == 1);//Kernel size only old
		V_INT_16 size = kernel % 2;

		return Matrix_<T>();
	}

	template<typename T>
	Matrix_<T>& Matrix_<T>::LPF(V_INT_16 i, V_INT_16 j)
	{
		if (i >= rows || j >= cols) return (*this);

		if (i <= 0 || j <= 0)
		{
			Matrix_<T> tmp(rows, cols);
			swap(tmp);
			return (*this);
		}

		T* ptr = _vimpl[0];
		T* end = _vimpl[i - 1] + cols;
		T* zeros = (T*)calloc(cols - j, sizeof(T));
		while (ptr != end)
		{
			ptr += j;
			memcpy(ptr, zeros, (cols - j)*sizeof(T));
			ptr += cols - j;
		}
		T* zerosex = (T*)calloc(cols, sizeof(T));

		T* endex = _vimpl[0] + _Mimpl.size();

		while (ptr != endex)
		{
			memcpy(ptr, zerosex, cols*sizeof(T));
			ptr += cols;
		}
		return *this;
	}

	//! HPFilter
	template<typename T>
	inline Matrix_<T>& Matrix_<T>::HPF(V_INT_16 i, V_INT_16 j)
	{
		if (i <= 0 || j <= 0) return (*this);
		if (i >= rows || j >= cols)
		{
			Matrix_<T> tmp(rows, cols);
			swap(tmp);
			return (*this);
		}
		T* ptr = _vimpl[0];
		T* end = _vimpl[i - 1] + cols;
		T* zeros = (T*)calloc(j, sizeof(T));
		while (ptr != end)
		{
			memcpy(ptr, zeros, j*sizeof(T));
			ptr += cols;
		}
		return *this;
	}


	template<typename T>
	inline std::vector<byte> Matrix_<T>::toByte()
	{
		std::vector<byte> _result(0);
		std::vector<byte> b_vector;
		T value;
		for (index_type i = 0; i < rows; i++)
		{
			for (index_type j = 0; j < rows; j++)
			{
				value = _Mimpl[i*rows + j];
				b_vector = Utils::toByte<T>(value);
				_result.insert(_result.end(), b_vector.begin(), b_vector.end());
			}
		}
		return _result;
	}
	template<typename T>
	inline Matrix_<T>* Matrix_<T>::parse(V_INT_16 _row, V_INT_16 _col, std::vector<byte> _data, index_type _index)
	{
		Matrix_<T> tmp(_row, _col);
		this->swap(tmp);
		byte* data = _data.data();
		T * ptr = &this->_Mimpl[0];
		index_type index = 0;

		for (index_type i = 0; i < this->rows; i++)
		{
			for (index_type j = 0; j < this->cols; j++)
			{
				index = i*this->rows + j;
				ptr = &this->_Mimpl[index];
				*ptr = Utils::toValue<T>(data, _index+index*sizeof(T));
			}
		}
		return this;
	}


	template<typename T>
	inline Matrix_<T>* Matrix_<T>::parse(V_INT_16 _row, V_INT_16 _col, byte* _data, index_type _index)
	{
		Matrix_<T> tmp(_row, _col);
		this->swap(tmp);
		T * ptr = &this->_Mimpl[0];
		index_type index = 0;

		for (index_type i = 0; i < this->rows; i++)
		{
			for (index_type j = 0; j < this->cols; j++)
			{
				index = i*this->rows + j;
				ptr = &this->_Mimpl[index];
				*ptr = Utils::toValue<T>(_data, _index+index*sizeof(T));
			}
		}
		return this;
	}

}
#endif
