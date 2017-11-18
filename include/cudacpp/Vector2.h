#ifndef __CUDACPP_VECTOR2_H__
#define __CUDACPP_VECTOR2_H__

#include "cudacpp/myString.h"
#include <cstdio>

namespace cudacpp
{
  /**
   * Vector2 is a templated class that holds two instances of the templated type.
   * Virtually any class, struct, enum, or primitive may be used, as long as the
   * templated type supports the copy constructor and assignment operator.
   *
   * @param T The type of the data to be held.
   */
  template <typename T>
  class Vector2
  {
    protected:
      /**
       * Returns a printf-compatible format string so that a string
       * representation of this class can be given. The default implementation
       * assumes that the type is a four-byte integer. Implementations are also
       * provided floats, unsigned types, and integers.
       */
      inline String formatString() const;
    public:
      /// The first data item. May be referenced as vectorInstance.x.
      T x;
      /// The second data item. May be referenced as vectorInstance.y.
      T y;

      /**
       * The default constructor, the data elements are not initailized, and are
       * not guaranteed to be zero.
       */
      inline Vector2() { }

      /**
       * A specialized constructor to assign the values tx and ty to their
       * corresponding data elements.
       *
       * @param tx Data element assigned to this->x.
       * @param ty Data element assigned to this->y.
       */
      inline Vector2(const T & tx, const T & ty) : x(tx), y(ty) { }

      /**
       * A specialized copy constructor. Copies the values from rhs, but does no
       * explicit type conversion. This guarantees that warnings are generated
       * upon loss of precision, and that errors are generated upon incompatible
       * types. If T2 == T, then no warnings or errors will be generated, provided
       * T has an available copy constructor.
       *
       * @param T2 The data type held within rhs.
       * @param rhs The data from which values should be copied.
       */
      template <typename T2> inline Vector2(const Vector2<T2> & rhs) : x(rhs.x), y(rhs.y) { }

      /**
       * A specialized assignment operator. Copies the values from rhs, but does
       * no * explicit type conversion. This guarantees that warnings are
       * generated * upon loss of precision, and that errors are generated upon
       * incompatible * types. If T2 == T, then no warnings or errors will be
       * generated, provided * T has an available copy constructor.
       *
       * @param T2 The data type held within rhs.
       * @param rhs The data from which values should be copied.
       * @return Returns an updated reference to *this.
       */
      template <typename T2> inline Vector2<T> & operator = (const Vector2<T2> & rhs)
      {
        x = rhs.x;
        y = rhs.y;
        return *this;
      }

      /**
       * This is a convenience function to allow one to access the vector as an
       * array. No bounds checking is performed.
       *
       * @param index The data item to be returned.
       * @return x if index == 0, y if index == 1, both by reference.
       */
      inline        T & operator [] (const int index)       { return *(&x + index); }

      /**
       * This is a convenience function to allow one to access the vector as an
       * array. No bounds checking is performed.
       *
       * @param index The data item to be returned.
       * @return x if index == 0, y if index == 1, both by reference.
       */
      inline const  T & operator [] (const int index) const { return *(&x + index); }

      /**
       * @return A string representation of this vector.
       */
      inline String toString() const
      {
        char buf[50];
        String fmt = formatString();
        sprintf(buf, ("Vector2(" + fmt + "," + fmt + ")").c_str(), x, y);
        return String(buf);
      }
  };
  template <typename T> inline String Vector2<T> ::formatString() const { return "%d";  }
  template <> inline String Vector2<int>         ::formatString() const { return "%d";  }
  template <> inline String Vector2<unsigned int>::formatString() const { return "%u";  }
  template <> inline String Vector2<float>       ::formatString() const { return "%f";  }
  template <> inline String Vector2<double>      ::formatString() const { return "%lf"; }
}

#endif
