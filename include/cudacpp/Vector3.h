#ifndef __CUDACPP_VECTOR3_H__
#define __CUDACPP_VECTOR3_H__

#include <cudacpp/Vector2.h>
#include <cudacpp/myString.h>

#include <cstring>
#include <cstdio>

namespace cudacpp
{
  /**
   * Vector3 is a templated class that holds three instances of the templated
   * type. Virtually any class, struct, enum, or primitive may be used, as long
   * as the templated type supports the copy constructor and assignment
   * operator.
   *
   * @param T The type of the data to be held.
   */
  template <typename T>
  class Vector3
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
      /// The third data item. May be referenced as vectorInstance.z.
      T z;

      /**
       * The default constructor, the data elements are not initailized, and are
       * not guaranteed to be zero.
       */
      inline Vector3() { }

      /**
       * A specialized constructor to assign the values tx, ty, and tz to their
       * corresponding data elements.
       *
       * @param tx Data element assigned to this->x.
       * @param ty Data element assigned to this->y.
       * @param tz Data element assigned to this->z.
       */
      inline Vector3(const T & tx, const T & ty, const T & tz) : x(tx), y(ty), z(ty) { }

      /**
       * Constructor used to convert a two-element vector of type T to a three
       * element vector of type T. 0 is assigned to this->z.
       *
       * @param rhs The two element vector.
       */
      inline Vector3(const Vector2<T> & rhs) : x(rhs.x), y(rhs.y), z(0) { }

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
      template <typename T2> inline Vector3(const Vector3<T2> & rhs) : x(rhs.x), y(rhs.y), z(rhs.z) { }

      /**
       * Specialized assignment operator used to convert a two-element vector
       * of type T to a three element vector of type T. 0 is assigned to
       * this->z.
       *
       * @param rhs The two element vector.
       */
      inline Vector3<T> & operator = (const Vector2<T> & rhs)
      {
        x = rhs.x;
        y = rhs.y;
        z = static_cast<T>(0);
        return *this;
      }


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
      template <typename T2> inline Vector3<T> & operator = (const Vector3<T2> & rhs)
      {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        return *this;
      }

      /**
       * This is a convenience function to allow one to access the vector as an
       * array. No bounds checking is performed.
       *
       * @param index The data item to be returned.
       * @return x if index == 0, y if index == 1, z if index == 2, all by reference.
       */
      inline        T & operator [] (const int index)       { return *(&x + index); }

      /**
       * This is a convenience function to allow one to access the vector as an
       * array. No bounds checking is performed.
       *
       * @param index The data item to be returned.
       * @return x if index == 0, y if index == 1, z if index == 2, all by reference.
       */
      inline const  T & operator [] (const int index) const { return *(&x + index); }

      /**
       * @return A string representation of this vector.
       */
      inline String toString() const
      {
        char buf[50];
        String fmt = formatString();
        sprintf(buf, ("Vector3(" + fmt + "," + fmt + "," + fmt + ")").c_str(), x, y, z);
        return String(buf);
      }
  };
  template <typename T> inline String Vector3<T> ::formatString() const { return "%d";  }
  template <> inline String Vector3<int>         ::formatString() const { return "%d";  }
  template <> inline String Vector3<unsigned int>::formatString() const { return "%u";  }
  template <> inline String Vector3<float>       ::formatString() const { return "%f";  }
  template <> inline String Vector3<double>      ::formatString() const { return "%lf"; }
}

#endif
