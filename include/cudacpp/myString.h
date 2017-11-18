#ifndef __CUDACPP_STRING_H__
#define __CUDACPP_STRING_H__

namespace cudacpp
{
  /**
   * Provides a wrapper to the STL string class. This class is necessary as some
   * objects' definitions are used within files compiled by nvcc.
   *
   * All functions simply call the STL version of the function. For
   * documentation, please consult Google or RTFM.
   */
  class String
  {
    protected:
      /// A pointer to an std::string.
      void * internal;
    public:
      String();
      String(const int n);
      String(const int n, const char fill);
      String(const char * const str);
      String(const String & rhs);
      ~String();

      String & operator = (const String & rhs);
      String   operator + (const String & rhs) const;
      String   operator + (const char * const rhs) const;
      String & operator += (const String & rhs);
      String & operator += (const char * const rhs);

      int size() const;
      String substr(const int start, const int len = -1) const;
      const char * c_str() const;
  };
  inline String operator + (const char * const lhs, const String & rhs)
  {
    return String(lhs) + rhs;
  }
}

#endif
