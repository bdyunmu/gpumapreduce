#ifndef __UNSIGNED16_H__
#define __UNSIGNED16_H__
#include <string>
#include <exception>
#include <stdarg.h>
using namespace std;
typedef char byte;

class Unsigned16Exception:public exception{
      public:
	    template<typename... Args>
            Unsigned16Exception(const Args &...rest):exception()
            {
		printf(rest...);
		exit(-1);
            }
};

class Unsigned16{
private:
	unsigned long hi8;
	unsigned long lo8;
public:
	Unsigned16();
	Unsigned16(long l);
	Unsigned16(const Unsigned16 &other);
	int hashCode();
	Unsigned16(string s);
	void set(string s);
	void set(long l);
	static int getHexDigit(char ch);
	byte getByte(int b);
	char getHexDigit(int p);
	unsigned long getHigh8();
	unsigned long getLow8();
	void multiply(Unsigned16 b);
	void add(Unsigned16 b);
	void shiftLeft(int bits);
};


#endif
