#ifndef _UNSIGNED_H
#define _UNSIGNED_H

#include <string>
#include <exception>
using namespace std;
typedef char byte;

class numException:public exception{
      public:
            numException():exception()
            {
            }
};

class Unsigned16{
public:
	long hi8;
	long lo8;
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
	long getHigh8();
	long getLow8();
	void multiply(Unsigned16 b);
	void add(Unsigned16 b);
	void shiftLeft(int bits);
};


#endif
