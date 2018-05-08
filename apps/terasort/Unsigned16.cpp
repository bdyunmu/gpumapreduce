/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


/**
 * This file is copied and simplified from Hadoop package org.apache.hadoop.examples.terasort.
 */

/**
 * An unsigned 16 byte integer class that supports addition, multiplication,
 * and left shifts.
 */
#include <string>
#include <exception>

using namespace std;
typedef char byte;


class numException:public exception
{
public:
	numException():exception()
	{
	}
};


class Unsigned16 {

public:

  long hi8;
  long lo8;

  Unsigned16() {
    hi8 = 0;
    lo8 = 0;
  }

  Unsigned16(long l) {
    hi8 = 0;
    lo8 = l;
  }

  Unsigned16(const Unsigned16 &other) {
    hi8 = other.hi8;
    lo8 = other.lo8;
  }

#if 0
@Override
  public boolean equals(Object o) {
    if (o instanceof Unsigned16) {
      Unsigned16 other = (Unsigned16) o;
      return other.hi8 == hi8 && other.lo8 == lo8;
    }
    return false;
  }
#endif

  int hashCode() {
    return (int) lo8;
  }

  /*
   * Parse a hex string
   * @param s the hex string
   */

  Unsigned16(string s) {
    set(s);
  } //Unsigned

  /**
   * Set the number from a hex string
   * @param s the number in hexadecimal
   * @throws NumberFormatException if the number is invalid
   */

  void set(string s) {
    hi8 = 0;
    lo8 = 0;
    long lastDigit = 0xfl << 60;

    for (int i = 0; i < s.length(); ++i) {

      int digit = getHexDigit(s[i]);
      if ((lastDigit & hi8) != 0) {
      //  throw new NumberFormatException(s + " overflowed 16 bytes");
      }
      hi8 <<= 4;
      unsigned long ul_lo8_lastDigit = (unsigned)(lo8 & lastDigit);
      hi8 |= (ul_lo8_lastDigit >> 60);
      lo8 <<= 4;
      lo8 |= digit;

    }

  }

  /**
   * Set the number to a given long.
   * @param l the new value, which is treated as an unsigned number
   */
  void set(long l) {
    lo8 = l;
    hi8 = 0;
  }

  /**
   * Map a hexadecimal character into a digit.
   * @param ch the character
   * @return the digit from 0 to 15
   * @throws NumberFormatException
   */

static int getHexDigit(char ch) {
    if (ch >= '0' && ch <= '9') {
      return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f') {
      return ch - 'a' + 10;
    }
    if (ch >= 'A' && ch <= 'F') {
      return ch - 'A' + 10;
    }
    //throw new NumberFormatException(ch + " is not a valid hex digit");
  }

  /**
   * Return the number as a hex string.
   */
#if 0
  public String toString() {
    if (hi8 == 0) {
      return Long.toHexString(lo8);
    } else {
      StringBuilder result = new StringBuilder();
      result.append(Long.toHexString(hi8));
      String loString = Long.toHexString(lo8);
      for(int i=loString.length(); i < 16; ++i) {
        result.append('0');
      }
      result.append(loString);
      return result.toString();
    }
  }
#endif

  /**
   * Get a given byte from the number.
   * @param b the byte to get with 0 meaning the most significant byte
   * @return the byte or 0 if b is outside of 0..15
   */
byte getByte(int b) {
    if (b >= 0 && b < 16) {
      if (b < 8) {
        return (byte) (hi8 >> (56 - 8*b));
      } else {
        return (byte) (lo8 >> (120 - 8*b));
      }
    }
    return 0;
  }

  /**
   * Get the hexadecimal digit at the given position.
   * @param p the digit position to get with 0 meaning the most significant
   * @return the character or '0' if p is outside of 0..31
   */
  char getHexDigit(int p) {
    byte digit = getByte(p / 2);
    if (p % 2 == 0) {
      unsigned int ub_digit = (unsigned)digit;
      digit = (unsigned)(ub_digit>>4);
    }
    digit &= 0xf;
    if (digit < 10) {
      return (char) ('0' + digit);
    } else {
      return (char) ('A' + digit - 10);
    }
  }

  /**
   * Get the high 8 bytes as a long.
   */
  long getHigh8() {
    return hi8;
  }

  /**
   * Get the low 8 bytes as a long.
   */
  long getLow8() {
    return lo8;
  }

  /**
   * Multiple the current number by a 16 byte unsigned integer. Overflow is not
   * detected and the result is the low 16 bytes of the result. The numbers
   * are divided into 32 and 31 bit chunks so that the product of two chucks
   * fits in the unsigned 63 bits of a long.
   * @param b the other number
   */
  void multiply(Unsigned16 b) {
    // divide the left into 4 32 bit chunks
    long* left = new long[4];
    left[0] = lo8 & 0xffffffffl;
    unsigned long ul_lo8 = (unsigned)lo8;
    left[1] = ul_lo8 >> 32;
    left[2] = hi8 & 0xffffffffl;
    unsigned long ul_hi8 = (unsigned)hi8;
    left[3] = ul_hi8 >> 32;
    // divide the right into 5 31 bit chunks
    long* right = new long[5];
    right[0] = b.lo8 & 0x7fffffffl;
    unsigned long ul_blo8 = (unsigned)b.lo8;
    right[1] = (ul_blo8 >> 31) & 0x7fffffffl;
    right[2] = (ul_blo8 >> 62) + ((b.hi8 & 0x1fffffffl) << 2);
    unsigned long ul_bhi8 = (unsigned)b.hi8;
    right[3] = (ul_bhi8 >> 29) & 0x7fffffffl;
    right[4] = (ul_bhi8 >> 60);
    // clear the cur value
    set(0);
    Unsigned16 *tmp = new Unsigned16();
    for(int l=0; l < 4; ++l) {
      for (int r=0; r < 5; ++r) {
        long prod = left[l] * right[r];
        if (prod != 0) {
          int off = l*32 + r*31;
          tmp->set(prod);
          tmp->shiftLeft(off);
          add(*tmp);
        }
      }
    }
  }

  /**
   * Add the given number into the current number.
   * @param b the other number
   */
  void add(Unsigned16 b) {
    long sumHi;
    long sumLo;
    long  reshibit, hibit0, hibit1;

    sumHi = hi8 + b.hi8;

    hibit0 = (lo8 & 0x8000000000000000L);
    hibit1 = (b.lo8 & 0x8000000000000000L);
    sumLo = lo8 + b.lo8;
    reshibit = (sumLo & 0x8000000000000000L);
    if ((hibit0 & hibit1) != 0 | ((hibit0 ^ hibit1) != 0 && reshibit == 0))
      sumHi++;  /* add carry bit */
    hi8 = sumHi;
    lo8 = sumLo;
  }

  /**
   * Shift the number a given number of bit positions. The number is the low
   * order bits of the result.
   * @param bits the bit positions to shift by
   */
  void shiftLeft(int bits) {
    if (bits != 0) {
      if (bits < 64) {
        hi8 <<= bits;
 	unsigned long ul_lo8 = (unsigned)lo8;
        hi8 |= (ul_lo8 >> (64 - bits));
        lo8 <<= bits;
      } else if (bits < 128) {
        hi8 = lo8 << (bits - 64);
        lo8 = 0;
      } else {
        hi8 = 0;
        lo8 = 0;
      }
    }
  }
};
