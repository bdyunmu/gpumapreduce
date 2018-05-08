#ifndef _RANDOM16_H
#define _RANDOM16_H

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
 * This file is copied from Hadoop package org.apache.hadoop.examples.terasort.
 */

/**
 * This class implements a 128-bit linear congruential generator.
 * Specifically, if X0 is the most recently issued 128-bit random
 * number (or a seed of 0 if no random number has already been generated,
 * the next number to be generated, X1, is equal to:
 * X1 = (a * X0 + c) mod 2**128
 * where a is 47026247687942121848144207491837523525
 *            or 0x2360ed051fc65da44385df649fccf645
 *   and c is 98910279301475397889117759788405497857
 *            or 0x4a696d47726179524950202020202001
 * The coefficient "a" is suggested by:
 * Pierre L'Ecuyer, "Tables of linear congruential generators of different
 * sizes and good lattice structure", Mathematics of Computation, 68
 * pp. 249 - 260 (1999)
 * http://www.ams.org/mcom/1999-68-225/S0025-5718-99-00996-5/S0025-5718-99-00996-5.pdf
 * The constant "c" meets the simple suggestion by the same reference that
 * it be odd.
 *
 * There is also a facility for quickly advancing the state of the
 * generator by a fixed number of steps - this facilitates parallel
 * generation.
 *
 * This is based on 1.0 of rand16.c from Chris Nyberg
 * <chris.nyberg@ordinal.com>.
 */

#include "Unsigned16.h"

class Random16 {

  /**
   * The "Gen" array contain powers of 2 of the linear congruential generator.
   * The index 0 struct contain the "a" coefficient and "c" constant for the
   * generator.  That is, the generator is:
   *    f(x) = (Gen[0].a * x + Gen[0].c) mod 2**128
   *
   * All structs after the first contain an "a" and "c" that
   * comprise the square of the previous function.
   *
   * f**2(x) = (Gen[1].a * x + Gen[1].c) mod 2**128
   * f**4(x) = (Gen[2].a * x + Gen[2].c) mod 2**128
   * f**8(x) = (Gen[3].a * x + Gen[3].c) mod 2**128
   * ...

   */
public:

  class RandomConstant {
  public:
    const Unsigned16 a;
    const Unsigned16 c;
    RandomConstant(string left, string right); 
  };

  static const RandomConstant* genArray;

  /**
   *  generate the random number that is "advance" steps
   *  from an initial random number of 0.  This is done by
   *  starting with 0, and then advancing the by the
   *  appropriate powers of 2 of the linear congruential
   *  generator.
   */

  static Unsigned16 skipAhead(Unsigned16 advance);

  /**
   * Generate the next 16 byte random number.
   */
  static void nextRand(Unsigned16 rand);

};

#endif
