/*
 * IEEE Half-Precision Floating Point Conversions
 * Copyright (c) 2010, Mark Wiebe
 *
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NumPy Developers nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTERS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "bfloat16.h"
#include "numpy/ufuncobject.h"

/*
 * This chooses between 'ties to even' and 'ties away from zero'.
 */
#define HALF_ROUND_TIES_TO_EVEN 1
/*
 * If these are 1, the conversions try to trigger underflow
 * and overflow in the FP system when needed.
 */
#define HALF_GENERATE_OVERFLOW 1
#define HALF_GENERATE_UNDERFLOW 1
#define HALF_GENERATE_INVALID 1

#if !defined(generate_overflow_error)
static double numeric_over_big = 1e300;
static void generate_overflow_error(void) {
        double dummy;
        dummy = numeric_over_big * 1e300;
        if (dummy)
           return;
        else
           numeric_over_big += 0.1;
        return;
        return;
}
#endif

#if !defined(generate_underflow_error)
static double numeric_under_small = 1e-300;
static void generate_underflow_error(void) {
        double dummy;
        dummy = numeric_under_small * 1e-300;
        if (!dummy)
           return;
        else
           numeric_under_small += 1e-300;
        return;
}
#endif

#if !defined(generate_invalid_error)
static double numeric_inv_inf = 1e1000;
static void generate_invalid_error(void) {
        double dummy;
        dummy = numeric_inv_inf - 1e1000;
        if (!dummy)
           return;
        else
           numeric_inv_inf += 1.0;
        return;
}
#endif


/*
 ********************************************************************
 *                   HALF-PRECISION ROUTINES                        *
 ********************************************************************
 */

float
half_to_float(npy_half h)
{
    float ret;
    *((npy_uint32*)&ret) = halfbits_to_floatbits(h);
    return ret;
}

double
half_to_double(npy_half h)
{
    double ret;
    *((npy_uint64*)&ret) = halfbits_to_doublebits(h);
    return ret;
}

npy_half
float_to_half(float f)
{
    return floatbits_to_halfbits(*((npy_uint32*)&f));
}

npy_half
double_to_half(double d)
{
    return doublebits_to_halfbits(*((npy_uint64*)&d));
}

int
half_isnonzero(npy_half h)
{
    return (h&0x7fff) != 0;
}

int
half_isnan(npy_half h)
{
    return ((h&0x7c00u) == 0x7c00u) && ((h&0x03ffu) != 0x0000u);
}

int
half_isinf(npy_half h)
{
    return ((h&0x7c00u) == 0x7c00u) && ((h&0x03ffu) == 0x0000u);
}

int
half_isfinite(npy_half h)
{
    return ((h&0x7c00u) != 0x7c00u);
}

int
half_signbit(npy_half h)
{
    return (h&0x8000u) != 0;
}

npy_half
half_spacing(npy_half h)
{
    npy_half ret;
    npy_uint16 h_exp = h&0x7c00u;
    npy_uint16 h_man = h&0x03ffu;
    if (h_exp == 0x7c00u || h == 0x7bffu) {
#if HALF_GENERATE_INVALID
        generate_invalid_error();
#endif
        ret = HALF_NAN;
    } else if ((h&0x8000u) && h_man == 0) { /* Negative boundary case */
        if (h_exp > 0x2c00u) { /* If result is normalized */
            ret = h_exp - 0x2c00u;
        } else if(h_exp > 0x0400u) { /* The result is denormalized, but not the smallest */
            ret = 1 << ((h_exp >> 10) - 2);
        } else {
            ret = 0x0001u; /* Smallest denormalized half */
        }
    } else if (h_exp > 0x2800u) { /* If result is still normalized */
        ret = h_exp - 0x2800u;
    } else if (h_exp > 0x0400u) { /* The result is denormalized, but not the smallest */
        ret = 1 << ((h_exp >> 10) - 1);
    } else {
        ret = 0x0001u;
    }

    return ret;
}

npy_half
half_copysign(npy_half x, npy_half y)
{
    return (x&0x7fffu) | (y&0x8000u);
}

npy_half
half_nextafter(npy_half x, npy_half y)
{
    npy_half ret;

    if (!half_isfinite(x) || half_isnan(y)) {
#if HALF_GENERATE_INVALID
        generate_invalid_error();
#endif
        ret = HALF_NAN;
    } else if (half_eq_nonan(x, y)) {
        ret = x;
    } else if (!half_isnonzero(x)) {
        ret = (y&0x8000u) + 1; /* Smallest denormalized half */
    } else if (!(x&0x8000u)) { /* x > 0 */
        if ((npy_int16)x > (npy_int16)y) { /* x > y */
            ret = x-1;
        } else {
            ret = x+1;
        }
    } else {
        if (!(y&0x8000u) || (x&0x7fffu) > (y&0x7fffu)) { /* x < y */
            ret = x-1;
        } else {
            ret = x+1;
        }
    }
#ifdef HALF_GENERATE_OVERFLOW
    if (half_isinf(ret)) {
        generate_overflow_error();
    }
#endif

    return ret;
}
 
int
half_eq_nonan(npy_half h1, npy_half h2)
{
    return (h1 == h2 || ((h1 | h2) & 0x7fff) == 0);
}

int
half_eq(npy_half h1, npy_half h2)
{
    /*
     * The equality cases are as follows:
     *   - If either value is NaN, never equal.
     *   - If the values are equal, equal.
     *   - If the values are both signed zeros, equal.
     */
    return (!half_isnan(h1) && !half_isnan(h2)) &&
           (h1 == h2 || ((h1 | h2) & 0x7fff) == 0);
}

int
half_ne(npy_half h1, npy_half h2)
{
    return !half_eq(h1, h2);
}

int
half_lt_nonan(npy_half h1, npy_half h2)
{
    if (h1&0x8000u) {
        if (h2&0x8000u) {
            return (h1&0x7fffu) > (h2&0x7fffu);
        } else {
            /* Signed zeros are equal, have to check for it */
            return (h1 != 0x8000u) || (h2 != 0x0000u);
        }
    } else {
        if (h2&0x8000u) {
            return 0;
        } else {
            return (h1&0x7fffu) < (h2&0x7fffu);
        }
    }
}

int
half_lt(npy_half h1, npy_half h2)
{
    return (!half_isnan(h1) && !half_isnan(h2)) && half_lt_nonan(h1, h2);
}

int
half_gt(npy_half h1, npy_half h2)
{
    return half_lt(h2, h1);
}

int
half_le_nonan(npy_half h1, npy_half h2)
{
    if (h1&0x8000u) {
        if (h2&0x8000u) {
            return (h1&0x7fffu) >= (h2&0x7fffu);
        } else {
            return 1;
        }
    } else {
        if (h2&0x8000u) {
            /* Signed zeros are equal, have to check for it */
            return (h1 == 0x0000u) && (h2 == 0x8000u);
        } else {
            return (h1&0x7fffu) <= (h2&0x7fffu);
        }
    }
}

int
half_le(npy_half h1, npy_half h2)
{
    return (!half_isnan(h1) && !half_isnan(h2)) && half_le_nonan(h1, h2);
}

int
half_ge(npy_half h1, npy_half h2)
{
    return half_le(h2, h1);
}



/*
 ********************************************************************
 *                     BIT-LEVEL CONVERSIONS                        *
 ********************************************************************
 */

/*TODO
 * Should these routines query the CPU float rounding flags?
 * The routine currently does 'ties to even', or 'ties away
 * from zero', depending on a #define above.
 */

npy_uint16
floatbits_to_halfbits(npy_uint32 f)
{
  npy_uint32 lsb = (f >> 16) & 1;
  npy_uint32 rounding_bias = 0x7fff + lsb;
  npy_uint32 output = (f + rounding_bias) >> 16;
  return (npy_uint16)output;
}

npy_uint16
doublebits_to_halfbits(npy_uint64 d)
{
  float f = (float)*(double*)&d;
  return floatbits_to_halfbits(*(npy_uint32*)&f);
}

npy_uint32
halfbits_to_floatbits(npy_uint16 h)
{
    return ((npy_uint32)h) << 16;
}

npy_uint64
halfbits_to_doublebits(npy_uint16 h)
{
    npy_uint32 fbits = halfbits_to_floatbits(h);
    float f = *(float*)&fbits;
    double d = (double)f;
    return *(npy_uint64*)&d;
}
 
