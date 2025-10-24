/**
 * @file int4_ptq.c
 * @brief Implementation of 4-bit (INT4) weight-only quantization utilities.
 *
 * See @ref ie_int4_quant for API details and packing rules.
 */

#include "ie_quant_int4.h"

#include <math.h>
#include <float.h>
#include <string.h>

/* --------------------------------- helpers -------------------------------- */

/**
 * @brief Clamp a float to the symmetric INT4 range [-7, +7] and convert to int.
 *
 * @param x  Input value (already divided by scale).
 * @return   Rounded and clamped integer in [-7, +7].
 */
static inline int clamp_q4(float x) {
  /* round to nearest even is fine; typical compilers use roundf. */
  float r = roundf(x);
  if (r > 7.0f)  return 7;
  if (r < -7.0f) return -7;
  return (int)r;
}

/**
 * @brief Pack two signed 4-bit integers (in [-7, +7]) into a single byte.
 *
 * Values are first biased by +8 to map them to [1, 15] (0 is unused here),
 * then stored as low/high nibbles: low for even column, high for odd column.
 *
 * @param q0  First value (even column).
 * @param q1  Second value (odd column).
 * @return    Packed byte.
 */
static inline uint8_t pack_q4_pair(int q0, int q1) {
  uint8_t n0 = (uint8_t)(q0 + 8); /* 1..15 (0 reserved for -8, not used) */
  uint8_t n1 = (uint8_t)(q1 + 8);
  return (uint8_t)((n1 << 4) | (n0 & 0x0F));
}

/**
 * @brief Unpack two signed 4-bit integers from a single byte.
 *
 * @param byte  Packed byte.
 * @param out_q0  Output pointer for low-nibble decoded value in [-8, +7].
 * @param out_q1  Output pointer for high-nibble decoded value in [-8, +7].
 */
static inline void unpack_q4_pair(uint8_t byte, int *out_q0, int *out_q1) {
  int n0 = (int)(byte & 0x0F);
  int n1 = (int)((byte >> 4) & 0x0F);
  *out_q0 = n0 - 8;
  *out_q1 = n1 - 8;
}

/**
 * @brief Return true if any of a, b, c is NaN or infinite.
 */
static inline int bad_float3(double a, double b, double c) {
  return (!isfinite(a) || !isfinite(b) || !isfinite(c));
}

/* --------------------------------- public --------------------------------- */

size_t ie_int4_rowbytes(size_t cols) {
  return (cols + 1u) / 2u;
}

int ie_int4_absmax(const float *src, size_t rows, size_t cols, float *out_absmax) {
  if (!src || !out_absmax || rows == 0 || cols == 0) return IE_INT4_STATUS_BADARG;
  size_t n = rows * cols;
  float m = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float v = fabsf(src[i]);
    if (v > m) m = v;
  }
  *out_absmax = m;
  return IE_INT4_STATUS_OK;
}

int ie_int4_absmax_per_row(const float *src, size_t rows, size_t cols, float *out_absmax_rows) {
  if (!src || !out_absmax_rows || rows == 0 || cols == 0) return IE_INT4_STATUS_BADARG;
  for (size_t r = 0; r < rows; ++r) {
    const float *row = src + r * cols;
    float m = 0.0f;
    for (size_t c = 0; c < cols; ++c) {
      float v = fabsf(row[c]);
      if (v > m) m = v;
    }
    out_absmax_rows[r] = m;
  }
  return IE_INT4_STATUS_OK;
}

int ie_int4_quantize_per_tensor(const float *src,
                                size_t rows,
                                size_t cols,
                                uint8_t *dst_packed,
                                float *out_scale) {
  if (!src || !dst_packed || !out_scale || rows == 0 || cols == 0)
    return IE_INT4_STATUS_BADARG;

  float absmax = 0.0f;
  int st = ie_int4_absmax(src, rows, cols, &absmax);
  if (st != IE_INT4_STATUS_OK) return st;

  float scale = (absmax > 0.0f) ? (absmax / 7.0f) : 1.0f;
  if (!isfinite(scale) || scale <= 0.0f) return IE_INT4_STATUS_NUMERIC;
  *out_scale = scale;

  const size_t rb = ie_int4_rowbytes(cols);
  for (size_t r = 0; r < rows; ++r) {
    const float *row = src + r * cols;
    uint8_t *out = dst_packed + r * rb;

    size_t c = 0;
    for (; c + 1 < cols; c += 2) {
      int q0 = clamp_q4(row[c]     / scale);
      int q1 = clamp_q4(row[c + 1] / scale);
      out[c / 2] = pack_q4_pair(q0, q1);
    }
    if (c < cols) {
      /* Odd tail: encode last even column and set high nibble to zero code. */
      int q0 = clamp_q4(row[c] / scale);
      out[c / 2] = pack_q4_pair(q0, 0 /* high nibble = 0 -> encoded as 8 after bias */);
    }
  }

  return IE_INT4_STATUS_OK;
}

int ie_int4_dequantize_per_tensor(const uint8_t *src_packed,
                                  size_t rows,
                                  size_t cols,
                                  float scale,
                                  float *dst) {
  if (!src_packed || !dst || rows == 0 || cols == 0) return IE_INT4_STATUS_BADARG;
  if (!(scale > 0.0f) || !isfinite(scale)) return IE_INT4_STATUS_NUMERIC;

  const size_t rb = ie_int4_rowbytes(cols);
  for (size_t r = 0; r < rows; ++r) {
    const uint8_t *in = src_packed + r * rb;
    float *row = dst + r * cols;

    size_t c = 0;
    for (; c + 1 < cols; c += 2) {
      int q0, q1;
      unpack_q4_pair(in[c / 2], &q0, &q1);
      row[c]     = (float)q0 * scale;
      row[c + 1] = (float)q1 * scale;
    }
    if (c < cols) {
      int q0, q1;
      unpack_q4_pair(in[c / 2], &q0, &q1);
      row[c] = (float)q0 * scale;
      /* ignore q1 for odd tail */
    }
  }

  return IE_INT4_STATUS_OK;
}

int ie_int4_quantize_per_row(const float *src,
                             size_t rows,
                             size_t cols,
                             uint8_t *dst_packed,
                             float *out_scales) {
  if (!src || !dst_packed || !out_scales || rows == 0 || cols == 0)
    return IE_INT4_STATUS_BADARG;

  const size_t rb = ie_int4_rowbytes(cols);

  for (size_t r = 0; r < rows; ++r) {
    const float *row = src + r * cols;
    float absmax = 0.0f;
    /* compute row absmax */
    for (size_t c = 0; c < cols; ++c) {
      float v = fabsf(row[c]);
      if (v > absmax) absmax = v;
    }
    float scale = (absmax > 0.0f) ? (absmax / 7.0f) : 1.0f;
    if (!isfinite(scale) || scale <= 0.0f) return IE_INT4_STATUS_NUMERIC;
    out_scales[r] = scale;

    uint8_t *out = dst_packed + r * rb;

    size_t c = 0;
    for (; c + 1 < cols; c += 2) {
      int q0 = clamp_q4(row[c]     / scale);
      int q1 = clamp_q4(row[c + 1] / scale);
      out[c / 2] = pack_q4_pair(q0, q1);
    }
    if (c < cols) {
      int q0 = clamp_q4(row[c] / scale);
      out[c / 2] = pack_q4_pair(q0, 0);
    }
  }

  return IE_INT4_STATUS_OK;
}

int ie_int4_dequantize_per_row(const uint8_t *src_packed,
                               size_t rows,
                               size_t cols,
                               const float *scales,
                               float *dst) {
  if (!src_packed || !dst || !scales || rows == 0 || cols == 0)
    return IE_INT4_STATUS_BADARG;

  const size_t rb = ie_int4_rowbytes(cols);

  for (size_t r = 0; r < rows; ++r) {
    float scale = scales[r];
    if (!(scale > 0.0f) || !isfinite(scale)) return IE_INT4_STATUS_NUMERIC;

    const uint8_t *in = src_packed + r * rb;
    float *row = dst + r * cols;

    size_t c = 0;
    for (; c + 1 < cols; c += 2) {
      int q0, q1;
      unpack_q4_pair(in[c / 2], &q0, &q1);
      row[c]     = (float)q0 * scale;
      row[c + 1] = (float)q1 * scale;
    }
    if (c < cols) {
      int q0, q1;
      unpack_q4_pair(in[c / 2], &q0, &q1);
      row[c] = (float)q0 * scale;
    }
  }

  return IE_INT4_STATUS_OK;
}

int ie_int4_error_metrics(const float *ref,
                          const float *test,
                          size_t n,
                          double *out_mse,
                          double *out_cosine) {
  if (!ref || !test || !out_mse || !out_cosine || n == 0) return IE_INT4_STATUS_BADARG;

  long double se = 0.0L;
  long double dot = 0.0L;
  long double nr = 0.0L;
  long double nt = 0.0L;

  for (size_t i = 0; i < n; ++i) {
    long double r = ref[i];
    long double t = test[i];
    long double d = r - t;
    se  += d * d;
    dot += r * t;
    nr  += r * r;
    nt  += t * t;
  }

  double mse = (double)(se / (long double)n);
  double denom = sqrt((double)nr) * sqrt((double)nt);
  double cosine = (denom > 0.0) ? (double)(dot / denom) : 0.0;

  if (bad_float3(mse, cosine, denom)) return IE_INT4_STATUS_NUMERIC;

  *out_mse = mse;
  *out_cosine = cosine;
  return IE_INT4_STATUS_OK;
}
