/*
 * mfcc_reference.c -- SLOW, HOST-ONLY reference of features.extract_mfcc() in
 * float32, written the way an MCU port would do it (radix-2 FFT, float
 * accumulation). It exists to (1) prove the vectors and tables in this bundle
 * are consistent from C, and (2) give a known-good implementation to diff your
 * firmware against. Replace it with your own kws_extract_mfcc() on the device.
 *
 * Pipeline (identical to features.py):
 *   pcm/32768 -> pre-emphasis (whole clip) -> 49 frames of 640, hop 320
 *   -> symmetric Hann -> zero-pad to 1024 -> |FFT|^2 (bins 0..512, no 1/N)
 *   -> 40 mel filters -> log(x + 1e-6) -> 10x40 DCT-II matrix
 */
#include <math.h>
#include <stdint.h>

#include "kws_frontend_constants.h"
#include "kws_mel_filterbank.h"

#define PI_F 3.14159265358979323846f

static void fft_radix2(float *re, float *im, int n) {
    for (int i = 1, j = 0; i < n; i++) {           /* bit-reversal */
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (int len = 2; len <= n; len <<= 1) {
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < len / 2; k++) {
                float ang = -2.0f * PI_F * (float)k / (float)len;
                float wr = cosf(ang), wi = sinf(ang);
                int a = i + k, b = i + k + len / 2;
                float xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr;        im[a] += xi;
            }
        }
    }
}

void kws_extract_mfcc(const int16_t *pcm, float out[KWS_NUM_FRAMES][KWS_NUM_MFCC]) {
    static float x[KWS_NUM_FRAMES * KWS_STRIDE_LEN + KWS_FRAME_LEN];
    const int n_samples = (KWS_NUM_FRAMES - 1) * KWS_STRIDE_LEN + KWS_FRAME_LEN;   /* 16000 */

    x[0] = (float)pcm[0] / KWS_INT16_SCALE;
    for (int n = 1; n < n_samples; n++)
        x[n] = (float)pcm[n] / KWS_INT16_SCALE - KWS_PREEMPHASIS * ((float)pcm[n - 1] / KWS_INT16_SCALE);

    for (int f = 0; f < KWS_NUM_FRAMES; f++) {
        float re[KWS_NFFT], im[KWS_NFFT], power[KWS_NUM_FREQ_BINS], logmel[KWS_NUM_MEL];
        for (int i = 0; i < KWS_NFFT; i++) { re[i] = 0.0f; im[i] = 0.0f; }
        for (int i = 0; i < KWS_FRAME_LEN; i++) re[i] = x[f * KWS_STRIDE_LEN + i] * kws_hann_window[i];
        fft_radix2(re, im, KWS_NFFT);
        for (int k = 0; k < KWS_NUM_FREQ_BINS; k++) power[k] = re[k] * re[k] + im[k] * im[k];
        for (int m = 0; m < KWS_NUM_MEL; m++) {
            float e = 0.0f;
            for (int k = 0; k < KWS_NUM_FREQ_BINS; k++) e += kws_mel_filterbank[m][k] * power[k];
            logmel[m] = logf(e + KWS_LOG_MEL_FLOOR);
        }
        for (int c = 0; c < KWS_NUM_MFCC; c++) {
            float s = 0.0f;
            for (int m = 0; m < KWS_NUM_MEL; m++) s += kws_dct_matrix[c][m] * logmel[m];
            out[f][c] = s;
        }
    }
}
