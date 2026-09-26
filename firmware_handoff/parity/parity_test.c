/*
 * parity_test.c -- asserts kws_extract_mfcc() against the Python vectors.
 *
 *   host:  gcc -O2 -std=c99 -I features -I test_vectors -I parity \
 *              parity/parity_test.c parity/mfcc_reference.c -lm -o parity_test && ./parity_test
 *   yours: link your own kws_extract_mfcc() instead of mfcc_reference.c.
 *
 * FAIL if, for any vector, max |mfcc - expected| >= KWS_PARITY_MFCC_TOL, or any
 * int8 model-input value is off by more than 1. Exit code 0 = all pass.
 *
 * A correct float32 port lands around 1e-4..1e-3. The 0.05 gate is the spec's,
 * but it is loose: a periodic (N instead of N-1) Hann window, for one, only
 * reaches ~0.06 on some clips and slips under it on others. So any vector above
 * KWS_SUSPICIOUS is flagged WARN -- treat WARN as a bug to find, not a pass.
 */
#include <math.h>
#include <stdio.h>

#include "kws_mel_filterbank.h"
#include "kws_parity_vectors.h"

#define KWS_SUSPICIOUS 5e-3f

void kws_extract_mfcc(const int16_t *pcm, float out[KWS_NUM_FRAMES][KWS_NUM_MFCC]);

int main(void) {
    static float got[KWS_PARITY_NUM_FRAMES][KWS_PARITY_NUM_MFCC];
    int failed = 0, warned = 0;

    if (sizeof(KWS_PARITY_CONTRACT_HASH) != sizeof(KWS_FEATURE_CONTRACT_HASH) ||
        __builtin_strcmp(KWS_PARITY_CONTRACT_HASH, KWS_FEATURE_CONTRACT_HASH) != 0) {
        printf("FAIL contract hash: vectors %s vs tables %s\n", KWS_PARITY_CONTRACT_HASH, KWS_FEATURE_CONTRACT_HASH);
        return 1;
    }
    printf("contract hash %s\n\n%-20s %10s %9s %8s %7s\n", KWS_PARITY_CONTRACT_HASH,
           "vector", "max|diff|", "at f,c", "q!=exp", "q>1lsb");

    for (int v = 0; v < KWS_PARITY_NUM_VECTORS; v++) {
        const kws_parity_vector_t *vec = &kws_parity_vectors[v];
        kws_extract_mfcc(vec->input, got);

        float maxd = 0.0f; int mf = 0, mc = 0, qdiff = 0, qbig = 0;
        for (int f = 0; f < KWS_PARITY_NUM_FRAMES; f++) {
            for (int c = 0; c < KWS_PARITY_NUM_MFCC; c++) {
                float d = fabsf(got[f][c] - vec->mfcc[f][c]);
                if (d > maxd) { maxd = d; mf = f; mc = c; }
                float qf = roundf(got[f][c] / KWS_PARITY_IN_SCALE + (float)KWS_PARITY_IN_ZERO_POINT);
                int qi = qf > 127.0f ? 127 : (qf < -128.0f ? -128 : (int)qf);
                int dq = qi - vec->q[f * KWS_PARITY_NUM_MFCC + c];
                if (dq != 0) qdiff++;
                if (dq > 1 || dq < -1) qbig++;
            }
        }
        int ok = (maxd < KWS_PARITY_MFCC_TOL) && qbig == 0 && isfinite(maxd);
        int warn = ok && maxd > KWS_SUSPICIOUS;
        printf("%-20s %10.2e %4d,%-4d %8d %7d  %s\n", vec->name, maxd, mf, mc, qdiff, qbig,
               !ok ? "FAIL" : (warn ? "WARN" : "PASS"));
        failed += !ok;
        warned += warn;
    }
    printf("\n%s (%d/%d vectors)%s\n", failed ? "PARITY FAILED" : "PARITY PASSED",
           KWS_PARITY_NUM_VECTORS - failed, KWS_PARITY_NUM_VECTORS,
           warned ? "  -- WARN: max diff above 5e-3 on some vectors, find the cause" : "");
    return failed != 0;
}
