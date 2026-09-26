/*
 * go_nogo_test.c -- does the firmware actually WAKE? (parity_test.c only proves the
 * front end; a port can pass parity and still never fire.)
 *
 * You provide the whole inference path for one 1 s window:
 *     float kws_keyword_prob(const int16_t *window16000);
 *   = MFCC -> int8 quantise -> model -> dequantise -> softmax -> P(keyword).
 *
 * For each go/no-go vector this checks, using YOUR probabilities:
 *   1. P(keyword) on the peak window is within 0.03 of Python's, and on the right side of THETA
 *      (go: > THETA, no-go: < THETA);
 *   2. the same holds for each of the 5 stream windows (catches a model that only works on the peak);
 *   3. the k-of-window rule over those 5 probabilities (at least K above THETA) fires for go and
 *      not for no-go. Your real detector should be run over the stream too and agree.
 * Exit code 0 = GO. Anything else = do not flash.
 */
#include <math.h>
#include <stdio.h>

#include "kws_parity_vectors.h"

float kws_keyword_prob(const int16_t *window16000);

int main(void) {
    int failed = 0, seen = 0;
    printf("theta %.2f  k %d of %d\n\n%-22s %-6s %9s %9s %9s %6s  %s\n", KWS_PARITY_THETA, KWS_PARITY_K,
           KWS_PARITY_WINDOW, "vector", "want", "P(peak)", "P(python)", "max|dP|", "fires", "result");
    for (int v = 0; v < KWS_PARITY_NUM_VECTORS; v++) {
        const kws_parity_vector_t *vec = &kws_parity_vectors[v];
        if (vec->expect_fire < 0) continue;
        seen++;
        float p = kws_keyword_prob(vec->input);
        float maxd = fabsf(p - vec->keyword_prob);
        int over = 0, side_ok = 1;
        for (int j = 0; j < KWS_PARITY_STREAM_WINDOWS; j++) {
            float pj = kws_keyword_prob(vec->stream + j * KWS_PARITY_HOP_SAMPLES);
            float d = fabsf(pj - vec->stream_prob[j]);
            if (!(d <= maxd)) maxd = d;                     /* also catches NaN */
            over += pj > KWS_PARITY_THETA;
        }
        if (vec->expect_fire ? !(p > KWS_PARITY_THETA) : !(p < KWS_PARITY_THETA)) side_ok = 0;
        int fires = over >= KWS_PARITY_K;
        int ok = side_ok && maxd < 0.03f && fires == vec->expect_fire;
        printf("%-22s %-6s %9.4f %9.4f %9.4f %6s  %s\n", vec->name, vec->expect_fire ? "FIRE" : "quiet", p,
               vec->keyword_prob, maxd, fires ? "yes" : "no", ok ? "PASS" : "FAIL");
        failed += !ok;
    }
    if (seen != 2) { printf("FAIL: expected exactly one go and one no-go vector, found %d\n", seen); return 1; }
    printf("\n%s\n", failed ? "NO-GO" : "GO");
    return failed != 0;
}
