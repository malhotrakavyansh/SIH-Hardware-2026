/*
 * go_nogo_stub.c -- HOST-ONLY. NOT A MODEL. It answers kws_keyword_prob() by looking the window up in
 * the vectors' own streams and returning Python's stored probability. Its only purpose is to prove
 * go_nogo_test.c itself compiles and reports GO when handed correct answers. Never link it into
 * firmware; link your own kws_keyword_prob() instead.
 */
#include <string.h>

#include "kws_parity_vectors.h"

float kws_keyword_prob(const int16_t *window16000) {
    for (int v = 0; v < KWS_PARITY_NUM_VECTORS; v++) {
        const kws_parity_vector_t *vec = &kws_parity_vectors[v];
        if (vec->expect_fire < 0) continue;
        for (int j = 0; j < KWS_PARITY_STREAM_WINDOWS; j++)
            if (memcmp(window16000, vec->stream + j * KWS_PARITY_HOP_SAMPLES, KWS_PARITY_CLIP_SAMPLES * sizeof(int16_t)) == 0)
                return vec->stream_prob[j];
    }
    return -1.0f;
}
