// =============================================================================
// pq_sdc_pipeline.h — PQ-SDC batch search: sequential / OMP / 3-stage pipeline
//
// SDC (Symmetric Distance Computation): the QUERY is quantized too.  With the
// query's code q_m and a precomputed centroid-to-centroid IP table
//     ctab[m][a][b] = IP(centroid[m][a], centroid[m][b])
// the per-query "LUT" is not computed but LOOKED UP: row m = ctab[m][q_m][·].
// The scan is then byte-for-byte identical to the ADC gather scan.  SDC pays
// a second (query-side) quantization error; the exact rerank recovers most of
// the recall, as in every other two-phase method in this project.
//
// Pipeline motivation (course manual §3.1.2 "更多探索：SDC 的流水线并行"):
// a batch of queries flows through three stages —
//     S1 encode  (quantize the query,        1 dedicated pthread)
//     S2 scan    (table lookups over base,   SCAN_T worker pthreads)
//     S3 rerank  (exact float32 top-k,       1 dedicated pthread)
// connected by two bounded queues (pthread_mutex + pthread_cond, classic
// producer-consumer).  Stage S1 for chunk c+1 overlaps S2 for chunk c and S3
// for chunk c−1, hiding the encode and rerank latencies behind the scan.
//
// Three implementations share identical per-query kernels so their outputs
// (and recalls) are the same; only the schedule differs:
//   pq_sdc_batch_sequential — 1 thread, bulk phases, per-phase timing
//   pq_sdc_batch_omp        — flat query-level data parallelism (fairness
//                             baseline: same cores, no stage partitioning)
//   pq_sdc_batch_pipeline   — 1 + SCAN_T + 1 pthreads, bounded queues,
//                             per-stage busy-time accounting
// =============================================================================
#pragma once
#include <arm_neon.h>
#include <pthread.h>
#include <sys/time.h>
#include <vector>
#include <queue>
#include <utility>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include "../Alg_normal/pq_flat_normal.h"
#include "../Alg_parallel/pq_flat_simd.h"   // PQIndexSIMD, pq_build_lut_cc_unroll, neon_gather4

typedef std::priority_queue<std::pair<float, uint32_t>> PQSDCResult;

// ---------------------------------------------------------------------------
// PQSDCTables — offline centroid-to-centroid IP tables + centroid norms
// Memory: M×K×K floats = 8×256×256×4 B = 2 MB (fits L3; each query touches
// only M rows = 8 KB of it).
// ---------------------------------------------------------------------------
struct PQSDCTables
{
    size_t M, K;
    std::vector<float> ctab;    // [(m*K + a)*K + b] = IP(c_ma, c_mb)
    std::vector<float> cnorm;   // [m*K + c] = ||c_mc||²  (for query encoding)
};

inline void pq_sdc_build_tables(const PQIndex& idx, PQSDCTables& t)
{
    const size_t M    = idx.M;
    const size_t K    = idx.K;
    const size_t dsub = idx.dsub;
    t.M = M;
    t.K = K;
    t.ctab.resize(M * K * K);
    t.cnorm.resize(M * K);

    for (size_t m = 0; m < M; ++m)
    {
        const float* c_base = idx.centroids.data() + m * K * dsub;
        for (size_t a = 0; a < K; ++a)
        {
            const float* ca = c_base + a * dsub;
            float n2 = 0.0f;
            for (size_t j = 0; j < dsub; ++j) n2 += ca[j] * ca[j];
            t.cnorm[m * K + a] = n2;

            float* row = t.ctab.data() + (m * K + a) * K;
            for (size_t b = 0; b < K; ++b)
            {
                const float* cb = c_base + b * dsub;
                float ip = 0.0f;
                for (size_t j = 0; j < dsub; ++j) ip += ca[j] * cb[j];
                row[b] = ip;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// pq_sdc_encode_query — quantize the query: per subspace, the L2-nearest
// centroid.  argmin ||q_m − c||² = argmin (||c||² − 2·IP(q_m, c)), so the IP
// table from the existing cc_unroll SIMD LUT builder is reused and the
// encode costs one ADC-LUT build + an M×K scalar argmin.
// ---------------------------------------------------------------------------
inline void pq_sdc_encode_query(const PQIndexSIMD& pq_simd, const PQSDCTables& t,
                                const float* query, uint8_t* qcode,
                                float* ip_scratch /* M*K floats */)
{
    const size_t M = t.M;
    const size_t K = t.K;
    pq_build_lut_cc_unroll(pq_simd, query, ip_scratch);
    for (size_t m = 0; m < M; ++m)
    {
        const float* ip = ip_scratch + m * K;
        const float* n2 = t.cnorm.data() + m * K;
        float   best = std::numeric_limits<float>::max();
        uint8_t bk   = 0;
        for (size_t c = 0; c < K; ++c)
        {
            float score = n2[c] - 2.0f * ip[c];
            if (score < best) { best = score; bk = (uint8_t)c; }
        }
        qcode[m] = bk;
    }
}

// ---------------------------------------------------------------------------
// pq_sdc_assemble_dtable — the SDC "LUT build": M row copies from ctab
// ---------------------------------------------------------------------------
inline void pq_sdc_assemble_dtable(const PQSDCTables& t, const uint8_t* qcode,
                                   float* dtable)
{
    const size_t K = t.K;
    for (size_t m = 0; m < t.M; ++m)
        memcpy(dtable + m * K,
               t.ctab.data() + (m * K + qcode[m]) * K,
               K * sizeof(float));
}

// ---------------------------------------------------------------------------
// pq_sdc_scan_topp — coarse scan → top-p candidate ids.
// M == 8: NEON gather path (identical kernel to pq_rerank_from_dtable_gather's
// Phase 1); otherwise scalar fallback.
// cand_out must hold p entries; *cand_cnt receives the actual count.
// ---------------------------------------------------------------------------
inline void pq_sdc_scan_topp(const PQIndex& index, const float* dtable,
                             size_t p, uint32_t* cand_out, uint32_t* cand_cnt)
{
    const size_t M = index.M;
    const size_t K = index.K;
    const size_t N = index.base_number;
    const uint8_t* codes = index.codes.data();

    std::priority_queue<std::pair<float, uint32_t>> heap;
    // Push helper (max-heap keeps the p smallest distances)
    #define PQ_SDC_PUSH(ip_val, id_val)                                   \
        do {                                                              \
            float _dis = 1.0f - (ip_val);                                 \
            if (heap.size() < p) heap.push({_dis, (uint32_t)(id_val)});   \
            else if (_dis < heap.top().first)                             \
            { heap.push({_dis, (uint32_t)(id_val)}); heap.pop(); }        \
        } while (0)

    if (M == 8)
    {
        const float* lut[8] =
        {
            dtable,       dtable +   K, dtable + 2*K, dtable + 3*K,
            dtable + 4*K, dtable + 5*K, dtable + 6*K, dtable + 7*K
        };
        size_t i = 0;
        for (; i + 4 <= N; i += 4)
        {
            uint8x8_t cv0 = vld1_u8(codes + (i+0)*M);
            uint8x8_t cv1 = vld1_u8(codes + (i+1)*M);
            uint8x8_t cv2 = vld1_u8(codes + (i+2)*M);
            uint8x8_t cv3 = vld1_u8(codes + (i+3)*M);

            float32x4_t lo0 = neon_gather4(lut[0]+vget_lane_u8(cv0,0), lut[1]+vget_lane_u8(cv0,1),
                                           lut[2]+vget_lane_u8(cv0,2), lut[3]+vget_lane_u8(cv0,3));
            float32x4_t hi0 = neon_gather4(lut[4]+vget_lane_u8(cv0,4), lut[5]+vget_lane_u8(cv0,5),
                                           lut[6]+vget_lane_u8(cv0,6), lut[7]+vget_lane_u8(cv0,7));
            float32x4_t lo1 = neon_gather4(lut[0]+vget_lane_u8(cv1,0), lut[1]+vget_lane_u8(cv1,1),
                                           lut[2]+vget_lane_u8(cv1,2), lut[3]+vget_lane_u8(cv1,3));
            float32x4_t hi1 = neon_gather4(lut[4]+vget_lane_u8(cv1,4), lut[5]+vget_lane_u8(cv1,5),
                                           lut[6]+vget_lane_u8(cv1,6), lut[7]+vget_lane_u8(cv1,7));
            float32x4_t lo2 = neon_gather4(lut[0]+vget_lane_u8(cv2,0), lut[1]+vget_lane_u8(cv2,1),
                                           lut[2]+vget_lane_u8(cv2,2), lut[3]+vget_lane_u8(cv2,3));
            float32x4_t hi2 = neon_gather4(lut[4]+vget_lane_u8(cv2,4), lut[5]+vget_lane_u8(cv2,5),
                                           lut[6]+vget_lane_u8(cv2,6), lut[7]+vget_lane_u8(cv2,7));
            float32x4_t lo3 = neon_gather4(lut[0]+vget_lane_u8(cv3,0), lut[1]+vget_lane_u8(cv3,1),
                                           lut[2]+vget_lane_u8(cv3,2), lut[3]+vget_lane_u8(cv3,3));
            float32x4_t hi3 = neon_gather4(lut[4]+vget_lane_u8(cv3,4), lut[5]+vget_lane_u8(cv3,5),
                                           lut[6]+vget_lane_u8(cv3,6), lut[7]+vget_lane_u8(cv3,7));

            PQ_SDC_PUSH(vaddvq_f32(vaddq_f32(lo0, hi0)), i+0);
            PQ_SDC_PUSH(vaddvq_f32(vaddq_f32(lo1, hi1)), i+1);
            PQ_SDC_PUSH(vaddvq_f32(vaddq_f32(lo2, hi2)), i+2);
            PQ_SDC_PUSH(vaddvq_f32(vaddq_f32(lo3, hi3)), i+3);
        }
        for (; i < N; ++i)
        {
            uint8x8_t cv = vld1_u8(codes + i*M);
            float32x4_t lo = neon_gather4(lut[0]+vget_lane_u8(cv,0), lut[1]+vget_lane_u8(cv,1),
                                          lut[2]+vget_lane_u8(cv,2), lut[3]+vget_lane_u8(cv,3));
            float32x4_t hi = neon_gather4(lut[4]+vget_lane_u8(cv,4), lut[5]+vget_lane_u8(cv,5),
                                          lut[6]+vget_lane_u8(cv,6), lut[7]+vget_lane_u8(cv,7));
            PQ_SDC_PUSH(vaddvq_f32(vaddq_f32(lo, hi)), i);
        }
    }
    else
    {
        for (size_t i = 0; i < N; ++i)
        {
            const uint8_t* code = codes + i * M;
            float approx_ip = 0.0f;
            for (size_t m = 0; m < M; ++m) approx_ip += dtable[m * K + code[m]];
            PQ_SDC_PUSH(approx_ip, i);
        }
    }
    #undef PQ_SDC_PUSH

    uint32_t cnt = 0;
    while (!heap.empty())
    {
        cand_out[cnt++] = heap.top().second;
        heap.pop();
    }
    *cand_cnt = cnt;
}

// ---------------------------------------------------------------------------
// pq_sdc_rerank_exact — exact float32 IP rerank of the candidate list
// ---------------------------------------------------------------------------
inline PQSDCResult pq_sdc_rerank_exact(const float* base, const float* query,
                                       size_t d, size_t k,
                                       const uint32_t* cands, uint32_t cnt)
{
    PQSDCResult result;
    for (uint32_t idx = 0; idx < cnt; ++idx)
    {
        uint32_t     id = cands[idx];
        const float* bv = base + (size_t)id * d;
        float ip = 0.0f;
        for (size_t j = 0; j < d; ++j) ip += bv[j] * query[j];
        float dis = 1.0f - ip;
        if (result.size() < k)
        {
            result.push({dis, id});
        }
        else if (dis < result.top().first)
        {
            result.push({dis, id});
            result.pop();
        }
    }
    return result;
}

static inline double pq_sdc_now_us()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

// ===========================================================================
// 1) Sequential baseline — bulk phases, per-phase timing (where does the
//    time go?  Predicts the pipeline's upper bound.)
// ===========================================================================
inline void pq_sdc_batch_sequential(const PQIndex& idx, const PQIndexSIMD& pq_simd,
                                    const PQSDCTables& tabs, const float* base,
                                    const float* queries, int nq, size_t vecdim,
                                    size_t k, size_t p, PQSDCResult* out,
                                    double* enc_us, double* scan_us, double* rr_us)
{
    const size_t M = tabs.M;
    const size_t K = tabs.K;
    std::vector<uint8_t>  qcodes((size_t)nq * M);
    std::vector<uint32_t> cands((size_t)nq * p);
    std::vector<uint32_t> ccnt(nq, 0);
    std::vector<float>    scratch(M * K);
    std::vector<float>    dtable(M * K);

    double t0 = pq_sdc_now_us();
    for (int q = 0; q < nq; ++q)
        pq_sdc_encode_query(pq_simd, tabs, queries + (size_t)q * vecdim,
                            qcodes.data() + (size_t)q * M, scratch.data());
    double t1 = pq_sdc_now_us();

    for (int q = 0; q < nq; ++q)
    {
        pq_sdc_assemble_dtable(tabs, qcodes.data() + (size_t)q * M, dtable.data());
        pq_sdc_scan_topp(idx, dtable.data(), p,
                         cands.data() + (size_t)q * p, &ccnt[q]);
    }
    double t2 = pq_sdc_now_us();

    for (int q = 0; q < nq; ++q)
        out[q] = pq_sdc_rerank_exact(base, queries + (size_t)q * vecdim,
                                     vecdim, k,
                                     cands.data() + (size_t)q * p, ccnt[q]);
    double t3 = pq_sdc_now_us();

    if (enc_us)  *enc_us  = t1 - t0;
    if (scan_us) *scan_us = t2 - t1;
    if (rr_us)   *rr_us   = t3 - t2;
}

// ===========================================================================
// 2) Flat OMP baseline — query-level data parallelism over the SAME kernels.
//    Every thread runs encode+scan+rerank end-to-end for its queries: no
//    stage partitioning, so no stage can starve.  This is the fair reference
//    the pipeline must be judged against (not just the sequential version).
// ===========================================================================
inline void pq_sdc_batch_omp(const PQIndex& idx, const PQIndexSIMD& pq_simd,
                             const PQSDCTables& tabs, const float* base,
                             const float* queries, int nq, size_t vecdim,
                             size_t k, size_t p, PQSDCResult* out, int threads)
{
    const size_t M = tabs.M;
    const size_t K = tabs.K;
    #pragma omp parallel for schedule(dynamic, 8) num_threads(threads)
    for (int q = 0; q < nq; ++q)
    {
        std::vector<float>    scratch(M * K);
        std::vector<float>    dtable(M * K);
        std::vector<uint32_t> cands(p);
        std::vector<uint8_t>  qcode(M);
        uint32_t cnt = 0;

        const float* query = queries + (size_t)q * vecdim;
        pq_sdc_encode_query(pq_simd, tabs, query, qcode.data(), scratch.data());
        pq_sdc_assemble_dtable(tabs, qcode.data(), dtable.data());
        pq_sdc_scan_topp(idx, dtable.data(), p, cands.data(), &cnt);
        out[q] = pq_sdc_rerank_exact(base, query, vecdim, k, cands.data(), cnt);
    }
}

// ===========================================================================
// 3) 3-stage producer-consumer pipeline (pthread + bounded queues)
// ===========================================================================

// Bounded MPMC queue of chunk ids (−1 = shutdown sentinel)
struct SDCBoundedQueue
{
    int             buf[64];
    size_t          cap, head, cnt;
    pthread_mutex_t mtx;
    pthread_cond_t  not_empty, not_full;
};

static inline void sdcq_init(SDCBoundedQueue* q, size_t cap)
{
    q->cap  = (cap > 64) ? 64 : cap;
    q->head = 0;
    q->cnt  = 0;
    pthread_mutex_init(&q->mtx, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

static inline void sdcq_destroy(SDCBoundedQueue* q)
{
    pthread_mutex_destroy(&q->mtx);
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
}

static inline void sdcq_push(SDCBoundedQueue* q, int v)
{
    pthread_mutex_lock(&q->mtx);
    while (q->cnt == q->cap) pthread_cond_wait(&q->not_full, &q->mtx);
    q->buf[(q->head + q->cnt) % q->cap] = v;
    ++q->cnt;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mtx);
}

static inline int sdcq_pop(SDCBoundedQueue* q)
{
    pthread_mutex_lock(&q->mtx);
    while (q->cnt == 0) pthread_cond_wait(&q->not_empty, &q->mtx);
    int v   = q->buf[q->head];
    q->head = (q->head + 1) % q->cap;
    --q->cnt;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mtx);
    return v;
}

#define SDC_MAX_SCAN_THREADS 16

struct SDCPipeCtx
{
    const PQIndex*     idx;
    const PQIndexSIMD* pq_simd;
    const PQSDCTables* tabs;
    const float*       base;
    const float*       queries;
    int    nq, chunk, nchunks, scan_threads;
    size_t vecdim, k, p, M, K;

    uint8_t*     qcodes;    // nq * M
    uint32_t*    cands;     // nq * p
    uint32_t*    ccnt;      // nq
    PQSDCResult* out;

    SDCBoundedQueue q1;     // encoder → scanners
    SDCBoundedQueue q2;     // scanners → reranker

    double busy_encode;
    double busy_rerank;
    double busy_scan[SDC_MAX_SCAN_THREADS];
};

struct SDCScanArg
{
    SDCPipeCtx* ctx;
    int         wid;
};

// Stage 1 — encoder (producer): quantize chunk c's queries, push c to q1
static inline void* sdc_pipe_encoder(void* argp)
{
    SDCPipeCtx* ctx = (SDCPipeCtx*)argp;
    std::vector<float> scratch(ctx->M * ctx->K);
    for (int c = 0; c < ctx->nchunks; ++c)
    {
        int q_beg = c * ctx->chunk;
        int q_end = (q_beg + ctx->chunk < ctx->nq) ? q_beg + ctx->chunk : ctx->nq;
        double t0 = pq_sdc_now_us();
        for (int q = q_beg; q < q_end; ++q)
            pq_sdc_encode_query(*ctx->pq_simd, *ctx->tabs,
                                ctx->queries + (size_t)q * ctx->vecdim,
                                ctx->qcodes + (size_t)q * ctx->M, scratch.data());
        ctx->busy_encode += pq_sdc_now_us() - t0;
        sdcq_push(&ctx->q1, c);
    }
    for (int s = 0; s < ctx->scan_threads; ++s) sdcq_push(&ctx->q1, -1);
    return NULL;
}

// Stage 2 — scan workers: pop chunk, assemble dtable + scan each query
static inline void* sdc_pipe_scanner(void* argp)
{
    SDCScanArg* arg = (SDCScanArg*)argp;
    SDCPipeCtx* ctx = arg->ctx;
    std::vector<float> dtable(ctx->M * ctx->K);
    for (;;)
    {
        int c = sdcq_pop(&ctx->q1);
        if (c < 0) { sdcq_push(&ctx->q2, -1); break; }
        int q_beg = c * ctx->chunk;
        int q_end = (q_beg + ctx->chunk < ctx->nq) ? q_beg + ctx->chunk : ctx->nq;
        double t0 = pq_sdc_now_us();
        for (int q = q_beg; q < q_end; ++q)
        {
            pq_sdc_assemble_dtable(*ctx->tabs, ctx->qcodes + (size_t)q * ctx->M,
                                   dtable.data());
            pq_sdc_scan_topp(*ctx->idx, dtable.data(), ctx->p,
                             ctx->cands + (size_t)q * ctx->p, &ctx->ccnt[q]);
        }
        ctx->busy_scan[arg->wid] += pq_sdc_now_us() - t0;
        sdcq_push(&ctx->q2, c);
    }
    return NULL;
}

// Stage 3 — reranker (consumer): counts real chunks, ignores sentinels
static inline void* sdc_pipe_reranker(void* argp)
{
    SDCPipeCtx* ctx  = (SDCPipeCtx*)argp;
    int         done = 0;
    while (done < ctx->nchunks)
    {
        int c = sdcq_pop(&ctx->q2);
        if (c < 0) continue;
        int q_beg = c * ctx->chunk;
        int q_end = (q_beg + ctx->chunk < ctx->nq) ? q_beg + ctx->chunk : ctx->nq;
        double t0 = pq_sdc_now_us();
        for (int q = q_beg; q < q_end; ++q)
            ctx->out[q] = pq_sdc_rerank_exact(ctx->base,
                                              ctx->queries + (size_t)q * ctx->vecdim,
                                              ctx->vecdim, ctx->k,
                                              ctx->cands + (size_t)q * ctx->p,
                                              ctx->ccnt[q]);
        ctx->busy_rerank += pq_sdc_now_us() - t0;
        ++done;
    }
    return NULL;
}

// Driver — spawns 1 encoder + scan_threads scanners + 1 reranker, joins all.
// busy_* outputs let main print per-stage utilization (busy / wall).
inline void pq_sdc_batch_pipeline(const PQIndex& idx, const PQIndexSIMD& pq_simd,
                                  const PQSDCTables& tabs, const float* base,
                                  const float* queries, int nq, size_t vecdim,
                                  size_t k, size_t p, PQSDCResult* out,
                                  int scan_threads, int chunk,
                                  double* busy_enc, double* busy_scan_total,
                                  double* busy_rr)
{
    if (scan_threads < 1) scan_threads = 1;
    if (scan_threads > SDC_MAX_SCAN_THREADS) scan_threads = SDC_MAX_SCAN_THREADS;

    std::vector<uint8_t>  qcodes((size_t)nq * tabs.M);
    std::vector<uint32_t> cands((size_t)nq * p);
    std::vector<uint32_t> ccnt(nq, 0);

    SDCPipeCtx ctx;
    ctx.idx     = &idx;      ctx.pq_simd = &pq_simd;  ctx.tabs = &tabs;
    ctx.base    = base;      ctx.queries = queries;
    ctx.nq      = nq;        ctx.chunk   = chunk;
    ctx.nchunks = (nq + chunk - 1) / chunk;
    ctx.scan_threads = scan_threads;
    ctx.vecdim  = vecdim;    ctx.k = k;   ctx.p = p;
    ctx.M       = tabs.M;    ctx.K = tabs.K;
    ctx.qcodes  = qcodes.data();
    ctx.cands   = cands.data();
    ctx.ccnt    = ccnt.data();
    ctx.out     = out;
    ctx.busy_encode = 0.0;
    ctx.busy_rerank = 0.0;
    for (int s = 0; s < SDC_MAX_SCAN_THREADS; ++s) ctx.busy_scan[s] = 0.0;
    sdcq_init(&ctx.q1, 16);
    sdcq_init(&ctx.q2, 16);

    pthread_t  enc_th, rr_th;
    pthread_t  scan_th[SDC_MAX_SCAN_THREADS];
    SDCScanArg scan_args[SDC_MAX_SCAN_THREADS];

    pthread_create(&enc_th, NULL, sdc_pipe_encoder, &ctx);
    for (int s = 0; s < scan_threads; ++s)
    {
        scan_args[s].ctx = &ctx;
        scan_args[s].wid = s;
        pthread_create(&scan_th[s], NULL, sdc_pipe_scanner, &scan_args[s]);
    }
    pthread_create(&rr_th, NULL, sdc_pipe_reranker, &ctx);

    pthread_join(enc_th, NULL);
    for (int s = 0; s < scan_threads; ++s) pthread_join(scan_th[s], NULL);
    pthread_join(rr_th, NULL);

    sdcq_destroy(&ctx.q1);
    sdcq_destroy(&ctx.q2);

    if (busy_enc) *busy_enc = ctx.busy_encode;
    if (busy_scan_total)
    {
        double s = 0.0;
        for (int w = 0; w < scan_threads; ++w) s += ctx.busy_scan[w];
        *busy_scan_total = s;
    }
    if (busy_rr) *busy_rr = ctx.busy_rerank;
}
