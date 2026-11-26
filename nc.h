/*
 * nc.h - nanochat d34 on Apple Silicon.
 *
 * pt.c   reads a PyTorch checkpoint (zip + pickle) and writes a .ncq file.
 * nc.c   maps a .ncq file, tokenizes, runs the model on the CPU, samples.
 * gpu.m  runs the same model with Metal kernels from kernels.metal.
 * main.m is the app.
 *
 * Model (nanochat commit 2c4473d, the d34 base):
 *	x = norm(wte[tok])
 *	per layer: x += Wo attn(rope_then_qknorm(Wq n, Wk n), Wv n), n = norm(x)
 *	           x += Wpr relu(Wfc norm(x))^2
 *	logits = 15 tanh(lm norm(x) / 15)
 * All norms are RMSNorm without weights. No biases.
 */
#ifndef NC_H
#define NC_H

#include <stddef.h>
#include <stdint.h>

#define NC_MAGIC	0x3151434eu	/* "NCQ1" */
#define NC_VERSION	1
#define NC_PAGE		16384		/* header size and file alignment */
#define NC_BLK		32		/* quant block */
#define NC_SOFTCAP	15.0f
#define NC_EPS		1.1920929e-7f	/* torch.finfo(float32).eps */
#define NC_MAXCTX	4096		/* attention kernel limit */

enum { NC_Q8 = 1, NC_Q4 = 2, NC_BF16 = 3 };

/* .ncq file: header page, tokenizer, then tensors */
typedef struct {
	uint32_t type, rows, cols, pad;
	uint64_t qoff, soff;		/* quants, bf16 scales (none for BF16) */
} NCTen;

typedef struct {
	uint32_t magic, version;
	uint32_t nlayer, nhead, nkv, dim, ffn;
	uint32_t vocab, ntok, seqlen, qtype, bos;
	uint64_t tokoff, toklen;
	uint32_t nten, pad;
	NCTen t[];			/* wte, lm, then q k v o fc pr per layer */
} NCHdr;

/*
 * BF16: rows*cols bf16 values, no scales.
 * Q8: rows*cols int8, value = q * scale.
 * Q4: rows*cols/2 bytes. Byte j of a block holds value j (low nibble)
 *     and value j+16 (high nibble). value = (nibble - 8) * scale.
 * Q8, Q4: one bf16 scale per 32 values of a row.
 */
static inline uint64_t
nc_qbytes(int type, uint64_t rows, uint64_t cols)
{
	return type == NC_BF16 ? rows * cols * 2 : type == NC_Q8 ? rows * cols :
	    rows * cols / 2;
}

static inline uint64_t
nc_sbytes(int type, uint64_t rows, uint64_t cols)
{
	return type == NC_BF16 ? 0 : rows * (cols / NC_BLK) * 2;
}

static inline const char *
nc_tname(int type)
{
	return type == NC_BF16 ? "BF16" : type == NC_Q8 ? "Q8" : "Q4";
}

typedef struct {
	int type, rows, cols;
	uint64_t qoff, soff;
	const uint8_t *q;
	const uint16_t *s;
} NCMat;

typedef struct {
	int nlayer, nhead, nkv, dim, hdim, ffn;
	int vocab;			/* rows of wte and lm (padded) */
	int ntok;			/* real token count, specials included */
	int seqlen, qtype, bos;
	NCMat wte, lm;
	NCMat *wq, *wk, *wv, *wo, *wfc, *wpr;
	const uint8_t **tb;		/* bytes of token id */
	int *tl;			/* length of token id */
	uint8_t *special;
	int *hash, hmask;		/* bytes -> id, ordinary tokens */
	float *cosv, *sinv;		/* seqlen x hdim/2 */
	uint8_t *map;
	size_t mapsz;
} NCModel;

/* nc.c */
float nc_bf(uint16_t h);
uint16_t nc_tobf(float f);
NCModel *nc_load(const char *path, char *err, int errlen);
void nc_free(NCModel *m);
int nc_encode(NCModel *m, const char *s, size_t n, int *out, int max);
int nc_sample(NCModel *m, float *logits, float temp, int topk, uint64_t *rng);

typedef struct NCCpu NCCpu;
NCCpu *ncc_new(NCModel *m, int nctx);
float *ncc_step(NCCpu *c, int tok, int pos, int logits);
void ncc_free(NCCpu *c);

/* pt.c */
typedef void (*NCProgress)(void *ctx, double frac, const char *msg);
int nc_import(const char *path, const char *out, int qtype,
    NCProgress cb, void *ctx, char *err, int errlen);

#ifdef __OBJC__
#import <Foundation/Foundation.h>

@interface NCGpu : NSObject
- (instancetype)initWithModel:(NCModel *)m ctx:(int)nctx error:(NSString **)err;
/* run n tokens from position pos, return raw logits of the last one */
- (float *)run:(const int *)toks count:(int)n pos:(int)pos;
@end
#endif

#endif
