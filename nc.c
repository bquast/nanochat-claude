/*
 * nc.c - map a .ncq model, tokenize, run on the CPU, sample.
 *
 * The CPU path is the reference for the Metal path in gpu.m: same
 * weights, same order of operations. It is also a fallback.
 */
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

#include "nc.h"

float
nc_bf(uint16_t h)
{
	uint32_t u = (uint32_t)h << 16;
	float f;

	memcpy(&f, &u, 4);
	return f;
}

uint16_t
nc_tobf(float f)
{
	uint32_t u;

	memcpy(&u, &f, 4);
	u += 0x7fff + ((u >> 16) & 1);		/* round to nearest even */
	return u >> 16;
}

/* ---- load ---- */

static uint32_t
fnv(const uint8_t *p, int n)
{
	uint32_t h = 2166136261u;

	while (n--)
		h = (h ^ *p++) * 16777619u;
	return h;
}

static int
setmat(NCModel *m, NCMat *w, const NCTen *t)
{
	uint64_t qb, sb;

	if ((t->type != NC_Q8 && t->type != NC_Q4 && t->type != NC_BF16) ||
	    t->cols % NC_BLK)
		return -1;
	qb = nc_qbytes(t->type, t->rows, t->cols);
	sb = nc_sbytes(t->type, t->rows, t->cols);
	if (t->qoff % 256 || t->soff % 256 || t->qoff + qb > m->mapsz ||
	    t->soff + sb > m->mapsz)
		return -1;
	w->type = t->type;
	w->rows = t->rows;
	w->cols = t->cols;
	w->qoff = t->qoff;
	w->soff = t->soff;
	w->q = m->map + t->qoff;
	w->s = sb ? (const uint16_t *)(m->map + t->soff) : NULL;
	return 0;
}

NCModel *
nc_load(const char *path, char *err, int errlen)
{
	NCModel *m;
	NCHdr *h;
	struct stat st;
	const uint8_t *b, *end;
	uint32_t len, hs;
	float f, inv;
	int fd, i, l, p, L;

	if (!(m = calloc(1, sizeof *m))) {
		snprintf(err, errlen, "out of memory");
		return NULL;
	}
	if ((fd = open(path, O_RDONLY)) < 0 || fstat(fd, &st) < 0) {
		snprintf(err, errlen, "cannot open %s", path);
		goto fail;
	}
	m->mapsz = st.st_size;
	if (m->mapsz < NC_PAGE || m->mapsz % NC_PAGE) {
		snprintf(err, errlen, "not a .ncq file");
		goto fail;
	}
	m->map = mmap(NULL, m->mapsz, PROT_READ, MAP_SHARED, fd, 0);
	close(fd);
	fd = -1;
	if (m->map == MAP_FAILED) {
		m->map = NULL;
		snprintf(err, errlen, "cannot map %s", path);
		goto fail;
	}
	h = (NCHdr *)m->map;
	if (h->magic != NC_MAGIC || h->version != NC_VERSION) {
		snprintf(err, errlen, "not a .ncq file (or made by another version)");
		goto fail;
	}
	L = m->nlayer = h->nlayer;
	m->nhead = h->nhead;
	m->nkv = h->nkv;
	m->dim = h->dim;
	m->ffn = h->ffn;
	m->vocab = h->vocab;
	m->ntok = h->ntok;
	m->seqlen = h->seqlen;
	m->qtype = h->qtype;
	m->bos = h->bos;
	if (L < 1 || L > 1000 || m->nhead < 1 || m->nkv < 1 ||
	    m->nhead % m->nkv || m->dim % m->nhead || h->nten != 2 + 6u * L ||
	    m->seqlen < 1 || m->seqlen > NC_MAXCTX || m->ntok > m->vocab ||
	    m->bos < 0 || m->bos >= m->ntok) {
		snprintf(err, errlen, "bad .ncq header");
		goto fail;
	}
	m->hdim = m->dim / m->nhead;
	m->wq = calloc(6 * L, sizeof *m->wq);
	if (!m->wq) {
		snprintf(err, errlen, "out of memory");
		goto fail;
	}
	m->wk = m->wq + L;
	m->wv = m->wk + L;
	m->wo = m->wv + L;
	m->wfc = m->wo + L;
	m->wpr = m->wfc + L;
	if (setmat(m, &m->wte, &h->t[0]) || setmat(m, &m->lm, &h->t[1]))
		goto bad;
	for (l = 0; l < L; l++) {
		const NCTen *t = &h->t[2 + 6 * l];

		if (setmat(m, &m->wq[l], t) || setmat(m, &m->wk[l], t + 1) ||
		    setmat(m, &m->wv[l], t + 2) || setmat(m, &m->wo[l], t + 3) ||
		    setmat(m, &m->wfc[l], t + 4) || setmat(m, &m->wpr[l], t + 5))
			goto bad;
		if (m->wq[l].rows != m->dim || m->wk[l].rows != m->nkv * m->hdim ||
		    m->wv[l].rows != m->nkv * m->hdim || m->wo[l].cols != m->dim ||
		    m->wfc[l].rows != m->ffn || m->wpr[l].cols != m->ffn ||
		    m->wq[l].cols != m->dim || m->wpr[l].rows != m->dim)
			goto bad;
	}
	if (m->wte.type == NC_Q4 || m->wte.rows != m->vocab ||
	    m->lm.rows != m->vocab || m->wte.cols != m->dim)
		goto bad;

	/* tokenizer */
	if (h->tokoff + h->toklen > m->mapsz)
		goto bad;
	m->tb = calloc(m->ntok, sizeof *m->tb);
	m->tl = calloc(m->ntok, sizeof *m->tl);
	m->special = calloc(m->ntok, 1);
	for (hs = 1; hs < 2u * m->ntok; hs <<= 1)
		;
	m->hash = malloc(hs * sizeof *m->hash);
	m->hmask = hs - 1;
	if (!m->tb || !m->tl || !m->special || !m->hash) {
		snprintf(err, errlen, "out of memory");
		goto fail;
	}
	memset(m->hash, 0xff, hs * sizeof *m->hash);
	b = m->map + h->tokoff;
	end = b + h->toklen;
	for (i = 0; i < m->ntok; i++) {
		if (end - b < 4)
			goto bad;
		memcpy(&len, b, 4);
		m->special[i] = len >> 31;
		len &= 0x7fffffff;
		if ((uint64_t)(end - b - 4) < len)
			goto bad;
		m->tb[i] = b + 4;
		m->tl[i] = len;
		b += 4 + len;
		if (!m->special[i] && len > 0) {
			uint32_t k = fnv(m->tb[i], len) & m->hmask;

			while (m->hash[k] >= 0)
				k = (k + 1) & m->hmask;
			m->hash[k] = i;
		}
	}

	/* rotary tables, as nanochat: fp32 math, kept in bf16 */
	m->cosv = malloc(sizeof(float) * m->seqlen * m->hdim / 2);
	m->sinv = malloc(sizeof(float) * m->seqlen * m->hdim / 2);
	if (!m->cosv || !m->sinv) {
		snprintf(err, errlen, "out of memory");
		goto fail;
	}
	for (i = 0; i < m->hdim / 2; i++) {
		inv = 1.0f / powf(10000.0f, (float)(2 * i) / m->hdim);
		for (p = 0; p < m->seqlen; p++) {
			f = (float)p * inv;
			m->cosv[p * m->hdim / 2 + i] = nc_bf(nc_tobf(cosf(f)));
			m->sinv[p * m->hdim / 2 + i] = nc_bf(nc_tobf(sinf(f)));
		}
	}
	return m;
bad:
	snprintf(err, errlen, "corrupt .ncq file");
fail:
	if (fd >= 0)
		close(fd);
	nc_free(m);
	return NULL;
}

void
nc_free(NCModel *m)
{
	if (!m)
		return;
	if (m->map)
		munmap(m->map, m->mapsz);
	free(m->wq);
	free(m->tb);
	free(m->tl);
	free(m->special);
	free(m->hash);
	free(m->cosv);
	free(m->sinv);
	free(m);
}

/* ---- tokenizer ---- */

enum { CL, CN, CS, CO };	/* letter, number, space, other */

static int
inr(uint32_t c, uint32_t a, uint32_t b)
{
	return c >= a && c <= b;
}

/*
 * Unicode classes for the split pattern. ASCII is exact. Above ASCII
 * this is a table of the common space, digit and symbol ranges; the
 * rest counts as a letter. Any byte string still encodes and decodes
 * exactly, the table only decides where pieces split.
 */
static int
cclass(uint32_t c)
{
	if (c < 128) {
		if (inr(c | 32, 'a', 'z'))
			return CL;
		if (inr(c, '0', '9'))
			return CN;
		if (c == ' ' || inr(c, 9, 13))
			return CS;
		return CO;
	}
	if (c == 0x85 || c == 0xa0 || c == 0x1680 || inr(c, 0x2000, 0x200a) ||
	    c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000)
		return CS;
	if (c == 0xb2 || c == 0xb3 || c == 0xb9 || inr(c, 0xbc, 0xbe) ||
	    inr(c, 0x660, 0x669) || inr(c, 0x6f0, 0x6f9) || inr(c, 0x966, 0x96f) ||
	    inr(c, 0x9e6, 0x9ef) || inr(c, 0xe50, 0xe59) || c == 0x2070 ||
	    inr(c, 0x2074, 0x2079) || inr(c, 0x2080, 0x2089) ||
	    inr(c, 0x2150, 0x2189) || inr(c, 0x2460, 0x249b) ||
	    inr(c, 0x24ea, 0x24ff) || inr(c, 0x2776, 0x2793) || c == 0x3007 ||
	    inr(c, 0x3021, 0x3029) || inr(c, 0xff10, 0xff19))
		return CN;
	if (c == 0xaa || c == 0xb5 || c == 0xba)
		return CL;
	if (inr(c, 0x80, 0xbf) || c == 0xd7 || c == 0xf7 ||
	    inr(c, 0x300, 0x36f) || inr(c, 0x2000, 0x2bff) ||
	    inr(c, 0x3000, 0x3004) || inr(c, 0x3008, 0x3020) ||
	    inr(c, 0x3030, 0x303f) || inr(c, 0xe000, 0xf8ff) ||
	    inr(c, 0xfe00, 0xfe6f) || c == 0xfeff || inr(c, 0xff00, 0xff0f) ||
	    inr(c, 0xff1a, 0xff20) || inr(c, 0xff3b, 0xff40) ||
	    inr(c, 0xff5b, 0xff65) || inr(c, 0x1f000, 0x1faff) || c >= 0xe0000)
		return CO;
	return CL;
}

static int
nl(uint32_t c)
{
	return c == '\r' || c == '\n';
}

/*
 * Length in code points of the next piece at i, following
 * '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}|
 *  ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
 */
static int
piece(const uint32_t *c, const uint8_t *k, int i, int n)
{
	uint32_t a, b;
	int j, e;

	if (c[i] == '\'' && i + 1 < n && c[i + 1] < 128) {
		a = c[i + 1] | 32;
		if (a == 's' || a == 'd' || a == 'm' || a == 't')
			return 2;
		if (i + 2 < n && c[i + 2] < 128) {
			b = c[i + 2] | 32;
			if ((a == 'l' && b == 'l') || (a == 'v' && b == 'e') ||
			    (a == 'r' && b == 'e'))
				return 3;
		}
	}
	j = i;
	if (k[j] != CL && k[j] != CN && !nl(c[j]))
		j++;
	if (j < n && k[j] == CL) {
		while (j < n && k[j] == CL)
			j++;
		return j - i;
	}
	if (k[i] == CN)
		return i + 1 < n && k[i + 1] == CN ? 2 : 1;
	j = i + (c[i] == ' ');
	if (j < n && k[j] == CO) {
		while (j < n && k[j] == CO)
			j++;
		while (j < n && nl(c[j]))
			j++;
		return j - i;
	}
	for (e = i; e < n && k[e] == CS; e++)
		;
	for (j = e; j > i; j--)
		if (nl(c[j - 1]))
			return j - i;
	if (e == n || e - i == 1)
		return e > i ? e - i : 1;
	return e - i - 1;
}

static int
lookup(NCModel *m, const uint8_t *p, int n)
{
	uint32_t k = fnv(p, n) & m->hmask;
	int id;

	for (; (id = m->hash[k]) >= 0; k = (k + 1) & m->hmask)
		if (m->tl[id] == n && memcmp(m->tb[id], p, n) == 0)
			return id;
	return -1;
}

/* byte pair merge, as tiktoken: lowest rank first */
static int
bpe(NCModel *m, const uint8_t *p, int n, int *out, int max)
{
	int *pos, *rk, np, i, best, id, cnt = 0;

	if ((id = lookup(m, p, n)) >= 0) {
		if (max > 0)
			out[cnt++] = id;
		return cnt;
	}
	pos = malloc(sizeof(int) * (n + 1));
	rk = malloc(sizeof(int) * (n + 1));
	if (!pos || !rk) {
		free(pos);
		free(rk);
		return 0;
	}
	for (np = n + 1, i = 0; i < np; i++)
		pos[i] = i;
	for (i = 0; i < np; i++)
		rk[i] = i + 2 < np ? lookup(m, p + i, 2) : -1;
	for (;;) {
		for (best = -1, i = 0; i + 2 < np; i++)
			if (rk[i] >= 0 && (best < 0 || rk[i] < rk[best]))
				best = i;
		if (best < 0)
			break;
		memmove(pos + best + 1, pos + best + 2, sizeof(int) * (np - best - 2));
		memmove(rk + best + 1, rk + best + 2, sizeof(int) * (np - best - 2));
		np--;
		rk[best] = best + 2 < np ?
		    lookup(m, p + pos[best], pos[best + 2] - pos[best]) : -1;
		if (best > 0)
			rk[best - 1] = lookup(m, p + pos[best - 1],
			    pos[best + 1] - pos[best - 1]);
	}
	for (i = 0; i + 1 < np && cnt < max; i++)
		if ((id = lookup(m, p + pos[i], pos[i + 1] - pos[i])) >= 0)
			out[cnt++] = id;
	free(pos);
	free(rk);
	return cnt;
}

/* decode one UTF-8 code point; bad bytes count as one "other" char */
static int
utf8(const uint8_t *s, size_t n, uint32_t *cp)
{
	int len, i;
	uint32_t c = s[0];

	if (c < 0x80)
		len = 1;
	else if (c >= 0xc2 && c < 0xe0)
		len = 2, c &= 0x1f;
	else if (c >= 0xe0 && c < 0xf0)
		len = 3, c &= 0x0f;
	else if (c >= 0xf0 && c < 0xf5)
		len = 4, c &= 0x07;
	else
		goto bad;
	if ((size_t)len > n)
		goto bad;
	for (i = 1; i < len; i++) {
		if ((s[i] & 0xc0) != 0x80)
			goto bad;
		c = c << 6 | (s[i] & 0x3f);
	}
	*cp = c;
	return len;
bad:
	*cp = 0xfffd;
	return 1;
}

int
nc_encode(NCModel *m, const char *str, size_t n, int *out, int max)
{
	const uint8_t *s = (const uint8_t *)str;
	uint32_t *c;
	size_t *at;
	uint8_t *k;
	int nc = 0, i, len, cnt = 0;
	size_t b;

	c = malloc(sizeof *c * (n + 1));
	at = malloc(sizeof *at * (n + 1));
	k = malloc(n + 1);
	if (!c || !at || !k)
		goto done;
	for (b = 0; b < n; b += len, nc++) {
		at[nc] = b;
		len = utf8(s + b, n - b, &c[nc]);
		k[nc] = c[nc] == 0xfffd && len == 1 ? CO : cclass(c[nc]);
	}
	at[nc] = n;
	for (i = 0; i < nc && cnt < max; i += len) {
		len = piece(c, k, i, nc);
		cnt += bpe(m, s + at[i], at[i + len] - at[i], out + cnt, max - cnt);
	}
done:
	free(c);
	free(at);
	free(k);
	return cnt;
}

/* ---- sampling ---- */

static float
urand(uint64_t *s)
{
	*s ^= *s >> 12;
	*s ^= *s << 25;
	*s ^= *s >> 27;
	return (float)((*s * 2685821657736338717ull) >> 40) / 16777216.0f;
}

int
nc_sample(NCModel *m, float *lg, float temp, int topk, uint64_t *rng)
{
	int n = m->ntok, i, j, cnt = 0, best = 0, *idx;
	float mx, sum, r;

	for (i = 0; i < n; i++) {
		lg[i] = NC_SOFTCAP * tanhf(lg[i] / NC_SOFTCAP);
		if (lg[i] > lg[best])
			best = i;
	}
	if (temp <= 0 || topk == 1)
		return best;
	if (topk <= 0 || topk > n)
		topk = n;
	if (!(idx = malloc(sizeof *idx * topk)))
		return best;
	if (topk == n)
		for (cnt = n, i = 0; i < n; i++)
			idx[i] = i;
	else
		for (i = 0; i < n; i++) {
			if (cnt == topk && lg[i] <= lg[idx[cnt - 1]])
				continue;
			j = cnt < topk ? cnt++ : cnt - 1;
			for (; j > 0 && lg[idx[j - 1]] < lg[i]; j--)
				idx[j] = idx[j - 1];
			idx[j] = i;
		}
	mx = lg[best];
	for (sum = 0, i = 0; i < cnt; i++)
		sum += expf((lg[idx[i]] - mx) / temp);
	r = urand(rng) * sum;
	for (i = 0; i < cnt - 1; i++)
		if ((r -= expf((lg[idx[i]] - mx) / temp)) <= 0)
			break;
	best = idx[i];
	free(idx);
	return best;
}

/* ---- CPU engine ---- */

struct NCCpu {
	NCModel *m;
	int nctx, l, pos;
	float *x, *xb, *q, *k, *v, *att, *hb, *lg, *sc;
	_Float16 *kc, *vc;
};

static void
par(size_t n, void *ctx, void (*fn)(void *, size_t))
{
#ifdef __APPLE__
	dispatch_apply_f(n, DISPATCH_APPLY_AUTO, ctx, fn);
#else
	size_t i;

	for (i = 0; i < n; i++)
		fn(ctx, i);
#endif
}

static float
rowdot(const NCMat *w, int r, const float *x)
{
	int nb = w->cols / NC_BLK, b, j;
	const uint16_t *s = w->s ? w->s + (size_t)r * nb : NULL;
	float sum = 0, t;

	if (w->type == NC_BF16) {
		const uint16_t *q = (const uint16_t *)w->q + (size_t)r * w->cols;

		for (b = 0; b < nb; b++, q += NC_BLK, x += NC_BLK) {
			for (t = 0, j = 0; j < NC_BLK; j++)
				t += nc_bf(q[j]) * x[j];
			sum += t;
		}
	} else if (w->type == NC_Q8) {
		const int8_t *q = (const int8_t *)w->q + (size_t)r * w->cols;

		for (b = 0; b < nb; b++, q += NC_BLK, x += NC_BLK) {
			for (t = 0, j = 0; j < NC_BLK; j++)
				t += q[j] * x[j];
			sum += t * nc_bf(s[b]);
		}
	} else {
		const uint8_t *q = w->q + (size_t)r * w->cols / 2;

		for (b = 0; b < nb; b++, q += NC_BLK / 2, x += NC_BLK) {
			for (t = 0, j = 0; j < NC_BLK / 2; j++)
				t += ((q[j] & 15) - 8) * x[j] +
				    ((q[j] >> 4) - 8) * x[j + 16];
			sum += t * nc_bf(s[b]);
		}
	}
	return sum;
}

typedef struct {
	const NCMat *w;
	const float *x;
	float *y;
	int acc;
} MV;

static void
mvchunk(void *ctx, size_t c)
{
	MV *a = ctx;
	int r = c * 16, e = r + 16 > a->w->rows ? a->w->rows : r + 16;

	for (; r < e; r++)
		a->y[r] = (a->acc ? a->y[r] : 0) + rowdot(a->w, r, a->x);
}

static void
mv(const NCMat *w, const float *x, float *y, int acc)
{
	MV a = { w, x, y, acc };

	par((w->rows + 15) / 16, &a, mvchunk);
}

static void
rms(const float *x, float *y, int n)
{
	float ss = 0, r;
	int i;

	for (i = 0; i < n; i++)
		ss += x[i] * x[i];
	r = 1.0f / sqrtf(ss / n + NC_EPS);
	for (i = 0; i < n; i++)
		y[i] = x[i] * r;
}

/* rotate pairs (i, i+d/2) as nanochat, then QK norm */
static void
ropenorm(float *v, const float *cs, const float *sn, int hd)
{
	int i, h = hd / 2;
	float a, b;

	for (i = 0; i < h; i++) {
		a = v[i];
		b = v[i + h];
		v[i] = a * cs[i] + b * sn[i];
		v[i + h] = -a * sn[i] + b * cs[i];
	}
	rms(v, v, hd);
}

static void
attnhead(void *ctx, size_t h)
{
	NCCpu *c = ctx;
	NCModel *m = c->m;
	int hd = m->hdim, kv = h / (m->nhead / m->nkv), p, i;
	size_t stride = (size_t)m->nkv * hd;
	const _Float16 *kb = c->kc + (size_t)c->l * c->nctx * stride + kv * hd;
	const _Float16 *vb = c->vc + (size_t)c->l * c->nctx * stride + kv * hd;
	const float *q = c->q + h * hd;
	float *sc = c->sc + h * c->nctx, *o = c->att + h * hd;
	float scale = 1.0f / sqrtf(hd), mx = -INFINITY, sum = 0, d;

	for (p = 0; p <= c->pos; p++) {
		for (d = 0, i = 0; i < hd; i++)
			d += q[i] * (float)kb[p * stride + i];
		sc[p] = d * scale;
		if (sc[p] > mx)
			mx = sc[p];
	}
	for (p = 0; p <= c->pos; p++)
		sum += sc[p] = expf(sc[p] - mx);
	for (i = 0; i < hd; i++)
		o[i] = 0;
	for (p = 0; p <= c->pos; p++)
		for (d = sc[p] / sum, i = 0; i < hd; i++)
			o[i] += d * (float)vb[p * stride + i];
}

NCCpu *
ncc_new(NCModel *m, int nctx)
{
	NCCpu *c = calloc(1, sizeof *c);
	size_t kvn;

	if (!c)
		return NULL;
	if (nctx > m->seqlen || nctx < 1)
		nctx = m->seqlen;
	c->m = m;
	c->nctx = nctx;
	kvn = (size_t)m->nlayer * nctx * m->nkv * m->hdim;
	c->x = malloc(sizeof(float) * m->dim);
	c->xb = malloc(sizeof(float) * m->dim);
	c->q = malloc(sizeof(float) * m->dim);
	c->k = malloc(sizeof(float) * m->nkv * m->hdim);
	c->v = malloc(sizeof(float) * m->nkv * m->hdim);
	c->att = malloc(sizeof(float) * m->dim);
	c->hb = malloc(sizeof(float) * m->ffn);
	c->lg = malloc(sizeof(float) * m->vocab);
	c->sc = malloc(sizeof(float) * m->nhead * nctx);
	c->kc = malloc(sizeof(_Float16) * kvn);
	c->vc = malloc(sizeof(_Float16) * kvn);
	if (!c->x || !c->xb || !c->q || !c->k || !c->v || !c->att || !c->hb ||
	    !c->lg || !c->sc || !c->kc || !c->vc) {
		ncc_free(c);
		return NULL;
	}
	return c;
}

void
ncc_free(NCCpu *c)
{
	if (!c)
		return;
	free(c->x);
	free(c->xb);
	free(c->q);
	free(c->k);
	free(c->v);
	free(c->att);
	free(c->hb);
	free(c->lg);
	free(c->sc);
	free(c->kc);
	free(c->vc);
	free(c);
}

float *
ncc_step(NCCpu *c, int tok, int pos, int logits)
{
	NCModel *m = c->m;
	int D = m->dim, hd = m->hdim, KV = m->nkv, i, l;
	size_t stride = (size_t)KV * hd, at;
	const float *cs = m->cosv + (size_t)pos * hd / 2;
	const float *sn = m->sinv + (size_t)pos * hd / 2;
	const int8_t *e;
	const uint16_t *eb;

	if (tok < 0 || tok >= m->vocab || pos < 0 || pos >= c->nctx)
		return NULL;
	if (m->wte.type == NC_BF16) {
		eb = (const uint16_t *)m->wte.q + (size_t)tok * D;
		for (i = 0; i < D; i++)
			c->x[i] = nc_bf(eb[i]);
	} else {
		e = (const int8_t *)m->wte.q + (size_t)tok * D;
		for (i = 0; i < D; i++)
			c->x[i] = e[i] * nc_bf(m->wte.s[(size_t)tok * (D / NC_BLK) + i / NC_BLK]);
	}
	rms(c->x, c->x, D);
	c->pos = pos;
	for (l = 0; l < m->nlayer; l++) {
		c->l = l;
		rms(c->x, c->xb, D);
		mv(&m->wq[l], c->xb, c->q, 0);
		mv(&m->wk[l], c->xb, c->k, 0);
		mv(&m->wv[l], c->xb, c->v, 0);
		for (i = 0; i < m->nhead; i++)
			ropenorm(c->q + i * hd, cs, sn, hd);
		at = ((size_t)l * c->nctx + pos) * stride;
		for (i = 0; i < KV; i++)
			ropenorm(c->k + i * hd, cs, sn, hd);
		for (i = 0; i < KV * hd; i++) {
			c->kc[at + i] = (_Float16)c->k[i];
			c->vc[at + i] = (_Float16)c->v[i];
		}
		par(m->nhead, c, attnhead);
		mv(&m->wo[l], c->att, c->x, 1);
		rms(c->x, c->xb, D);
		mv(&m->wfc[l], c->xb, c->hb, 0);
		for (i = 0; i < m->ffn; i++)
			c->hb[i] = c->hb[i] > 0 ? c->hb[i] * c->hb[i] : 0;
		mv(&m->wpr[l], c->hb, c->x, 1);
	}
	if (!logits)
		return NULL;
	rms(c->x, c->xb, D);
	mv(&m->lm, c->xb, c->lg, 0);
	return c->lg;
}
