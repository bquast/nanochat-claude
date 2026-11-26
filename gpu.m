/*
 * gpu.m - run the model with Metal.
 *
 * The .ncq file is mapped once and wrapped as one shared buffer, so the
 * weights are never copied. Each token is ~250 small dispatches in one
 * serial compute encoder; a prompt goes in chunks of 16 tokens per
 * command buffer, with no CPU round trip in between.
 */
#import <Metal/Metal.h>

#include "nc.h"

#define CHUNK 16
#define MSL(...) #__VA_ARGS__

/*
 * The kernels: one-token decode, same math and order as the CPU path in
 * nc.c. They are compiled at launch with newLibraryWithSource, so the
 * build needs no Metal toolchain and a compile error shows in the app.
 * MSL() makes a string of the code; keep # lines in the prefix above it.
 */
static const char *msl =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "#define MAXCTX 4096\n"		/* NC_MAXCTX */
    MSL(

constant float EPS = 1.1920929e-7f;	/* NC_EPS */

static inline float
bf(ushort s)
{
	return as_type<float>(uint(s) << 16);
}

/*
 * y[r] = W[r] . x  (acc: y[r] += ...)
 * 256 threads = 8 simdgroups = 8 rows per threadgroup.
 * Each lane takes every 32nd block of 32 values.
 */
kernel void
mv_q8(device const char *q [[buffer(0)]],
    device const ushort *s [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    constant uint &acc [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint ln [[thread_index_in_simdgroup]])
{
	uint r = tg * 8 + sg, nb = K / 32, b, j;
	float sum = 0, t;

	if (r >= N)
		return;
	device const char4 *qr = (device const char4 *)(q + (ulong)r * K);
	device const ushort *sr = s + (ulong)r * nb;
	device const float4 *xv = (device const float4 *)x;
	for (b = ln; b < nb; b += 32) {
		for (t = 0, j = 0; j < 8; j++)
			t += dot(float4(qr[b * 8 + j]), xv[b * 8 + j]);
		sum += t * bf(sr[b]);
	}
	sum = simd_sum(sum);
	if (ln == 0)
		y[r] = acc ? y[r] + sum : sum;
}

/* BF16: no scales; each lane takes every 32nd block of 32 values */
kernel void
mv_bf16(device const ushort *q [[buffer(0)]],
    device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    constant uint &acc [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint ln [[thread_index_in_simdgroup]])
{
	uint r = tg * 8 + sg, nb = K / 32, b, j;
	float sum = 0;

	if (r >= N)
		return;
	device const ushort4 *qr = (device const ushort4 *)(q + (ulong)r * K);
	device const float4 *xv = (device const float4 *)x;
	for (b = ln; b < nb; b += 32)
		for (j = 0; j < 8; j++)
			sum += dot(as_type<float4>(uint4(qr[b * 8 + j]) << 16),
			    xv[b * 8 + j]);
	sum = simd_sum(sum);
	if (ln == 0)
		y[r] = acc ? y[r] + sum : sum;
}

/* Q4: byte j of a block holds value j (low) and value j+16 (high) */
kernel void
mv_q4(device const uchar *q [[buffer(0)]],
    device const ushort *s [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    constant uint &acc [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint ln [[thread_index_in_simdgroup]])
{
	uint r = tg * 8 + sg, nb = K / 32, b, j;
	float sum = 0, t;

	if (r >= N)
		return;
	device const uchar4 *qr = (device const uchar4 *)(q + (ulong)r * (K / 2));
	device const ushort *sr = s + (ulong)r * nb;
	device const float4 *xv = (device const float4 *)x;
	for (b = ln; b < nb; b += 32) {
		for (t = 0, j = 0; j < 4; j++) {
			uchar4 v = qr[b * 4 + j];
			float4 lo = float4(v & uchar4(15)) - 8.0f;
			float4 hi = float4(v >> uchar4(4)) - 8.0f;

			t += dot(lo, xv[b * 8 + j]) + dot(hi, xv[b * 8 + 4 + j]);
		}
		sum += t * bf(sr[b]);
	}
	sum = simd_sum(sum);
	if (ln == 0)
		y[r] = acc ? y[r] + sum : sum;
}

/* x = wte[tok], Q8 */
kernel void
embed(device const char *q [[buffer(0)]],
    device const ushort *s [[buffer(1)]],
    device float *x [[buffer(2)]],
    constant uint &K [[buffer(3)]],
    constant uint &tok [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
	if (i < K)
		x[i] = float(q[(ulong)tok * K + i]) * bf(s[(ulong)tok * (K / 32) + i / 32]);
}

/* y = x / rms(x), one threadgroup */
kernel void
rmsnorm(device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint t [[thread_position_in_threadgroup]],
    uint nt [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint ln [[thread_index_in_simdgroup]])
{
	threadgroup float part[32];
	float ss = 0, r;
	uint i;

	for (i = t; i < n; i += nt)
		ss += x[i] * x[i];
	ss = simd_sum(ss);
	if (ln == 0)
		part[sg] = ss;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (ss = 0, i = 0; i < (nt + 31) / 32; i++)
		ss += part[i];
	r = rsqrt(ss / n + EPS);
	for (i = t; i < n; i += nt)
		y[i] = x[i] * r;
}

/*
 * Rotary embedding then QK norm, one simdgroup per head.
 * Heads 0..nh-1 are queries (in place), the rest are keys: those go
 * to the cache with their values.
 */
kernel void
rope(device float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device half *kc [[buffer(3)]],
    device half *vc [[buffer(4)]],
    device const float *cs [[buffer(5)]],
    device const float *sn [[buffer(6)]],
    constant uint &nh [[buffer(7)]],
    constant uint &hd [[buffer(8)]],
    uint h [[threadgroup_position_in_grid]],
    uint ln [[thread_index_in_simdgroup]])
{
	uint hh = hd / 2, i, c, o;
	float a[8], b[8], ss = 0, r, x1, x2;
	device const float *src = h < nh ? q + h * hd : k + (h - nh) * hd;

	for (i = ln, c = 0; i < hh; i += 32, c++) {
		x1 = src[i];
		x2 = src[i + hh];
		a[c] = x1 * cs[i] + x2 * sn[i];
		b[c] = -x1 * sn[i] + x2 * cs[i];
		ss += a[c] * a[c] + b[c] * b[c];
	}
	ss = simd_sum(ss);
	r = rsqrt(ss / hd + EPS);
	if (h < nh) {
		o = h * hd;
		for (i = ln, c = 0; i < hh; i += 32, c++) {
			q[o + i] = a[c] * r;
			q[o + i + hh] = b[c] * r;
		}
	} else {
		o = (h - nh) * hd;
		for (i = ln, c = 0; i < hh; i += 32, c++) {
			kc[o + i] = half(a[c] * r);
			kc[o + i + hh] = half(b[c] * r);
			vc[o + i] = half(v[o + i]);
			vc[o + i + hh] = half(v[o + i + hh]);
		}
	}
}

/*
 * Causal attention for one new query, one threadgroup per head.
 * kc, vc point at this layer: [pos][kv head][hd].
 */
kernel void
attn(device const float *q [[buffer(0)]],
    device const half *kc [[buffer(1)]],
    device const half *vc [[buffer(2)]],
    device float *o [[buffer(3)]],
    constant uint &np [[buffer(4)]],
    constant uint &grp [[buffer(5)]],
    constant uint &nkv [[buffer(6)]],
    constant uint &hd [[buffer(7)]],
    uint h [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]],
    uint nt [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint ln [[thread_index_in_simdgroup]])
{
	threadgroup float sc[MAXCTX];
	threadgroup float part[32];
	uint nsg = nt / 32, st = nkv * hd, kv = (h / grp) * hd, p, i;
	device const float *qh = q + h * hd;
	float scale = rsqrt(float(hd)), m = -3.0e38f, sum = 0, d;

	for (p = sg; p < np; p += nsg) {
		device const half *kp = kc + p * st + kv;

		for (d = 0, i = ln; i < hd; i += 32)
			d += qh[i] * float(kp[i]);
		d = simd_sum(d);
		if (ln == 0)
			sc[p] = d * scale;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (p = t; p < np; p += nt)
		m = max(m, sc[p]);
	m = simd_max(m);
	if (ln == 0)
		part[sg] = m;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (m = part[0], i = 1; i < nsg; i++)
		m = max(m, part[i]);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (p = t; p < np; p += nt) {
		d = exp(sc[p] - m);
		sc[p] = d;
		sum += d;
	}
	sum = simd_sum(sum);
	if (ln == 0)
		part[sg] = sum;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (sum = 0, i = 0; i < nsg; i++)
		sum += part[i];
	for (i = t; i < hd; i += nt) {
		for (d = 0, p = 0; p < np; p++)
			d += sc[p] * float(vc[p * st + kv + i]);
		o[h * hd + i] = d / sum;
	}
}

/* x = wte[tok], BF16 */
kernel void
embed_bf16(device const ushort *q [[buffer(0)]],
    device float *x [[buffer(2)]],
    constant uint &K [[buffer(3)]],
    constant uint &tok [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
	if (i < K)
		x[i] = bf(q[(ulong)tok * K + i]);
}

/* h = relu(h)^2 */
kernel void
relu2(device float *h [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
	if (i < n) {
		float v = max(h[i], 0.0f);

		h[i] = v * v;
	}
}

);

@implementation NCGpu {
	NCModel *m;
	int nctx;
	id<MTLDevice> dev;
	id<MTLCommandQueue> queue;
	id<MTLBuffer> wb, x, xb, q, k, v, att, hb, lg, kc, vc, cosb, sinb;
	id<MTLComputePipelineState> pq8, pq4, pbf, pemb, pembbf, prms, prope,
	    pattn, prelu;
	NSUInteger nrms, nattn;
}

static id<MTLComputePipelineState>
pipeline(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *name, NSString **err)
{
	id<MTLComputePipelineState> p;
	id<MTLFunction> f;
	NSError *e = nil;

	if (!(f = [lib newFunctionWithName:name])) {
		*err = [NSString stringWithFormat:@"Metal kernel %@ is missing.", name];
		return nil;
	}
	if (!(p = [dev newComputePipelineStateWithFunction:f error:&e]))
		*err = e.localizedDescription;
	return p;
}

/* threads per group: as many as asked, at most what the pipeline takes */
static NSUInteger
width(id<MTLComputePipelineState> p, NSUInteger want)
{
	NSUInteger n = p.maxTotalThreadsPerThreadgroup;

	n = n < want ? n : want;
	return n / 32 * 32;
}

- (instancetype)initWithModel:(NCModel *)mm ctx:(int)n error:(NSString **)err
{
	NSUInteger kvbytes, rope;
	id<MTLLibrary> lib;
	NSError *e = nil;

	if (!(self = [super init]))
		return nil;
	m = mm;
	nctx = n < 1 || n > m->seqlen ? m->seqlen : n;
	if (!(dev = MTLCreateSystemDefaultDevice())) {
		*err = @"This Mac has no Metal device.";
		return nil;
	}
	if (m->mapsz > dev.maxBufferLength) {
		*err = [NSString stringWithFormat:@"The model is %.2f GB, but Metal "
		    "takes at most %.2f GB in one buffer. Use a Q8 or Q4 file.",
		    m->mapsz / 1e9, dev.maxBufferLength / 1e9];
		return nil;
	}
	wb = [dev newBufferWithBytesNoCopy:m->map length:m->mapsz
	    options:MTLResourceStorageModeShared deallocator:nil];
	if (!wb) {
		*err = @"Metal cannot use the mapped model file.";
		return nil;
	}
	if (!(lib = [dev newLibraryWithSource:@(msl) options:nil error:&e])) {
		*err = [NSString stringWithFormat:@"The Metal kernels do not "
		    "compile:\n%@", e.localizedDescription];
		return nil;
	}
	if (!(pq8 = pipeline(dev, lib, @"mv_q8", err)) ||
	    !(pq4 = pipeline(dev, lib, @"mv_q4", err)) ||
	    !(pbf = pipeline(dev, lib, @"mv_bf16", err)) ||
	    !(pemb = pipeline(dev, lib, @"embed", err)) ||
	    !(pembbf = pipeline(dev, lib, @"embed_bf16", err)) ||
	    !(prms = pipeline(dev, lib, @"rmsnorm", err)) ||
	    !(prope = pipeline(dev, lib, @"rope", err)) ||
	    !(pattn = pipeline(dev, lib, @"attn", err)) ||
	    !(prelu = pipeline(dev, lib, @"relu2", err)))
		return nil;
	if (width(pq8, 256) < 256 || width(pq4, 256) < 256 || width(pbf, 256) < 256) {
		*err = @"The GPU does not take 256 threads per group.";
		return nil;
	}
	nrms = width(prms, 1024);
	nattn = width(pattn, 256);

	kvbytes = (NSUInteger)m->nlayer * nctx * m->nkv * m->hdim * 2;
	rope = sizeof(float) * m->seqlen * m->hdim / 2;
#define SHARED(n) [dev newBufferWithLength:(n) options:MTLResourceStorageModeShared]
	x = SHARED(sizeof(float) * m->dim);
	xb = SHARED(sizeof(float) * m->dim);
	q = SHARED(sizeof(float) * m->dim);
	k = SHARED(sizeof(float) * m->nkv * m->hdim);
	v = SHARED(sizeof(float) * m->nkv * m->hdim);
	att = SHARED(sizeof(float) * m->dim);
	hb = SHARED(sizeof(float) * m->ffn);
	lg = SHARED(sizeof(float) * m->vocab);
#undef SHARED
	kc = [dev newBufferWithLength:kvbytes options:MTLResourceStorageModePrivate];
	vc = [dev newBufferWithLength:kvbytes options:MTLResourceStorageModePrivate];
	cosb = [dev newBufferWithBytes:m->cosv length:rope options:MTLResourceStorageModeShared];
	sinb = [dev newBufferWithBytes:m->sinv length:rope options:MTLResourceStorageModeShared];
	queue = [dev newCommandQueue];
	if (!x || !xb || !q || !k || !v || !att || !hb || !lg || !kc || !vc ||
	    !cosb || !sinb || !queue) {
		*err = @"Not enough GPU memory for the context.";
		return nil;
	}
	return self;
}

static void
u32(id<MTLComputeCommandEncoder> e, uint32_t v, NSUInteger i)
{
	[e setBytes:&v length:4 atIndex:i];
}

- (void)mv:(id<MTLComputeCommandEncoder>)e w:(NCMat *)w
    in:(id<MTLBuffer>)in out:(id<MTLBuffer>)out acc:(uint32_t)acc
{
	[e setComputePipelineState:w->type == NC_BF16 ? pbf :
	    w->type == NC_Q8 ? pq8 : pq4];
	[e setBuffer:wb offset:w->qoff atIndex:0];
	if (w->type != NC_BF16)
		[e setBuffer:wb offset:w->soff atIndex:1];
	[e setBuffer:in offset:0 atIndex:2];
	[e setBuffer:out offset:0 atIndex:3];
	u32(e, w->cols, 4);
	u32(e, w->rows, 5);
	u32(e, acc, 6);
	[e dispatchThreadgroups:MTLSizeMake((w->rows + 7) / 8, 1, 1)
	    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

- (void)norm:(id<MTLComputeCommandEncoder>)e
    in:(id<MTLBuffer>)in out:(id<MTLBuffer>)out n:(uint32_t)n
{
	[e setComputePipelineState:prms];
	[e setBuffer:in offset:0 atIndex:0];
	[e setBuffer:out offset:0 atIndex:1];
	u32(e, n, 2);
	[e dispatchThreadgroups:MTLSizeMake(1, 1, 1)
	    threadsPerThreadgroup:MTLSizeMake(nrms, 1, 1)];
}

- (void)token:(id<MTLComputeCommandEncoder>)e tok:(int)tok pos:(int)pos
    logits:(int)want
{
	uint32_t D = m->dim, hd = m->hdim, nh = m->nhead, nkv = m->nkv;
	NSUInteger st = (NSUInteger)nkv * hd * 2, lay = (NSUInteger)nctx * st;
	NSUInteger ro = (NSUInteger)pos * hd / 2 * sizeof(float);
	int l;

	[e setComputePipelineState:m->wte.type == NC_BF16 ? pembbf : pemb];
	[e setBuffer:wb offset:m->wte.qoff atIndex:0];
	if (m->wte.type != NC_BF16)
		[e setBuffer:wb offset:m->wte.soff atIndex:1];
	[e setBuffer:xb offset:0 atIndex:2];
	u32(e, D, 3);
	u32(e, tok, 4);
	[e dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
	[self norm:e in:xb out:x n:D];

	for (l = 0; l < m->nlayer; l++) {
		[self norm:e in:x out:xb n:D];
		[self mv:e w:&m->wq[l] in:xb out:q acc:0];
		[self mv:e w:&m->wk[l] in:xb out:k acc:0];
		[self mv:e w:&m->wv[l] in:xb out:v acc:0];

		[e setComputePipelineState:prope];
		[e setBuffer:q offset:0 atIndex:0];
		[e setBuffer:k offset:0 atIndex:1];
		[e setBuffer:v offset:0 atIndex:2];
		[e setBuffer:kc offset:l * lay + pos * st atIndex:3];
		[e setBuffer:vc offset:l * lay + pos * st atIndex:4];
		[e setBuffer:cosb offset:ro atIndex:5];
		[e setBuffer:sinb offset:ro atIndex:6];
		u32(e, nh, 7);
		u32(e, hd, 8);
		[e dispatchThreadgroups:MTLSizeMake(nh + nkv, 1, 1)
		    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

		[e setComputePipelineState:pattn];
		[e setBuffer:q offset:0 atIndex:0];
		[e setBuffer:kc offset:l * lay atIndex:1];
		[e setBuffer:vc offset:l * lay atIndex:2];
		[e setBuffer:att offset:0 atIndex:3];
		u32(e, pos + 1, 4);
		u32(e, nh / nkv, 5);
		u32(e, nkv, 6);
		u32(e, hd, 7);
		[e dispatchThreadgroups:MTLSizeMake(nh, 1, 1)
		    threadsPerThreadgroup:MTLSizeMake(nattn, 1, 1)];

		[self mv:e w:&m->wo[l] in:att out:x acc:1];
		[self norm:e in:x out:xb n:D];
		[self mv:e w:&m->wfc[l] in:xb out:hb acc:0];

		[e setComputePipelineState:prelu];
		[e setBuffer:hb offset:0 atIndex:0];
		u32(e, m->ffn, 1);
		[e dispatchThreads:MTLSizeMake(m->ffn, 1, 1)
		    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

		[self mv:e w:&m->wpr[l] in:hb out:x acc:1];
	}
	if (want) {
		[self norm:e in:x out:xb n:D];
		[self mv:e w:&m->lm in:xb out:lg acc:0];
	}
}

- (float *)run:(const int *)toks count:(int)n pos:(int)pos
{
	id<MTLCommandBuffer> cb = nil;
	id<MTLComputeCommandEncoder> e;
	int i, j, c;

	if (n < 1 || pos < 0 || pos + n > nctx)
		return NULL;
	for (i = 0; i < n; i++)
		if (toks[i] < 0 || toks[i] >= m->vocab)
			return NULL;
	for (i = 0; i < n; i += c) {
		@autoreleasepool {
			c = n - i < CHUNK ? n - i : CHUNK;
			cb = [queue commandBuffer];
			e = [cb computeCommandEncoder];
			for (j = 0; j < c; j++)
				[self token:e tok:toks[i + j] pos:pos + i + j
				    logits:i + j == n - 1];
			[e endEncoding];
			[cb commit];
		}
	}
	[cb waitUntilCompleted];
	if (cb.status != MTLCommandBufferStatusCompleted)
		return NULL;
	return lg.contents;
}

@end
