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

@implementation NCGpu {
	NCModel *m;
	int nctx;
	id<MTLDevice> dev;
	id<MTLCommandQueue> queue;
	id<MTLBuffer> wb, x, xb, q, k, v, att, hb, lg, kc, vc, cosb, sinb;
	id<MTLComputePipelineState> pq8, pq4, pemb, prms, prope, pattn, prelu;
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
		    "takes at most %.2f GB in one buffer. Use a Q4 file.",
		    m->mapsz / 1e9, dev.maxBufferLength / 1e9];
		return nil;
	}
	wb = [dev newBufferWithBytesNoCopy:m->map length:m->mapsz
	    options:MTLResourceStorageModeShared deallocator:nil];
	if (!wb) {
		*err = @"Metal cannot use the mapped model file.";
		return nil;
	}
	if (!(lib = [dev newDefaultLibrary])) {
		*err = @"The Metal library (kernels.metal) is not in the app.";
		return nil;
	}
	if (!(pq8 = pipeline(dev, lib, @"mv_q8", err)) ||
	    !(pq4 = pipeline(dev, lib, @"mv_q4", err)) ||
	    !(pemb = pipeline(dev, lib, @"embed", err)) ||
	    !(prms = pipeline(dev, lib, @"rmsnorm", err)) ||
	    !(prope = pipeline(dev, lib, @"rope", err)) ||
	    !(pattn = pipeline(dev, lib, @"attn", err)) ||
	    !(prelu = pipeline(dev, lib, @"relu2", err)))
		return nil;
	if (width(pq8, 256) < 256 || width(pq4, 256) < 256) {
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
	[e setComputePipelineState:w->type == NC_Q8 ? pq8 : pq4];
	[e setBuffer:wb offset:w->qoff atIndex:0];
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

	[e setComputePipelineState:pemb];
	[e setBuffer:wb offset:m->wte.qoff atIndex:0];
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
