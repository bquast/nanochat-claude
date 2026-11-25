/*
 * kernels.metal - one-token decode for nanochat d34.
 * Same math, same order as the CPU path in nc.c.
 */
#include <metal_stdlib>
using namespace metal;

#define MAXCTX 4096			/* NC_MAXCTX */

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
