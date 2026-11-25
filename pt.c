/*
 * pt.c - read a nanochat PyTorch checkpoint and write a quantized .ncq.
 *
 * A .pt file is a zip (stored, not deflated) with a pickle (data.pkl)
 * and one raw file per storage (data/0, data/1, ...). We map the zip,
 * run a small pickle machine that knows just enough of torch to give
 * tensors, and quantize row by row. Memory use stays low: the 8 GB
 * checkpoint is only mapped, never loaded.
 *
 * tokenizer.pkl is a pickled tiktoken Encoding; the same machine reads it.
 */
#include <dirent.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "nc.h"

#define FAIL(...) do { snprintf(err, errlen, __VA_ARGS__); goto fail; } while (0)

static uint32_t
rd16(const uint8_t *p)
{
	return p[0] | p[1] << 8;
}

static uint32_t
rd32(const uint8_t *p)
{
	return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24;
}

static uint64_t
rd64(const uint8_t *p)
{
	return rd32(p) | (uint64_t)rd32(p + 4) << 32;
}

static int
grow(void *pp, size_t *cap, size_t need, size_t sz)
{
	void **p = pp, *q;
	size_t c = *cap;

	if (need <= c)
		return 0;
	for (c = c ? c : 16; c < need; c *= 2)
		;
	if (!(q = realloc(*p, c * sz)))
		return -1;
	*p = q;
	*cap = c;
	return 0;
}

static int
eq(const char *s, size_t n, const char *lit)
{
	return n == strlen(lit) && memcmp(s, lit, n) == 0;
}

/* ---- files ---- */

typedef struct {
	const uint8_t *p;
	size_t n;
} Map;

static int
mapfile(const char *path, Map *m)
{
	struct stat st;
	void *p;
	int fd;

	if ((fd = open(path, O_RDONLY)) < 0)
		return -1;
	if (fstat(fd, &st) < 0 || st.st_size < 1) {
		close(fd);
		return -1;
	}
	p = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if (p == MAP_FAILED)
		return -1;
	m->p = p;
	m->n = st.st_size;
	return 0;
}

static void
unmap(Map *m)
{
	if (m->p)
		munmap((void *)m->p, m->n);
	m->p = NULL;
}

/* ---- zip (stored entries, zip64) ---- */

typedef struct {
	const char *name;
	int nlen, stored;
	uint64_t off, size;
} ZEnt;

typedef struct {
	ZEnt *e;
	uint64_t n;
} Zip;

static int
zipopen(const uint8_t *p, size_t n, Zip *z, char *err, int errlen)
{
	const uint8_t *eocd = NULL, *x;
	uint64_t cnt, cd, q, k, usize, loff, data;
	size_t i, lo;
	int nl, xl, cl, j, id, sz, o;

	z->e = NULL;
	if (n < 22)
		FAIL("not a zip file");
	lo = n > 65557 ? n - 65557 : 0;
	for (i = n - 22; ; i--) {
		if (rd32(p + i) == 0x06054b50) {
			eocd = p + i;
			break;
		}
		if (i == lo)
			break;
	}
	if (!eocd)
		FAIL("not a zip file (no end record)");
	cnt = rd16(eocd + 10);
	cd = rd32(eocd + 16);
	if ((cnt == 0xffff || cd == 0xffffffff) && eocd - p >= 20 &&
	    rd32(eocd - 20) == 0x07064b50) {
		q = rd64(eocd - 20 + 8);
		if (q + 56 > n || rd32(p + q) != 0x06064b50)
			FAIL("bad zip64 record");
		cnt = rd64(p + q + 32);
		cd = rd64(p + q + 48);
	}
	if (cnt > n / 46)
		FAIL("bad zip directory");
	if (!(z->e = calloc(cnt ? cnt : 1, sizeof *z->e)))
		FAIL("out of memory");
	z->n = cnt;
	for (q = cd, k = 0; k < cnt; k++) {
		if (q + 46 > n || rd32(p + q) != 0x02014b50)
			FAIL("bad zip directory entry");
		usize = rd32(p + q + 24);
		nl = rd16(p + q + 28);
		xl = rd16(p + q + 30);
		cl = rd16(p + q + 32);
		loff = rd32(p + q + 42);
		if (q + 46 + nl + xl + cl > n)
			FAIL("bad zip directory entry");
		x = p + q + 46 + nl;
		for (j = 0; j + 4 <= xl; j += 4 + sz) {
			id = rd16(x + j);
			sz = rd16(x + j + 2);
			if (id != 1 || j + 4 + sz > xl)
				continue;
			o = 4;
			if (usize == 0xffffffff && o + 8 <= sz + 4)
				usize = rd64(x + j + o), o += 8;
			if (rd32(p + q + 20) == 0xffffffff && o + 8 <= sz + 4)
				o += 8;
			if (loff == 0xffffffff && o + 8 <= sz + 4)
				loff = rd64(x + j + o), o += 8;
		}
		if (loff + 30 > n || rd32(p + loff) != 0x04034b50)
			FAIL("bad zip local header");
		data = loff + 30 + rd16(p + loff + 26) + rd16(p + loff + 28);
		if (data + usize > n)
			FAIL("zip entry past end of file (download not complete?)");
		z->e[k].name = (const char *)p + q + 46;
		z->e[k].nlen = nl;
		z->e[k].stored = rd16(p + q + 10) == 0;
		z->e[k].off = data;
		z->e[k].size = usize;
		q += 46 + nl + xl + cl;
	}
	return 0;
fail:
	free(z->e);
	z->e = NULL;
	return -1;
}

static ZEnt *
zipfind(Zip *z, const char *pre, int npre, const char *name)
{
	uint64_t k;
	int n = strlen(name);

	for (k = 0; k < z->n; k++)
		if (z->e[k].nlen == npre + n &&
		    memcmp(z->e[k].name, pre, npre) == 0 &&
		    memcmp(z->e[k].name + npre, name, n) == 0)
			return &z->e[k];
	return NULL;
}

/* ---- pickle ---- */

enum { PNONE, PBOOL, PINT, PFLOAT, PSTR, PBYTES, PTUPLE, PLIST, PDICT,
    PGLOBAL, POBJ, PSTORE, PTENSOR, PMARK };

enum { DF32, DBF16, DF16 };

typedef struct V V;
struct V {
	int t;
	int64_t i, j;		/* int; storage dtype and numel; tensor offset */
	const char *s;		/* str, bytes, global "mod.name", storage key */
	size_t n;
	V **a;			/* items; dict as k0 v0 k1 v1 ... */
	size_t na, ca;
};

typedef struct {
	V **all;
	size_t nall, call;
	char **str;
	size_t nstr, cstr;
	V **st;
	size_t nst, cst;
	V **memo;
	size_t nmemo, cmemo;
} VM;

static V markv = { .t = PMARK };

static V *
mk(VM *vm, int t)
{
	V *v;

	if (grow(&vm->all, &vm->call, vm->nall + 1, sizeof *vm->all))
		return NULL;
	if (!(v = calloc(1, sizeof *v)))
		return NULL;
	v->t = t;
	vm->all[vm->nall++] = v;
	return v;
}

static int
add(V *v, V *x)
{
	if (grow(&v->a, &v->ca, v->na + 1, sizeof *v->a))
		return -1;
	v->a[v->na++] = x;
	return 0;
}

static void
vmfree(VM *vm)
{
	size_t k;

	for (k = 0; k < vm->nall; k++) {
		free(vm->all[k]->a);
		free(vm->all[k]);
	}
	for (k = 0; k < vm->nstr; k++)
		free(vm->str[k]);
	free(vm->all);
	free(vm->str);
	free(vm->st);
	free(vm->memo);
	memset(vm, 0, sizeof *vm);
}

static int
push(VM *vm, V *v)
{
	if (!v || grow(&vm->st, &vm->cst, vm->nst + 1, sizeof *vm->st))
		return -1;
	vm->st[vm->nst++] = v;
	return 0;
}

static V *
pop(VM *vm)
{
	if (vm->nst == 0 || vm->st[vm->nst - 1] == &markv)
		return NULL;
	return vm->st[--vm->nst];
}

static V *
top(VM *vm)
{
	if (vm->nst == 0 || vm->st[vm->nst - 1] == &markv)
		return NULL;
	return vm->st[vm->nst - 1];
}

/* items above the last mark; removes them and the mark */
static int
popmark(VM *vm, V ***items, size_t *n)
{
	size_t k = vm->nst;

	while (k > 0 && vm->st[k - 1] != &markv)
		k--;
	if (k == 0)
		return -1;
	*items = vm->st + k;
	*n = vm->nst - k;
	vm->nst = k - 1;
	return 0;
}

static int
memo(VM *vm, uint64_t idx, V *v)
{
	size_t k;

	if (!v || idx > 1u << 26)
		return -1;
	if (grow(&vm->memo, &vm->cmemo, idx + 1, sizeof *vm->memo))
		return -1;
	for (k = vm->nmemo; k <= idx; k++)
		vm->memo[k] = NULL;
	if (idx >= vm->nmemo)
		vm->nmemo = idx + 1;
	vm->memo[idx] = v;
	return 0;
}

static V *
global(VM *vm, const char *mod, size_t nm, const char *name, size_t nn)
{
	V *v = mk(vm, PGLOBAL);
	char *s;

	if (!v || grow(&vm->str, &vm->cstr, vm->nstr + 1, sizeof *vm->str))
		return NULL;
	if (!(s = malloc(nm + nn + 2)))
		return NULL;
	memcpy(s, mod, nm);
	s[nm] = '.';
	memcpy(s + nm + 1, name, nn);
	s[nm + nn + 1] = 0;
	vm->str[vm->nstr++] = s;
	v->s = s;
	v->n = nm + nn + 1;
	return v;
}

static V *
seq(VM *vm, int t, V **items, size_t n)
{
	V *v = mk(vm, t);
	size_t k;

	if (!v)
		return NULL;
	for (k = 0; k < n; k++)
		if (add(v, items[k]))
			return NULL;
	return v;
}

static int
isglobal(V *v, const char *name)
{
	return v && v->t == PGLOBAL && strcmp(v->s, name) == 0;
}

static V *
reduce(VM *vm, V *f, V *args)
{
	V *v, *st;

	if (!f || !args || args->t != PTUPLE)
		return NULL;
	if (isglobal(f, "collections.OrderedDict"))
		return mk(vm, PDICT);
	if (isglobal(f, "torch._utils._rebuild_parameter") && args->na >= 1)
		return args->a[0];
	if (isglobal(f, "torch._utils._rebuild_tensor_v2") && args->na >= 4) {
		st = args->a[0];
		if (st->t != PSTORE || args->a[1]->t != PINT ||
		    args->a[2]->t != PTUPLE || args->a[3]->t != PTUPLE)
			return NULL;
		if (!(v = mk(vm, PTENSOR)))
			return NULL;
		v->i = args->a[1]->i;
		if (add(v, st) || add(v, args->a[2]) || add(v, args->a[3]))
			return NULL;
		return v;
	}
	if (!(v = mk(vm, POBJ)) || add(v, f) || add(v, args))
		return NULL;
	return v;
}

static V *
persid(VM *vm, V *pid)
{
	V *v, *cls;

	if (!pid || pid->t != PTUPLE || pid->na < 5 || pid->a[0]->t != PSTR ||
	    !eq(pid->a[0]->s, pid->a[0]->n, "storage"))
		return NULL;
	cls = pid->a[1];
	if (cls->t != PGLOBAL || pid->a[2]->t != PSTR || pid->a[4]->t != PINT)
		return NULL;
	if (!(v = mk(vm, PSTORE)))
		return NULL;
	if (strcmp(cls->s, "torch.FloatStorage") == 0)
		v->i = DF32;
	else if (strcmp(cls->s, "torch.BFloat16Storage") == 0)
		v->i = DBF16;
	else if (strcmp(cls->s, "torch.HalfStorage") == 0)
		v->i = DF16;
	else
		v->i = -1;
	v->s = pid->a[2]->s;
	v->n = pid->a[2]->n;
	v->j = pid->a[4]->i;
	return v;
}

/* run a pickle; returns the result or NULL */
static V *
unpickle(VM *vm, const uint8_t *p, size_t len, char *err, int errlen)
{
	const uint8_t *end = p + len, *e, *e2;
	V *v, *w, *x, **it;
	uint64_t n;
	size_t ni, k;
	int op;

#define NEED(k) do { if ((uint64_t)(end - p) < (uint64_t)(k)) goto trunc; } while (0)
#define PUSH(v) do { if (push(vm, (v))) goto bad; } while (0)
	memset(vm, 0, sizeof *vm);
	while (p < end) {
		op = *p++;
		switch (op) {
		case 0x80:	/* PROTO */
			NEED(1);
			p++;
			break;
		case 0x95:	/* FRAME */
			NEED(8);
			p += 8;
			break;
		case '.':	/* STOP */
			if (!(v = pop(vm)))
				goto bad;
			return v;
		case '(':
			PUSH(&markv);
			break;
		case ')':
			PUSH(mk(vm, PTUPLE));
			break;
		case ']':
			PUSH(mk(vm, PLIST));
			break;
		case '}':
			PUSH(mk(vm, PDICT));
			break;
		case 'N':
			PUSH(mk(vm, PNONE));
			break;
		case 0x88:	/* NEWTRUE */
		case 0x89:	/* NEWFALSE */
			if (!(v = mk(vm, PBOOL)))
				goto bad;
			v->i = op == 0x88;
			PUSH(v);
			break;
		case 'K':	/* BININT1 */
			NEED(1);
			if (!(v = mk(vm, PINT)))
				goto bad;
			v->i = *p++;
			PUSH(v);
			break;
		case 'M':	/* BININT2 */
			NEED(2);
			if (!(v = mk(vm, PINT)))
				goto bad;
			v->i = rd16(p);
			p += 2;
			PUSH(v);
			break;
		case 'J':	/* BININT */
			NEED(4);
			if (!(v = mk(vm, PINT)))
				goto bad;
			v->i = (int32_t)rd32(p);
			p += 4;
			PUSH(v);
			break;
		case 0x8a:	/* LONG1 */
			NEED(1);
			n = *p++;
			NEED(n);
			if (n > 8 || !(v = mk(vm, PINT)))
				goto bad;
			v->i = 0;
			for (k = 0; k < n; k++)
				v->i |= (int64_t)((uint64_t)p[k] << (8 * k));
			if (n > 0 && n < 8 && (p[n - 1] & 0x80))
				v->i -= (int64_t)1 << (8 * n);
			p += n;
			PUSH(v);
			break;
		case 'G':	/* BINFLOAT */
			NEED(8);
			p += 8;
			PUSH(mk(vm, PFLOAT));
			break;
		case 'X':	/* BINUNICODE */
		case 'B':	/* BINBYTES */
			NEED(4);
			n = rd32(p);
			p += 4;
			goto str;
		case 0x8c:	/* SHORT_BINUNICODE */
		case 'C':	/* SHORT_BINBYTES */
			NEED(1);
			n = *p++;
			goto str;
		case 0x8d:	/* BINUNICODE8 */
		case 0x8e:	/* BINBYTES8 */
			NEED(8);
			n = rd64(p);
			p += 8;
		str:
			NEED(n);
			if (!(v = mk(vm, (op == 'X' || op == 0x8c || op == 0x8d) ?
			    PSTR : PBYTES)))
				goto bad;
			v->s = (const char *)p;
			v->n = n;
			p += n;
			PUSH(v);
			break;
		case 'c':	/* GLOBAL "mod\nname\n" */
			if (!(e = memchr(p, '\n', end - p)) ||
			    !(e2 = memchr(e + 1, '\n', end - e - 1)))
				goto trunc;
			v = global(vm, (const char *)p, e - p,
			    (const char *)e + 1, e2 - e - 1);
			p = e2 + 1;
			PUSH(v);
			break;
		case 0x93:	/* STACK_GLOBAL */
			w = pop(vm);
			v = pop(vm);
			if (!v || !w || v->t != PSTR || w->t != PSTR)
				goto bad;
			PUSH(global(vm, v->s, v->n, w->s, w->n));
			break;
		case 'q':	/* BINPUT */
			NEED(1);
			if (memo(vm, *p++, top(vm)))
				goto bad;
			break;
		case 'r':	/* LONG_BINPUT */
			NEED(4);
			if (memo(vm, rd32(p), top(vm)))
				goto bad;
			p += 4;
			break;
		case 0x94:	/* MEMOIZE */
			if (memo(vm, vm->nmemo, top(vm)))
				goto bad;
			break;
		case 'h':	/* BINGET */
		case 'j':	/* LONG_BINGET */
			NEED(op == 'h' ? 1 : 4);
			n = op == 'h' ? *p : rd32(p);
			p += op == 'h' ? 1 : 4;
			if (n >= vm->nmemo || !vm->memo[n])
				goto bad;
			PUSH(vm->memo[n]);
			break;
		case 't':	/* TUPLE */
		case 'l':	/* LIST */
			if (popmark(vm, &it, &ni))
				goto bad;
			PUSH(seq(vm, op == 't' ? PTUPLE : PLIST, it, ni));
			break;
		case 'd':	/* DICT */
			if (popmark(vm, &it, &ni) || ni % 2)
				goto bad;
			PUSH(seq(vm, PDICT, it, ni));
			break;
		case 0x85:	/* TUPLE1 */
		case 0x86:	/* TUPLE2 */
		case 0x87:	/* TUPLE3 */
			ni = op - 0x84;
			if (vm->nst < ni)
				goto bad;
			for (k = vm->nst - ni; k < vm->nst; k++)
				if (vm->st[k] == &markv)
					goto bad;
			vm->nst -= ni;
			PUSH(seq(vm, PTUPLE, vm->st + vm->nst, ni));
			break;
		case 'a':	/* APPEND */
			w = pop(vm);
			v = top(vm);
			if (!w || !v || v->t != PLIST || add(v, w))
				goto bad;
			break;
		case 'e':	/* APPENDS */
			if (popmark(vm, &it, &ni) || !(v = top(vm)))
				goto bad;
			for (k = 0; k < ni; k++)
				if (v->t == PLIST && add(v, it[k]))
					goto bad;
			break;
		case 's':	/* SETITEM */
			w = pop(vm);
			x = pop(vm);
			v = top(vm);
			if (!w || !x || !v)
				goto bad;
			if (v->t == PDICT && (add(v, x) || add(v, w)))
				goto bad;
			break;
		case 'u':	/* SETITEMS */
			if (popmark(vm, &it, &ni) || ni % 2 || !(v = top(vm)))
				goto bad;
			for (k = 0; k < ni && v->t == PDICT; k++)
				if (add(v, it[k]))
					goto bad;
			break;
		case 'R':	/* REDUCE */
		case 0x81:	/* NEWOBJ */
			w = pop(vm);
			v = pop(vm);
			if (op == 'R')
				x = reduce(vm, v, w);
			else if (!v || !w || !(x = mk(vm, POBJ)) ||
			    add(x, v) || add(x, w))
				x = NULL;
			if (!x)
				goto bad;
			PUSH(x);
			break;
		case 'b':	/* BUILD */
			w = pop(vm);
			v = top(vm);
			if (!w || !v)
				goto bad;
			if (v->t == POBJ && add(v, w))
				goto bad;
			break;
		case 'Q':	/* BINPERSID */
			PUSH(persid(vm, pop(vm)));
			break;
		case '0':	/* POP */
			if (vm->nst == 0)
				goto bad;
			vm->nst--;
			break;
		case '1':	/* POP_MARK */
			if (popmark(vm, &it, &ni))
				goto bad;
			break;
		case '2':	/* DUP */
			if (vm->nst == 0)
				goto bad;
			PUSH(vm->st[vm->nst - 1]);
			break;
		default:
			snprintf(err, errlen, "pickle: unsupported opcode 0x%02x", op);
			return NULL;
		}
	}
trunc:
	snprintf(err, errlen, "pickle: truncated");
	return NULL;
bad:
	snprintf(err, errlen, "pickle: unexpected data near opcode 0x%02x", op);
	return NULL;
#undef NEED
#undef PUSH
}

static V *
dget(V *d, const char *key)
{
	size_t k;

	if (!d || d->t != PDICT)
		return NULL;
	for (k = 0; k + 1 < d->na; k += 2)
		if (d->a[k]->t == PSTR && eq(d->a[k]->s, d->a[k]->n, key))
			return d->a[k + 1];
	return NULL;
}

/* ---- tokenizer.pkl ---- */

static int
loadtok(const char *path, uint8_t **blob, uint64_t *bloblen, int *ntok,
    int *bos, char *err, int errlen)
{
	const uint8_t **tb = NULL;
	int *tl = NULL, *sp = NULL, n = 0, id;
	uint8_t *buf = NULL, *b;
	V *top, *st, *ranks, *spec, *k, *v;
	uint64_t len;
	size_t i;
	Map m = { 0 };
	VM vm = { 0 };

	*blob = NULL;
	*bos = -1;
	if (mapfile(path, &m))
		FAIL("cannot read %s", path);
	if (!(top = unpickle(&vm, m.p, m.n, err, errlen)))
		goto fail;
	st = top->t == POBJ && top->na >= 3 ? top->a[2] : top;
	ranks = dget(st, "mergeable_ranks");
	spec = dget(st, "special_tokens");
	if (!ranks || ranks->t != PDICT || !spec || spec->t != PDICT)
		FAIL("tokenizer.pkl: not a tiktoken Encoding");
	for (i = 0; i + 1 < ranks->na; i += 2)
		if (ranks->a[i + 1]->t == PINT && ranks->a[i + 1]->i >= n)
			n = ranks->a[i + 1]->i + 1;
	for (i = 0; i + 1 < spec->na; i += 2)
		if (spec->a[i + 1]->t == PINT && spec->a[i + 1]->i >= n)
			n = spec->a[i + 1]->i + 1;
	if (n < 256 || n > 1 << 22)
		FAIL("tokenizer.pkl: bad vocabulary size %d", n);
	tb = calloc(n, sizeof *tb);
	tl = calloc(n, sizeof *tl);
	sp = calloc(n, sizeof *sp);
	if (!tb || !tl || !sp)
		FAIL("out of memory");
	for (i = 0; i + 1 < ranks->na + spec->na; i += 2) {
		int s = i >= ranks->na;

		k = s ? spec->a[i - ranks->na] : ranks->a[i];
		v = s ? spec->a[i - ranks->na + 1] : ranks->a[i + 1];
		if (k->t != (s ? PSTR : PBYTES) || v->t != PINT || v->i < 0)
			FAIL("tokenizer.pkl: bad entry");
		id = v->i;
		tb[id] = (const uint8_t *)k->s;
		tl[id] = k->n;
		sp[id] = s;
		if (s && (eq(k->s, k->n, "<|bos|>") ||
		    (*bos < 0 && eq(k->s, k->n, "<|endoftext|>"))))
			*bos = id;
	}
	if (*bos < 0)
		FAIL("tokenizer.pkl: no <|bos|> token");
	for (len = 0, id = 0; id < n; id++)
		len += 4 + tl[id];
	if (!(buf = malloc(len)))
		FAIL("out of memory");
	for (b = buf, id = 0; id < n; id++) {
		uint32_t h = tl[id] | (sp[id] ? 0x80000000u : 0);

		memcpy(b, &h, 4);
		memcpy(b + 4, tb[id], tl[id]);
		b += 4 + tl[id];
	}
	*blob = buf;
	*bloblen = len;
	*ntok = n;
	free(tb);
	free(tl);
	free(sp);
	vmfree(&vm);
	unmap(&m);
	return 0;
fail:
	free(buf);
	free(tb);
	free(tl);
	free(sp);
	vmfree(&vm);
	unmap(&m);
	return -1;
}

/* ---- tensors ---- */

typedef struct {
	char name[128];
	int dt, nd;
	int64_t shape[2], stride[2];
	const uint8_t *data;
} PT;

static float
h2f(uint16_t h)
{
	uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 31, m = h & 0x3ff, u;
	float f;

	if (e == 0) {
		f = ldexpf((float)m, -24);
		return s ? -f : f;
	}
	u = e == 31 ? s | 0x7f800000u | m << 13 : s | (e + 112) << 23 | m << 13;
	memcpy(&f, &u, 4);
	return f;
}

static void
getrow(const PT *t, int64_t r, float *out)
{
	int64_t c, K = t->shape[1], e;
	uint32_t u;
	uint16_t h;

	for (c = 0; c < K; c++) {
		e = r * t->stride[0] + c * t->stride[1];
		switch (t->dt) {
		case DF32:
			memcpy(&out[c], t->data + 4 * e, 4);
			break;
		case DBF16:
			memcpy(&h, t->data + 2 * e, 2);
			u = (uint32_t)h << 16;
			memcpy(&out[c], &u, 4);
			break;
		default:
			memcpy(&h, t->data + 2 * e, 2);
			out[c] = h2f(h);
			break;
		}
	}
}

/* collect 2-D tensors of the state dict */
static int
tensors(V *sd, Zip *z, const uint8_t *base, const char *pre, int npre,
    PT **out, int *nout, char *err, int errlen)
{
	PT *pt = NULL, *t;
	V *k, *v, *st, *sh, *sr;
	ZEnt *ze;
	char key[160];
	size_t i;
	int n = 0, es, d;
	int64_t last;

	if (sd->t == PDICT && !dget(sd, "transformer.wte.weight") &&
	    !dget(sd, "_orig_mod.transformer.wte.weight")) {
		if (dget(sd, "model"))
			sd = dget(sd, "model");
		else if (dget(sd, "state_dict"))
			sd = dget(sd, "state_dict");
	}
	if (sd->t != PDICT)
		FAIL("checkpoint: not a state dict");
	if (!(pt = calloc(sd->na / 2 + 1, sizeof *pt)))
		FAIL("out of memory");
	for (i = 0; i + 1 < sd->na; i += 2) {
		k = sd->a[i];
		v = sd->a[i + 1];
		if (k->t != PSTR || v->t != PTENSOR)
			continue;
		st = v->a[0];
		sh = v->a[1];
		sr = v->a[2];
		if (sh->na != 2 || sr->na != 2)
			continue;
		t = &pt[n];
		if (k->n > 10 && memcmp(k->s, "_orig_mod.", 10) == 0)
			snprintf(t->name, sizeof t->name, "%.*s", (int)k->n - 10, k->s + 10);
		else
			snprintf(t->name, sizeof t->name, "%.*s", (int)k->n, k->s);
		if (st->i < 0)
			FAIL("%s: dtype not supported (need float32, bfloat16 or float16)", t->name);
		t->dt = st->i;
		t->nd = 2;
		for (d = 0; d < 2; d++) {
			t->shape[d] = sh->a[d]->i;
			t->stride[d] = sr->a[d]->i;
		}
		es = t->dt == DF32 ? 4 : 2;
		snprintf(key, sizeof key, "data/%.*s", (int)st->n, st->s);
		if (!(ze = zipfind(z, pre, npre, key)) || !ze->stored)
			FAIL("%s: storage %s missing in zip", t->name, key);
		last = v->i + (t->shape[0] - 1) * t->stride[0] +
		    (t->shape[1] - 1) * t->stride[1];
		if (t->shape[0] < 1 || t->shape[1] < 1 || v->i < 0 ||
		    (uint64_t)(last + 1) * es > ze->size)
			FAIL("%s: storage too small", t->name);
		t->data = base + ze->off + (uint64_t)v->i * es;
		n++;
	}
	*out = pt;
	*nout = n;
	return 0;
fail:
	free(pt);
	return -1;
}

/* give back the pages of a tensor we are done with (8 GB file, 8 GB Mac) */
static void
dropsrc(const PT *t)
{
	uintptr_t pg = getpagesize(), a, b;
	int es = t->dt == DF32 ? 4 : 2;

	a = (uintptr_t)t->data & ~(pg - 1);
	b = (uintptr_t)t->data + es * ((t->shape[0] - 1) * t->stride[0] +
	    (t->shape[1] - 1) * t->stride[1] + 1);
	madvise((void *)a, b - a, MADV_DONTNEED);
}

static PT *
find(PT *pt, int n, const char *name)
{
	int i;

	for (i = 0; i < n; i++)
		if (strcmp(pt[i].name, name) == 0)
			return &pt[i];
	return NULL;
}

/* ---- quantize ---- */

static int
clampi(int v, int lo, int hi)
{
	return v < lo ? lo : v > hi ? hi : v;
}

static void
q8row(const float *x, int K, int8_t *q, uint16_t *s)
{
	float amax, d, id;
	int b, j;

	for (b = 0; b < K / NC_BLK; b++, x += NC_BLK, q += NC_BLK) {
		for (amax = 0, j = 0; j < NC_BLK; j++)
			amax = fmaxf(amax, fabsf(x[j]));
		s[b] = nc_tobf(amax / 127.0f);
		d = nc_bf(s[b]);
		id = d > 0 ? 1.0f / d : 0;
		for (j = 0; j < NC_BLK; j++)
			q[j] = clampi((int)lrintf(x[j] * id), -127, 127);
	}
}

static void
q4row(const float *x, int K, uint8_t *q, uint16_t *s)
{
	float amax, m, d, id;
	int b, j, lo, hi;

	for (b = 0; b < K / NC_BLK; b++, x += NC_BLK, q += NC_BLK / 2) {
		for (amax = m = 0, j = 0; j < NC_BLK; j++)
			if (fabsf(x[j]) > amax) {
				amax = fabsf(x[j]);
				m = x[j];
			}
		s[b] = nc_tobf(m / -8.0f);
		d = nc_bf(s[b]);
		id = d != 0 ? 1.0f / d : 0;
		for (j = 0; j < NC_BLK / 2; j++) {
			lo = clampi((int)lrintf(x[j] * id) + 8, 0, 15);
			hi = clampi((int)lrintf(x[j + 16] * id) + 8, 0, 15);
			q[j] = lo | hi << 4;
		}
	}
}

static int
padto(FILE *f, uint64_t to)
{
	static const uint8_t z[4096];
	off_t at = ftello(f);
	uint64_t n;

	if (at < 0 || (uint64_t)at > to)
		return -1;
	for (n = to - at; n > 0; n -= n > sizeof z ? sizeof z : n)
		if (fwrite(z, 1, n > sizeof z ? sizeof z : n, f) == 0)
			return -1;
	return 0;
}

static uint64_t
align(uint64_t v, uint64_t a)
{
	return (v + a - 1) / a * a;
}

static int
jint(const char *js, const char *key, int def)
{
	const char *p;
	char k[64];

	snprintf(k, sizeof k, "\"%s\"", key);
	if (!(p = strstr(js, k)))
		return def;
	for (p += strlen(k); *p == ' ' || *p == '\t' || *p == '\n' || *p == ':'; p++)
		;
	return *p >= '0' && *p <= '9' ? atoi(p) : def;
}

static int
exists(const char *path)
{
	struct stat st;

	return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

/* find model_<step>.pt (latest), meta_<step>.json and tokenizer.pkl */
static int
findfiles(const char *path, char *pt, char *meta, char *tok, size_t n)
{
	char dir[1024], *sl;
	struct dirent *de;
	struct stat st;
	const char *home;
	long step = -1, s;
	DIR *d;

	if (strchr(path, '/'))
		snprintf(dir, sizeof dir, "%s", path);
	else
		snprintf(dir, sizeof dir, "./%s", path);
	*pt = 0;
	if (stat(path, &st) == 0 && S_ISREG(st.st_mode)) {
		if (strlen(dir) > 3 && strcmp(dir + strlen(dir) - 3, ".pt") == 0)
			snprintf(pt, n, "%s", dir);
		if ((sl = strrchr(dir, '/')))
			*sl = 0;
		else
			snprintf(dir, sizeof dir, ".");
	}
	if (!*pt && (d = opendir(dir))) {
		while ((de = readdir(d))) {
			size_t l = strlen(de->d_name);

			if (strncmp(de->d_name, "model_", 6) || l < 10 ||
			    strcmp(de->d_name + l - 3, ".pt"))
				continue;
			s = strtol(de->d_name + 6, NULL, 10);
			if (s > step) {
				step = s;
				snprintf(pt, n, "%s/%s", dir, de->d_name);
			}
		}
		closedir(d);
	}
	if (!*pt)
		return -1;
	*meta = 0;
	sl = strrchr(pt, '/');
	if (sl && strncmp(sl, "/model_", 7) == 0) {
		snprintf(meta, n - 2, "%.*s/meta_%s", (int)(sl - pt), pt, sl + 7);
		strcpy(meta + strlen(meta) - 3, ".json");
	}
	snprintf(tok, n, "%s/tokenizer.pkl", dir);
	if (!exists(tok))
		snprintf(tok, n, "%s/tokenizer/tokenizer.pkl", dir);
	if (!exists(tok) && (home = getenv("HOME")))
		snprintf(tok, n, "%s/.cache/nanochat/tokenizer/tokenizer.pkl", home);
	return 0;
}

int
nc_import(const char *path, const char *out, int qtype, NCProgress cb,
    void *ctx, char *err, int errlen)
{
	static const char *lname[6] = { "attn.c_q", "attn.c_k", "attn.c_v",
	    "attn.c_proj", "mlp.c_fc", "mlp.c_proj" };
	char ptf[1024], metaf[1024], tokf[1024], name[160], msg[256];
	char *js = NULL;
	const char *mc = NULL;
	uint8_t *tokblob = NULL, *qb = NULL, *hdrb = NULL;
	uint16_t *sb = NULL;
	float *row = NULL;
	PT *pt = NULL, **src = NULL, *t;
	NCHdr *h;
	Map m = { 0 }, mm = { 0 };
	Zip z = { 0 };
	VM vm = { 0 };
	ZEnt *ze;
	V *sd;
	FILE *f = NULL;
	uint64_t toklen, off, qbytes, sbytes, done = 0, total = 0;
	int ntok, bos, npt, L, D, H, KV, hd, F, VO, i, l, nten, pre;
	int64_t r;

	if (findfiles(path, ptf, metaf, tokf, sizeof ptf))
		FAIL("no model_*.pt file in %s", path);
	if (cb)
		cb(ctx, 0, "reading tokenizer");
	if (!exists(tokf))
		FAIL("tokenizer.pkl not found next to the checkpoint");
	if (loadtok(tokf, &tokblob, &toklen, &ntok, &bos, err, errlen))
		goto fail;

	if (cb)
		cb(ctx, 0, "reading checkpoint");
	if (mapfile(ptf, &m))
		FAIL("cannot map %s", ptf);
	if (zipopen(m.p, m.n, &z, err, errlen))
		goto fail;
	for (ze = NULL, i = 0; i < (int)z.n; i++)
		if (z.e[i].nlen >= 8 && memcmp(z.e[i].name + z.e[i].nlen - 8,
		    "data.pkl", 8) == 0) {
			ze = &z.e[i];
			break;
		}
	if (!ze || !ze->stored)
		FAIL("%s: no data.pkl, not a torch.save file", ptf);
	pre = ze->nlen - 8;
	{
		ZEnt *bo = zipfind(&z, ze->name, pre, "byteorder");

		if (bo && !(bo->size >= 6 && memcmp(m.p + bo->off, "little", 6) == 0))
			FAIL("checkpoint is big-endian");
	}
	if (!(sd = unpickle(&vm, m.p + ze->off, ze->size, err, errlen)))
		goto fail;
	if (tensors(sd, &z, m.p, ze->name, pre, &pt, &npt, err, errlen))
		goto fail;

	if (!(t = find(pt, npt, "transformer.wte.weight")))
		FAIL("transformer.wte.weight not in checkpoint");
	VO = t->shape[0];
	D = t->shape[1];
	for (L = 0; ; L++) {
		snprintf(name, sizeof name, "transformer.h.%d.attn.c_q.weight", L);
		if (!find(pt, npt, name))
			break;
	}
	H = D / 128;
	if (mapfile(metaf, &mm) == 0) {
		if ((js = malloc(mm.n + 1))) {
			memcpy(js, mm.p, mm.n);
			js[mm.n] = 0;
			mc = strstr(js, "\"model_config\"");
			mc = mc ? mc : js;
			H = jint(mc, "n_head", H);
			if (jint(mc, "n_layer", L) != L)
				FAIL("meta json says %d layers, checkpoint has %d",
				    jint(mc, "n_layer", L), L);
		}
		unmap(&mm);
	}
	if (L < 1 || H < 1 || D % H || D % NC_BLK)
		FAIL("bad model shape: %d layers, dim %d, %d heads", L, D, H);
	hd = D / H;
	snprintf(name, sizeof name, "transformer.h.0.attn.c_k.weight");
	KV = (t = find(pt, npt, name)) ? t->shape[0] / hd : 0;
	snprintf(name, sizeof name, "transformer.h.0.mlp.c_fc.weight");
	F = (t = find(pt, npt, name)) ? t->shape[0] : 0;
	if (KV < 1 || H % KV || hd % 2 || hd > 512 || F % NC_BLK || VO < ntok)
		FAIL("bad model shape: kv heads %d, head dim %d, ffn %d, vocab %d/%d",
		    KV, hd, F, VO, ntok);

	nten = 2 + 6 * L;
	if (sizeof(NCHdr) + nten * sizeof(NCTen) > NC_PAGE)
		FAIL("too many layers");
	if (!(hdrb = calloc(1, NC_PAGE)) || !(src = calloc(nten, sizeof *src)))
		FAIL("out of memory");
	h = (NCHdr *)hdrb;
	src[0] = find(pt, npt, "transformer.wte.weight");
	src[1] = find(pt, npt, "lm_head.weight");
	for (l = 0; l < L; l++)
		for (i = 0; i < 6; i++) {
			snprintf(name, sizeof name, "transformer.h.%d.%s.weight", l, lname[i]);
			src[2 + 6 * l + i] = find(pt, npt, name);
		}
	off = align(NC_PAGE + toklen, NC_PAGE);
	for (i = 0; i < nten; i++) {
		int rows = i < 2 ? VO : (i - 2) % 6 == 4 ? F : (i - 2) % 6 == 1 ||
		    (i - 2) % 6 == 2 ? KV * hd : D;
		int cols = i >= 2 && (i - 2) % 6 == 5 ? F : D;

		if (!(t = src[i]))
			FAIL("tensor %d missing in checkpoint", i);
		if (t->shape[0] != rows || t->shape[1] != cols)
			FAIL("%s: shape %lldx%lld, want %dx%d", t->name,
			    (long long)t->shape[0], (long long)t->shape[1], rows, cols);
		h->t[i].type = i < 2 ? NC_Q8 : qtype;
		h->t[i].rows = rows;
		h->t[i].cols = cols;
		qbytes = (uint64_t)rows * cols / (h->t[i].type == NC_Q8 ? 1 : 2);
		sbytes = (uint64_t)rows * (cols / NC_BLK) * 2;
		h->t[i].qoff = off;
		off = align(off + qbytes, 256);
		h->t[i].soff = off;
		off = align(off + sbytes, 256);
		total += (uint64_t)rows * cols;
	}
	h->magic = NC_MAGIC;
	h->version = NC_VERSION;
	h->nlayer = L;
	h->nhead = H;
	h->nkv = KV;
	h->dim = D;
	h->ffn = F;
	h->vocab = VO;
	h->ntok = ntok;
	h->seqlen = mc ? jint(mc, "sequence_len", 2048) : 2048;
	if (h->seqlen > NC_MAXCTX)
		h->seqlen = NC_MAXCTX;
	h->qtype = qtype;
	h->bos = bos;
	h->tokoff = NC_PAGE;
	h->toklen = toklen;
	h->nten = nten;

	if (!(f = fopen(out, "wb")))
		FAIL("cannot write %s", out);
	if (fwrite(hdrb, 1, NC_PAGE, f) != NC_PAGE || fwrite(tokblob, 1, toklen, f) != toklen)
		FAIL("write failed (disk full?)");
	if (!(row = malloc(sizeof(float) * (D > F ? D : F))))
		FAIL("out of memory");
	for (i = 0; i < nten; i++) {
		NCTen *e = &h->t[i];
		int nb = e->cols / NC_BLK;

		t = src[i];
		snprintf(msg, sizeof msg, "%s %s", e->type == NC_Q8 ? "Q8" : "Q4", t->name);
		if (cb)
			cb(ctx, (double)done / total, msg);
		qbytes = (uint64_t)e->rows * e->cols / (e->type == NC_Q8 ? 1 : 2);
		sbytes = (uint64_t)e->rows * nb * 2;
		if (!(qb = malloc(qbytes)) || !(sb = malloc(sbytes)))
			FAIL("out of memory");
		for (r = 0; r < e->rows; r++) {
			getrow(t, r, row);
			if (e->type == NC_Q8)
				q8row(row, e->cols, (int8_t *)qb + r * e->cols, sb + r * nb);
			else
				q4row(row, e->cols, qb + r * e->cols / 2, sb + r * nb);
		}
		if (padto(f, e->qoff) || fwrite(qb, 1, qbytes, f) != qbytes ||
		    padto(f, e->soff) || fwrite(sb, 1, sbytes, f) != sbytes)
			FAIL("write failed (disk full?)");
		free(qb);
		free(sb);
		qb = NULL;
		sb = NULL;
		dropsrc(t);
		done += (uint64_t)e->rows * e->cols;
	}
	if (padto(f, align(off, NC_PAGE)) || fclose(f))
		FAIL("write failed (disk full?)");
	f = NULL;
	if (cb)
		cb(ctx, 1, "done");
	free(row);
	free(src);
	free(hdrb);
	free(pt);
	free(js);
	free(tokblob);
	free(z.e);
	vmfree(&vm);
	unmap(&m);
	return 0;
fail:
	if (f) {
		fclose(f);
		unlink(out);
	}
	free(qb);
	free(sb);
	free(row);
	free(src);
	free(hdrb);
	free(pt);
	free(js);
	free(tokblob);
	free(z.e);
	vmfree(&vm);
	unmap(&m);
	unmap(&mm);
	return -1;
}
