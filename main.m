/*
 * main.m - NanoChat: run the nanochat d34 base model on a Mac.
 *
 * Import a checkpoint folder (model_*.pt, meta_*.json, tokenizer.pkl)
 * to a quantized .ncq file, open it, type a prompt, press Cmd-Return.
 * A base model continues text: the reply is appended to the page.
 */
#import <Cocoa/Cocoa.h>

#include "nc.h"

@interface App : NSObject <NSApplicationDelegate>
@end

@interface App ()
- (void)progress:(double)f text:(NSString *)s;
@end

@implementation App {
	NSWindow *win;
	NSTextView *tv;
	NSTextField *status, *tempf, *topkf, *maxf;
	NSPopUpButton *quant, *engine;
	NSButton *gen, *imp, *opn;
	NSProgressIndicator *bar;
	NSDictionary *plain, *made;	/* typed and generated text */
	NCModel *model;
	NCGpu *gpu;
	NCCpu *cpu;
	NSString *info;
	int *cache, ncache;
	volatile int running, stop;
	int metal;			/* engine of the current run */
	uint64_t rng;
}

static NSButton *
button(NSString *title, id target, SEL action)
{
	NSButton *b = [NSButton buttonWithTitle:title target:target action:action];

	b.bezelStyle = NSBezelStyleRounded;
	return b;
}

static NSTextField *
label(NSString *s)
{
	NSTextField *l = [NSTextField labelWithString:s];

	l.textColor = NSColor.secondaryLabelColor;
	return l;
}

static NSTextField *
field(NSString *s, CGFloat w)
{
	NSTextField *f = [NSTextField textFieldWithString:s];

	f.alignment = NSTextAlignmentRight;
	[f.widthAnchor constraintEqualToConstant:w].active = YES;
	return f;
}

static NSPopUpButton *
popup(NSArray *items, id target, SEL action)
{
	NSPopUpButton *p = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];

	[p addItemsWithTitles:items];
	p.target = target;
	p.action = action;
	return p;
}

static NSMenuItem *
item(NSMenu *m, NSString *title, SEL action, NSString *key)
{
	return [m addItemWithTitle:title action:action keyEquivalent:key];
}

/* bytes at the start of p that make whole UTF-8 characters */
static NSUInteger
whole(const uint8_t *p, NSUInteger n)
{
	NSUInteger i = 0, k;

	while (i < n) {
		k = p[i] < 0x80 ? 1 : p[i] >> 5 == 6 ? 2 : p[i] >> 4 == 14 ? 3 :
		    p[i] >> 3 == 30 ? 4 : 1;
		if (i + k > n)
			break;
		i += k;
	}
	return i;
}

static NSString *
text(const void *p, NSUInteger n)
{
	NSString *s = [[NSString alloc] initWithBytes:p length:n encoding:NSUTF8StringEncoding];

	return s ? s : @"�";
}

- (void)menus
{
	NSMenu *mb = [NSMenu new], *m;
	NSMenuItem *it;

	m = [NSMenu new];
	item(m, @"About NanoChat", @selector(orderFrontStandardAboutPanel:), @"");
	[m addItem:NSMenuItem.separatorItem];
	item(m, @"Hide NanoChat", @selector(hide:), @"h");
	item(m, @"Quit NanoChat", @selector(terminate:), @"q");
	[mb addItemWithTitle:@"" action:nil keyEquivalent:@""].submenu = m;

	m = [[NSMenu alloc] initWithTitle:@"File"];
	item(m, @"Import Checkpoint…", @selector(importCheckpoint:), @"i");
	item(m, @"Open Model…", @selector(openModel:), @"o");
	[m addItem:NSMenuItem.separatorItem];
	item(m, @"Close Window", @selector(performClose:), @"w");
	[mb addItemWithTitle:@"" action:nil keyEquivalent:@""].submenu = m;

	m = [[NSMenu alloc] initWithTitle:@"Edit"];
	item(m, @"Undo", @selector(undo:), @"z");
	it = item(m, @"Redo", @selector(redo:), @"z");
	it.keyEquivalentModifierMask = NSEventModifierFlagCommand | NSEventModifierFlagShift;
	[m addItem:NSMenuItem.separatorItem];
	item(m, @"Cut", @selector(cut:), @"x");
	item(m, @"Copy", @selector(copy:), @"c");
	item(m, @"Paste", @selector(paste:), @"v");
	item(m, @"Select All", @selector(selectAll:), @"a");
	[mb addItemWithTitle:@"" action:nil keyEquivalent:@""].submenu = m;

	m = [[NSMenu alloc] initWithTitle:@"Model"];
	item(m, @"Generate / Stop", @selector(generate:), @"\r");
	item(m, @"Stop", @selector(stopGenerating:), @".");
	[m addItem:NSMenuItem.separatorItem];
	item(m, @"Clear Page", @selector(clearPage:), @"k");
	[mb addItemWithTitle:@"" action:nil keyEquivalent:@""].submenu = m;

	NSApp.mainMenu = mb;
}

- (void)window
{
	NSView *cv, *s1, *s2;
	NSScrollView *sv;
	NSStackView *top, *foot, *root;
	NSFont *font = [NSFont systemFontOfSize:15];

	win = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 980, 720)
	    styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
	    NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
	    backing:NSBackingStoreBuffered defer:NO];
	win.title = @"NanoChat";
	win.minSize = NSMakeSize(820, 360);
	cv = win.contentView;

	imp = button(@"Import…", self, @selector(importCheckpoint:));
	quant = popup(@[@"Q8", @"Q4"], nil, nil);
	opn = button(@"Open…", self, @selector(openModel:));
	engine = popup(@[@"Metal", @"CPU"], self, @selector(engineChanged:));
	tempf = field(@"0.8", 44);
	topkf = field(@"50", 44);
	maxf = field(@"256", 52);
	gen = button(@"Generate", self, @selector(generate:));
	gen.keyEquivalent = @"";

	top = [NSStackView new];
	top.orientation = NSUserInterfaceLayoutOrientationHorizontal;
	top.spacing = 8;
	top.edgeInsets = NSEdgeInsetsMake(10, 14, 10, 14);
	for (NSView *v in @[imp, quant, opn])
		[top addView:v inGravity:NSStackViewGravityLeading];
	for (NSView *v in @[engine, label(@"Temp"), tempf, label(@"Top-k"),
	    topkf, label(@"Tokens"), maxf, gen])
		[top addView:v inGravity:NSStackViewGravityTrailing];

	sv = [NSTextView scrollableTextView];
	tv = sv.documentView;
	tv.font = font;
	tv.richText = NO;
	tv.allowsUndo = YES;
	tv.automaticQuoteSubstitutionEnabled = NO;
	tv.automaticDashSubstitutionEnabled = NO;
	tv.automaticTextReplacementEnabled = NO;
	tv.automaticSpellingCorrectionEnabled = NO;
	tv.continuousSpellCheckingEnabled = NO;
	tv.textContainerInset = NSMakeSize(18, 16);
	plain = @{ NSFontAttributeName: font,
	    NSForegroundColorAttributeName: NSColor.labelColor };
	made = @{ NSFontAttributeName: font,
	    NSForegroundColorAttributeName: NSColor.systemBlueColor };
	tv.textColor = NSColor.labelColor;
	tv.typingAttributes = plain;
	tv.string = @"The capital of France is";
	[sv setContentHuggingPriority:1 forOrientation:NSLayoutConstraintOrientationVertical];

	status = label(@"Import a checkpoint folder, or open a .ncq model.");
	status.font = [NSFont systemFontOfSize:11];
	status.lineBreakMode = NSLineBreakByTruncatingTail;
	[status setContentCompressionResistancePriority:1
	    forOrientation:NSLayoutConstraintOrientationHorizontal];
	bar = [NSProgressIndicator new];
	bar.style = NSProgressIndicatorStyleBar;
	bar.indeterminate = NO;
	bar.minValue = 0;
	bar.maxValue = 1;
	bar.hidden = YES;
	[bar.widthAnchor constraintEqualToConstant:180].active = YES;
	foot = [NSStackView stackViewWithViews:@[status, bar]];
	foot.edgeInsets = NSEdgeInsetsMake(5, 14, 5, 14);

	s1 = [NSBox new];
	((NSBox *)s1).boxType = NSBoxSeparator;
	[s1.heightAnchor constraintEqualToConstant:1].active = YES;
	s2 = [NSBox new];
	((NSBox *)s2).boxType = NSBoxSeparator;
	[s2.heightAnchor constraintEqualToConstant:1].active = YES;
	root = [NSStackView stackViewWithViews:@[top, s1, sv, s2, foot]];
	root.orientation = NSUserInterfaceLayoutOrientationVertical;
	root.spacing = 0;
	root.distribution = NSStackViewDistributionFill;
	root.translatesAutoresizingMaskIntoConstraints = NO;
	[cv addSubview:root];
	[NSLayoutConstraint activateConstraints:@[
	    [root.topAnchor constraintEqualToAnchor:cv.topAnchor],
	    [root.bottomAnchor constraintEqualToAnchor:cv.bottomAnchor],
	    [root.leadingAnchor constraintEqualToAnchor:cv.leadingAnchor],
	    [root.trailingAnchor constraintEqualToAnchor:cv.trailingAnchor]]];
	for (NSView *v in root.arrangedSubviews)
		[v.widthAnchor constraintEqualToAnchor:root.widthAnchor].active = YES;

	if (![win setFrameUsingName:@"main"])
		[win center];
	win.frameAutosaveName = @"main";
	[win makeKeyAndOrderFront:nil];
	[win makeFirstResponder:tv];
	[tv moveToEndOfDocument:nil];
}

- (void)applicationDidFinishLaunching:(NSNotification *)n
{
	NSString *last = [NSUserDefaults.standardUserDefaults stringForKey:@"model"];

	rng = (uint64_t)arc4random() << 32 | arc4random() | 1;
	[self menus];
	[self window];
	[NSApp activate];
	if (last && [NSFileManager.defaultManager fileExistsAtPath:last])
		[self load:last quiet:YES];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)a
{
	return YES;
}

- (void)alert:(NSString *)title text:(NSString *)s
{
	NSAlert *a = [NSAlert new];

	a.messageText = title;
	a.informativeText = s ? s : @"";
	[a beginSheetModalForWindow:win completionHandler:nil];
}

- (BOOL)validateMenuItem:(NSMenuItem *)it
{
	SEL a = it.action;

	if (a == @selector(importCheckpoint:) || a == @selector(openModel:) ||
	    a == @selector(clearPage:))
		return !running;
	if (a == @selector(stopGenerating:))
		return running;
	return YES;
}

/* 0 idle, 1 generating, 2 importing */
- (void)busy:(int)b
{
	running = b != 0;
	imp.enabled = opn.enabled = quant.enabled = engine.enabled = b == 0;
	gen.enabled = b != 2;
	gen.title = b == 1 ? @"Stop" : @"Generate";
	tv.editable = b != 1;
	if (b == 0) {
		tv.typingAttributes = plain;
		[win makeFirstResponder:tv];
	}
}

/* ---- models ---- */

- (void)showInfo
{
	if (!model) {
		status.stringValue = @"Import a checkpoint folder, or open a .ncq model.";
		return;
	}
	info = [NSString stringWithFormat:@"%d layers · dim %d · %s · %.2f GB · %@",
	    model->nlayer, model->dim, model->qtype == NC_Q8 ? "Q8" : "Q4",
	    model->mapsz / 1e9, gpu && engine.indexOfSelectedItem == 0 ? @"Metal" : @"CPU"];
	status.stringValue = info;
}

- (BOOL)load:(NSString *)path quiet:(BOOL)quiet
{
	char err[512];
	NSString *e = nil;
	NCModel *mm;
	NCGpu *g;

	if (!(mm = nc_load(path.fileSystemRepresentation, err, sizeof err))) {
		if (!quiet)
			[self alert:@"Cannot open the model." text:@(err)];
		return NO;
	}
	g = [[NCGpu alloc] initWithModel:mm ctx:mm->seqlen error:&e];
	gpu = nil;
	ncc_free(cpu);
	cpu = NULL;
	nc_free(model);
	model = mm;
	gpu = g;
	free(cache);
	cache = calloc(model->seqlen + 1, sizeof *cache);
	ncache = 0;
	if (!gpu) {
		[engine selectItemAtIndex:1];
		[self alert:@"Metal is not available, the CPU will run the model." text:e];
	}
	win.title = [NSString stringWithFormat:@"NanoChat — %@", path.lastPathComponent];
	[NSUserDefaults.standardUserDefaults setObject:path forKey:@"model"];
	[self showInfo];
	return YES;
}

- (BOOL)application:(NSApplication *)a openFile:(NSString *)path
{
	return !running && [self load:path quiet:NO];
}

- (void)openModel:(id)sender
{
	NSOpenPanel *p = [NSOpenPanel openPanel];

	if (running)
		return;
	p.message = @"Choose a .ncq model file.";
	[p beginSheetModalForWindow:win completionHandler:^(NSModalResponse r) {
		if (r == NSModalResponseOK)
			[self load:p.URL.path quiet:NO];
	}];
}

- (void)engineChanged:(id)sender
{
	if (engine.indexOfSelectedItem == 0 && !gpu && model) {
		[engine selectItemAtIndex:1];
		NSBeep();
	}
	if (engine.indexOfSelectedItem == 0) {
		ncc_free(cpu);
		cpu = NULL;
	}
	ncache = 0;
	[self showInfo];
}

/* ---- import ---- */

static void
progress(void *ctx, double f, const char *msg)
{
	App *app = (__bridge App *)ctx;
	NSString *s = @(msg);

	dispatch_async(dispatch_get_main_queue(), ^{
		[app progress:f text:s];
	});
}

- (void)progress:(double)f text:(NSString *)s
{
	bar.doubleValue = f;
	status.stringValue = [NSString stringWithFormat:@"Importing: %@", s];
}

- (void)importCheckpoint:(id)sender
{
	NSOpenPanel *p = [NSOpenPanel openPanel];

	if (running)
		return;
	p.canChooseDirectories = YES;
	p.canChooseFiles = YES;
	p.prompt = @"Import";
	p.message = @"Choose the folder with model_*.pt, meta_*.json and tokenizer.pkl "
	    "(or the .pt file itself).";
	[p beginSheetModalForWindow:win completionHandler:^(NSModalResponse r) {
		if (r == NSModalResponseOK)
			dispatch_async(dispatch_get_main_queue(), ^{
				[self importFrom:p.URL];
			});
	}];
}

- (void)importFrom:(NSURL *)src
{
	NSSavePanel *s = [NSSavePanel savePanel];
	NSNumber *dir = nil;
	int qt = quant.indexOfSelectedItem == 0 ? NC_Q8 : NC_Q4;

	[src getResourceValue:&dir forKey:NSURLIsDirectoryKey error:nil];
	s.directoryURL = dir.boolValue ? src : src.URLByDeletingLastPathComponent;
	s.nameFieldStringValue = qt == NC_Q8 ? @"nanochat-d34-q8.ncq" : @"nanochat-d34-q4.ncq";
	s.message = @"Save the quantized model.";
	[s beginSheetModalForWindow:win completionHandler:^(NSModalResponse r) {
		NSString *in = src.path, *out = s.URL.path;

		if (r != NSModalResponseOK)
			return;
		[self busy:2];
		self->bar.doubleValue = 0;
		self->bar.hidden = NO;
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			char err[512] = "";
			double t = CFAbsoluteTimeGetCurrent();
			int rc = nc_import(in.fileSystemRepresentation, out.fileSystemRepresentation,
			    qt, progress, (__bridge void *)self, err, sizeof err);
			NSString *e = @(err);

			t = CFAbsoluteTimeGetCurrent() - t;
			dispatch_async(dispatch_get_main_queue(), ^{
				self->bar.hidden = YES;
				[self busy:0];
				if (rc) {
					[self showInfo];
					[self alert:@"Import failed." text:e];
				} else if ([self load:out quiet:NO])
					self->status.stringValue = [NSString stringWithFormat:
					    @"Imported in %.0f s · %@", t, self->info];
			});
		});
	}];
}

/* ---- generate ---- */

- (void)clearPage:(id)sender
{
	if (!running)
		tv.string = @"";
}

- (void)stopGenerating:(id)sender
{
	stop = 1;
}

- (void)emit:(NSString *)s
{
	NSAttributedString *a = [[NSAttributedString alloc] initWithString:s attributes:made];

	[tv.textStorage appendAttributedString:a];
	[tv scrollRangeToVisible:NSMakeRange(tv.string.length, 0)];
}

/* run toks[0..n) from pos; logits of the last one */
- (float *)eval:(const int *)toks count:(int)n pos:(int)pos
{
	float *lg = NULL;
	int i;

	if (metal)
		return [gpu run:toks count:n pos:pos];
	if (!cpu && !(cpu = ncc_new(model, model->seqlen)))
		return NULL;
	for (i = 0; i < n; i++)
		lg = ncc_step(cpu, toks[i], pos + i, i == n - 1);
	return lg;
}

- (void)generate:(id)sender
{
	NSData *d;
	float temp;
	int *toks, n, nctx, max, topk, keep;

	if (running) {
		stop = 1;
		return;
	}
	if (!gen.enabled)
		return;
	if (!model) {
		[self openModel:nil];
		return;
	}
	nctx = model->seqlen;
	d = [tv.string dataUsingEncoding:NSUTF8StringEncoding];
	toks = malloc(sizeof(int) * (d.length + 2));
	toks[0] = model->bos;
	n = 1 + nc_encode(model, d.bytes, d.length, toks + 1, (int)d.length + 1);
	temp = tempf.floatValue;
	topk = topkf.intValue;
	max = maxf.intValue > 0 ? maxf.intValue : 256;
	if (n + max > nctx) {
		/* keep the end of the page: bos + last tokens */
		keep = nctx - max > nctx / 2 ? nctx - max : nctx / 2;
		if (n > keep) {
			memmove(toks + 1, toks + n - keep + 1, sizeof(int) * (keep - 1));
			n = keep;
		}
	}
	[tv moveToEndOfDocument:nil];
	metal = gpu && engine.indexOfSelectedItem == 0;
	[self busy:1];
	stop = 0;
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		[self run:toks count:n max:max temp:temp topk:topk];
		free(toks);
	});
}

- (void)run:(int *)toks count:(int)n max:(int)max temp:(float)temp topk:(int)topk
{
	NSMutableData *pend = [NSMutableData data];
	NSString *why = @"", *rest, *s;
	double t0, t1;
	float *lg;
	int same = 0, t, np = n, nctx = model->seqlen, ngen = 0;
	NSUInteger k;

	/* reuse the cache for the part of the page that did not change */
	while (same < ncache && same < n && cache[same] == toks[same])
		same++;
	if (same == n)
		same--;
	t0 = CFAbsoluteTimeGetCurrent();
	lg = [self eval:toks + same count:n - same pos:same];
	memcpy(cache + same, toks + same, sizeof(int) * (n - same));
	ncache = lg ? n : 0;
	t1 = CFAbsoluteTimeGetCurrent();
	while (lg) {
		t = nc_sample(model, lg, temp, topk, &rng);
		if (model->special[t]) {
			why = @" · end of text";
			break;
		}
		[pend appendBytes:model->tb[t] length:model->tl[t]];
		if ((k = whole(pend.bytes, pend.length)) > 0) {
			s = text(pend.bytes, k);
			[pend replaceBytesInRange:NSMakeRange(0, k) withBytes:NULL length:0];
			dispatch_async(dispatch_get_main_queue(), ^{
				[self emit:s];
			});
		}
		if (++ngen >= max || stop)
			break;
		if (n >= nctx) {
			why = @" · context full";
			break;
		}
		lg = [self eval:&t count:1 pos:n];
		cache[n++] = t;
		ncache = lg ? n : 0;
		if (!lg)
			why = @" · engine error";
		if (ngen % 4 == 0) {
			s = [NSString stringWithFormat:@"%d tokens · %.1f tok/s · context %d/%d",
			    ngen, ngen / (CFAbsoluteTimeGetCurrent() - t1), n, nctx];
			dispatch_async(dispatch_get_main_queue(), ^{
				self->status.stringValue = s;
			});
		}
	}
	if (!lg && ngen == 0)
		why = @" · engine error";
	rest = pend.length ? text(pend.bytes, pend.length) : nil;
	s = [NSString stringWithFormat:@"prompt %d tok (%d new) in %.2f s · "
	    "%d tokens · %.1f tok/s%@", np, np - same, t1 - t0, ngen,
	    ngen / fmax(CFAbsoluteTimeGetCurrent() - t1, 1e-6), why];
	dispatch_async(dispatch_get_main_queue(), ^{
		if (rest)
			[self emit:rest];
		self->status.stringValue = [NSString stringWithFormat:@"%@ · %@",
		    s, self->info];
		[self busy:0];
	});
}

@end

int
main(int argc, const char **argv)
{
	@autoreleasepool {
		NSApplication *app = NSApplication.sharedApplication;
		App *d = [App new];

		app.activationPolicy = NSApplicationActivationPolicyRegular;
		app.delegate = d;
		[app run];
	}
	return 0;
}
