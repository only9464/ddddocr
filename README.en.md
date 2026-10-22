# ddddocr-go

> **Disclaimer: all code and documentation in this repository were written by deepseek v4.1 flash and
> uploaded to the GitHub repository. If anything here infringes your rights, contact the owner and it
> will be removed.**

[中文文档](README.md) | English

A **pure Go** re-implementation of the recognition model (`common.onnx`) from
[ddddocr](https://github.com/sml2h3/ddddocr), with the weights **embedded into the binary at compile
time**.

- no cgo, no onnxruntime shared library, no Python
- the only third-party dependency is the pure-Go [`golang.org/x/image`](https://pkg.go.dev/golang.org/x/image),
  and only to decode the image formats the standard library lacks (BMP/TIFF/WebP); about 0.3 MB
- nothing to load at runtime: cross-compiled binaries, distroless and scratch images just work
- the weights are split into a "20.4 MB backbone + a tiny classification head", and **only the charset
  you import gets embedded** — a digits-only project never carries the 33.7 MB full charset

```go
import "github.com/only9464/ddddocr-go/digits"

engine, err := digits.New()                        // 20.4 MB backbone + 45 KB digits head, already embedded
if err != nil {
    log.Fatal(err)
}
text, err := engine.RecognizeFile("captcha.jpg")   // "4003"
```

## Size: pay only for what you embed

The convolution + BiLSTM part of the model (the **backbone**) is about 20.4 MB and is identical for
every charset; only the final Gemm classification head (`1024 × number of classes` float32 values)
depends on the charset. That is why the weights ship as two pieces.

The binary sizes below are measured (windows/amd64, Go 1.26, an empty program that imports one charset
package and calls `New()`; an empty `main` is 1.8 MB, of which roughly 0.3 MB is the BMP/TIFF/WebP
decoders):

| package | charset | head size | resulting binary |
| --- | --- | --- | --- |
| `backbone` | — (the shared part, imported by the three packages below) | 20.4 MB | — |
| `digits` | blank + `0-9`, 11 classes | 45 KB | 24.4 MB |
| `alnum` | blank + `0-9A-Za-z`, 63 classes | 259 KB | 24.6 MB |
| `full` | ddddocr `CHARSET_BETA`, 8210 classes | 33.7 MB | 58.0 MB |

Go's linker only keeps **packages and data that are actually referenced**, so switching charsets is a
one-line import change:

```go
import "github.com/only9464/ddddocr-go/digits"   // embeds the backbone and the digits head only
```

With just `digits` imported, the `alnum` and `full` heads are never linked in — no build tags, no
manual trimming. (Note that a package imported as `import _ "..."` and never used through `New()` has
its weight data dropped entirely by the linker.)

For a custom charset (say "0-9 plus a few CJK characters"), export a custom head and combine it with
the backbone:

```bash
python tools/export.py --charset custom:0123456789加油华为 --head-out ./mymodel/head.bin
```

```go
import (
    ddddocr "github.com/only9464/ddddocr-go"
    "github.com/only9464/ddddocr-go/backbone"
)

head, _ := os.ReadFile("mymodel/head.bin")
engine, err := ddddocr.NewFromSet(must(backbone.Bytes()), head)
```

## Project layout

```
ddddocr-go/
├── the ddddocr engine (this package)
│   ├── recognizer.go    public API: New / NewFromSet / Recognize* / Classification* / Charset / Height
│   ├── weights.go       weight container parser (magic + JSON header + float32 data), attach a head
│   ├── forward.go       forward operators: conv / swish / add / to_seq / lstm / gemm / argmax
│   └── preprocess.go    decode (JPEG/PNG/GIF/BMP/TIFF/WebP) -> Lanczos -> gray -> [0,1]
├── backbone/            embeds the shared 20.4 MB backbone
├── digits/ alnum/ full/ embed their own classification head (same template, logic in internal/charset)
├── internal/charset/    shared charset-package logic: Once cache + backbone/head composition
├── internal/modelbin/   tiny helper that reads a weight file out of an embed.FS
├── examples/recognize/  command line example
├── tools/export.py      ONNX -> weight container exporter (development only, never shipped)
├── .gitattributes       force LF line endings; weights and images are marked binary
└── */model/, */testdata/ weights and the reference data used by the tests
```

## Why not `onnxruntime_go`

The existing Go ports on GitHub (`yangbin1322/go-ddddocr`, `FeilongTest/go-ddddocr`,
`Changbaiqi/ddddocr-go`, `KazemiyaMione/ddddocr-go`, `tensafe/goddddocr`) all wrap
`github.com/yalue/onnxruntime_go`, so they are still **cgo + a runtime-loaded `onnxruntime.dll`** plus a
54 MB `.onnx` file. This project "compiles" the ONNX graph offline into a linear forward program
(conv / swish / add / to_seq / lstm / gemm / argmax) and implements just those operators in Go:

| | this project | onnxruntime_go based |
| --- | --- | --- |
| cgo | not needed | required |
| runtime dependency | none | the onnxruntime shared library |
| third-party code | only `golang.org/x/image` (pure Go, image formats) | cgo wrapper + shared library |
| model files | compiled into the binary | ship the `.onnx` next to the program |
| cross compiling | just set `GOOS`/`GOARCH` | need a dll/so/dylib per platform |
| size | 24.4 MB (digits) / 58.0 MB (full) | ~20 MB library + 54 MB model |

## Install

```bash
go get github.com/only9464/ddddocr-go/digits   # digits (recommended)
go get github.com/only9464/ddddocr-go/alnum    # digits + letters
go get github.com/only9464/ddddocr-go/full     # all 8210 characters
```

The weights are build artifacts that may or may not be committed (`backbone.bin` 20.4 MB, heads 45 KB /
259 KB / 33.7 MB; all well below GitHub's 100 MB per-file limit, so no LFS is needed). Without them
`go build` still succeeds and the calls return an error at run time — export the weights first, see the
next section.

`go get` pulls in the single dependency `golang.org/x/image` (pure Go, no cgo).

## Re-exporting the weights

```bash
# onnx and ddddocr are needed to parse the model; onnxruntime is only used to cross-check against
# the reference implementation
pip install ddddocr onnx pillow numpy onnxruntime

# the backbone only has to be exported once (shared by every charset)
python tools/export.py --charset digits
#   -> backbone/model/backbone.bin  (20.4 MB)
#   -> digits/model/head.bin        (45 KB)
#   -> digits/testdata/             (reference data for go test)

python tools/export.py --charset alnum         # -> alnum/model/head.bin (259 KB)
python tools/export.py --charset full          # -> full/model/head.bin (33.7 MB)

# bring your own sample captchas (the script cross-checks them with onnxruntime)
python tools/export.py --charset digits --samples "path/to/captcha*.jpg"

# single-file mode: backbone and head in one blob, twice the size, kept for compatibility
python tools/export.py --charset digits --single-out /tmp/digits.bin
```

What the script does: parse the ONNX graph, extract the forward program, recompute every step with
numpy and **compare all time steps against onnxruntime** (any mismatch aborts with a failure), trim the
classification head according to `--charset`, then write the backbone, the head and the `testdata/`
reference data. Without sample images (or with `--no-verify`) the cross-check is skipped — and
onnxruntime is not needed — so only the weights are written. `python tools/export.py --help` lists every
option.

Keep `--single-out` / `--head-out` outside `*/model/`: those directories are embedded wholesale by
`//go:embed model`, so one extra file (a 54 MB single-file blob, say) makes the binary bigger for no
reason.

## Usage

### Coming from the Python package: barely anything changes

The usual Python code is "construct once, call `classification` once"; in Go it looks like this:

```python
# Python
import ddddocr

ocr = ddddocr.DdddOcr(beta=True)                     # the 8210-character beta model
text = ocr.classification(open("captcha.jpg", "rb").read())
```

```go
// Go
engine, err := full.New()                            // the equivalent of the beta model
if err != nil {
    log.Fatal(err)
}
text, err := engine.Classification(jpegBytes)         // exactly the same as Recognize
```

How the pieces map:

| Python | Go | notes |
| --- | --- | --- |
| `ddddocr.DdddOcr(beta=True)` | `full.New()` | both are `common.onnx` + `CHARSET_BETA` (8210 classes); weights and charset match one to one |
| `ddddocr.DdddOcr()` (default) | — | the default is `common_old.onnx` (the old charset), which this project does not export; export it yourself with `--model common_old.onnx` if you need it |
| `ocr.classification(bytes)` | `engine.Classification(bytes)` or `engine.Recognize(bytes)` | the alias calls Recognize, so the behavior is identical |
| `ocr.classification(path)` | `engine.RecognizeFile(path)` | Python also accepts a path string; Go has a separate method |
| `ocr.classification(PIL.Image)` | `engine.RecognizeImage(img)` | the parameter type is `image.Image` |
| `ocr.classification(bytes, probability=True)` | `engine.ClassificationProb(bytes)` | returns the text plus a confidence per character; it does not return the full time-step × charset matrix |
| `ocr.set_ranges("0123456789")` | `digits.New()` | trimming the head at export time is faster and smaller than filtering at run time |
| `ocr.get_charset()` | `engine.Charset()` | index 0 is the CTC blank |
| `show_ad` / `use_gpu` / `switch_device` | — | a pure Go library has neither ads nor a GPU |
| `png_fix=True` / `color_filter_*` | — | not implemented; for transparent PNGs composite a white background with `image/draw` first |
| `ocr.detection()` / `slide_match()` | — | object detection and slider matching were not ported |

Same idiom as Python: **construct once and reuse**. The only difference is that in Go both construction
and recognition return an `error` (Python raises), so handle it with `if err != nil`.

### Digits

```go
import "github.com/only9464/ddddocr-go/digits"

engine, err := digits.New()
text, err := engine.RecognizeFile("captcha.jpg")     // local file
text, err := engine.Recognize(jpegBytes)             // []byte (JPEG/PNG/GIF/BMP/TIFF/WebP)
text, err := engine.RecognizeImage(img)              // image.Image
```

### Digits + letters, or the full charset

```go
import "github.com/only9464/ddddocr-go/alnum"

engine, err := alnum.New()
text, err := engine.Recognize(data)
```

### Bring your own weights (custom head or single-file blob)

```go
raw, _ := os.ReadFile("my-weights.bin")
engine, err := ddddocr.New(raw)      // single-file weights (--single-out output)
```

The format is the container the exporter writes: 8-byte magic `OCRW1\0\0\0`, a uint32 header length, a
JSON header and the float32 data.

### Low-level API

```go
idx, _ := engine.RecognizeIndices(data)  // class index per time step (before CTC deduplication)
text, chars, _ := engine.ClassificationProb(data)
                                         // chars: []ddddocr.CharProb{{Char:"4", Prob:0.99}, ...}
engine.Charset()                         // head charset, index 0 is blank
engine.Height()                          // input height (64 for the common models)
```

A `Recognizer` is read-only after loading: the weights are never written and every call allocates its
own scratch buffers, so **one instance may be shared by any number of goroutines** with no locking.

Measured on an i5-12500H (12 cores / 16 threads, windows/amd64, a 70×30 four-digit captcha, preprocessing
included):

| scenario | time |
| --- | --- |
| one image, all cores (the forward pass parallelizes over output channels) | ~91 ms |
| one image, `GOMAXPROCS=1` (single core) | ~255 ms |
| average throughput with 16 concurrent calls | ~41 ms per image |
| allocations per call | ~6.9 MB / 5.3k allocations |

## Tests

```bash
go test ./...                    # tests that need the weights skip when they are missing
go test -race ./...              # includes the concurrency test (a lock-free shared recognizer)
go test -bench=. ./digits        # single-image benchmark
```

- `TestRecognizeMatchesReference`: feeds the preprocessing input saved at export time through the Go
  forward pass and compares every time-step index with onnxruntime (a wrong operator shows up
  immediately).
- `TestPreprocessMatchesReference`: aligns Go's "decode + Lanczos + gray" with the PIL-produced tensor,
  tolerance 4/255 (measured deviation ≤ 2/255), so preprocessing is not merely "looks right".
- `TestRecognizeImageFormats`: the same captcha saved as jpg/png/bmp/tiff/webp must give the same
  result (this also guards the BMP/TIFF/WebP decoder registration).
- `TestRecognizeSampleImages`: end-to-end run over the sample images, expectations come from
  `refs.json` and are not hardcoded.
- `TestClassification*`: the Python-style `Classification` alias and the `ClassificationProb` variant
  agree with `Recognize` (per-character probabilities in (0,1], and joining the characters gives back
  the text).
- `TestRecognizeConcurrent`: several goroutines recognizing with one instance, meant for `-race`.

The repo ships a `.gitattributes` (`* text=auto eol=lf` plus `binary` for the weights), so there is no
`core.autocrlf` fiddling on Windows; CI also checks `gofmt`.

## Known limitations

- Only the **recognition** model was ported; ddddocr's object detection (`common_det.onnx`) and slider
  matching were not.
- The Python post-processing switches `png_fix` / `color_filter_*` / `set_ranges` are not implemented:
  restricting the charset is done by importing the matching head (`digits` / `alnum` / `--charset
  custom`), and transparent backgrounds must be composited by the caller.
- The head is ddddocr's `CHARSET_BETA` (8210 classes), not `CHARSET_OLD`; the wrong charset decodes to
  garbage.
- Only JPEG/PNG/GIF/BMP/TIFF/WebP are recognized; other formats (ICO, JPEG 2000, ...) can be added with
  `image.RegisterFormat`, because `Recognize` goes through the standard `image.Decode`.
- Plain scalar float32, no SIMD: for the same image Python + onnxruntime takes about 6 ms while this
  project takes about 255 ms on one core. The convolution is already ordered so that each weight is
  read once and the output row stays in L1, but Go does not auto-vectorize; going further means cgo or
  assembly, which would kill the "dependency-free cross compiling" selling point.
- Weights are float32. They could be quantized to int8 (~1/4 the size), but the backbone and the head
  would have to be quantized together.
- Batch work is just a goroutine per image (the instance is concurrency-safe), no extra lock needed.

## Publishing to your own repository / fork

```bash
sed -i 's#github.com/only9464/ddddocr-go#github.com/<your-name>/ddddocr-go#' \
    go.mod $(grep -rl 'github.com/only9464/ddddocr-go' --include='*.go' .)
git init && git add . && git commit -m "ddddocr-go: pure Go ddddocr inference"
git push -u origin main && git tag v0.1.0 && git push origin v0.1.0
```

Then from another project: `go get github.com/<your-name>/ddddocr-go/digits@v0.1.0`.

## License

The code in this project is MIT licensed. The model weights and charsets come from
[sml2h3/ddddocr](https://github.com/sml2h3/ddddocr) (MIT); see [NOTICE](NOTICE).
