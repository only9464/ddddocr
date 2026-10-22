package ddddocr

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// refSample is one sample from refs.json (only the fields these tests use).
//
// refSample 是 refs.json 里的一个样本(只取本测试用到的字段)。
type refSample struct {
	Name        string `json:"name"`
	File        string `json:"file"`
	Height      int    `json:"height"`
	Width       int    `json:"width"`
	InputOffset int    `json:"input_offset"`
	InputFloats int    `json:"input_floats"`
	Indices     []int  `json:"indices"`
	Text        string `json:"text"`
	FullText    string `json:"full_text"`
}

// refMeta is the structure tools/export.py writes into refs.json.
//
// refMeta 是 tools/export.py 写出的 refs.json 结构。
type refMeta struct {
	Charset []string    `json:"charset"`
	Height  int         `json:"height"`
	Samples []refSample `json:"samples"`
}

// loadFixture reads the backbone, the digits head and the digits reference data,
// skipping the test when the weights were never exported.
//
// loadFixture 读 backbone + digits 头 + digits 参考数据, 没导出权重时跳过。
func loadFixture(tb testing.TB) (*Recognizer, refMeta, []byte) {
	tb.Helper()
	backboneBytes, err := os.ReadFile(filepath.Join("backbone", "model", "backbone.bin"))
	if err != nil {
		tb.Skipf("the backbone was never exported (backbone/model/backbone.bin): %v", err)
	}
	head, err := os.ReadFile(filepath.Join("digits", "model", "head.bin"))
	if err != nil {
		tb.Skipf("the digits head was never exported (digits/model/head.bin): %v", err)
	}
	metaRaw, err := os.ReadFile(filepath.Join("digits", "testdata", "refs.json"))
	if err != nil {
		tb.Skipf("no reference data (digits/testdata/refs.json): %v", err)
	}
	var meta refMeta
	if err := json.Unmarshal(metaRaw, &meta); err != nil {
		tb.Fatalf("cannot parse refs.json: %v", err)
	}
	inputs, err := os.ReadFile(filepath.Join("digits", "testdata", "inputs.f32"))
	if err != nil {
		tb.Skipf("no reference input (digits/testdata/inputs.f32): %v", err)
	}
	engine, err := NewFromSet(backboneBytes, head)
	if err != nil {
		tb.Fatalf("cannot load the weights: %v", err)
	}
	return engine, meta, inputs
}

// referenceInput rebuilds the preprocessed input of one sample, i.e. the tensor
// the model actually consumed.
//
// referenceInput 还原 refs.json 里某个样本的预处理输入(模型实际吃的张量)。
func referenceInput(tb testing.TB, sample refSample, inputs []byte) tensor3 {
	tb.Helper()
	start, end := sample.InputOffset, sample.InputOffset+sample.InputFloats*4
	if start < 0 || end > len(inputs) {
		tb.Fatalf("%s: reference input is out of range", sample.Name)
	}
	if sample.InputFloats != sample.Height*sample.Width {
		tb.Fatalf("%s: input has %d floats, want %dx%d",
			sample.Name, sample.InputFloats, sample.Height, sample.Width)
	}
	flat := make([]float32, sample.InputFloats)
	for i := range flat {
		flat[i] = math.Float32frombits(binary.LittleEndian.Uint32(inputs[start+i*4:]))
	}
	return tensor3{c: 1, h: sample.Height, w: sample.Width, data: flat}
}

// readSampleImage returns the encoded bytes of one sample image.
//
// readSampleImage 读取某个样本图片的原始字节。
func readSampleImage(tb testing.TB, sample refSample) []byte {
	tb.Helper()
	data, err := os.ReadFile(filepath.Join("digits", "testdata", sample.File))
	if err != nil {
		tb.Fatalf("cannot read %s: %v", sample.File, err)
	}
	return data
}

// TestRecognizeMatchesReference compares the per-step indices against the
// preprocessed input saved at export time, proving the Go forward pass matches
// onnxruntime exactly.
//
// TestRecognizeMatchesReference 用导出时的预处理输入逐下标对比模型输出, 保证 Go 侧的前向与 onnxruntime 完全一致。
func TestRecognizeMatchesReference(t *testing.T) {
	engine, meta, inputs := loadFixture(t)
	for _, sample := range meta.Samples {
		in := referenceInput(t, sample, inputs)
		got, err := engine.m.recognize(in)
		if err != nil {
			t.Fatalf("%s: forward pass failed: %v", sample.Name, err)
		}
		if len(got) != len(sample.Indices) {
			t.Fatalf("%s: got %d time steps, want %d", sample.Name, len(got), len(sample.Indices))
		}
		for i := range got {
			if got[i] != sample.Indices[i] {
				t.Fatalf("%s: index at step %d is %d, want %d", sample.Name, i, got[i], sample.Indices[i])
			}
		}
	}
}

// TestRecognizeSampleImages runs the whole "JPEG bytes -> preprocess -> forward
// -> CTC" chain.
//
// TestRecognizeSampleImages 跑完整的「JPEG 字节 → 预处理 → 前向 → CTC」链路。
func TestRecognizeSampleImages(t *testing.T) {
	engine, meta, _ := loadFixture(t)
	for _, sample := range meta.Samples {
		got, err := engine.Recognize(readSampleImage(t, sample))
		if err != nil {
			t.Fatalf("%s: recognition failed: %v", sample.Name, err)
		}
		if got != sample.Text {
			t.Fatalf("%s: got %q, want %q", sample.Name, got, sample.Text)
		}
	}
}

// TestRecognizeImageFormats covers the image decoders: the standard library only
// handles JPEG/PNG/GIF, BMP/TIFF/WebP are registered by the golang.org/x/image
// sub-packages.
//
// TestRecognizeImageFormats 覆盖各种图片解码器: 标准库只认 JPEG/PNG/GIF,
// BMP/TIFF/WebP 由 golang.org/x/image 的子包注册。
func TestRecognizeImageFormats(t *testing.T) {
	engine, meta, _ := loadFixture(t)
	if len(meta.Samples) == 0 {
		t.Skip("no reference samples")
	}
	sample := meta.Samples[0]
	base := strings.TrimSuffix(sample.File, filepath.Ext(sample.File))
	for _, ext := range []string{".jpg", ".png", ".bmp", ".tiff", ".webp"} {
		name := filepath.Join("digits", "testdata", base+ext)
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatalf("cannot read %s: %v", name, err)
		}
		got, err := engine.Recognize(data)
		if err != nil {
			t.Fatalf("%s: recognition failed: %v", name, err)
		}
		if got != sample.Text {
			t.Fatalf("%s: got %q, want %q", name, got, sample.Text)
		}
	}
}

// TestClassification checks that the Python-style method name Classification is
// equivalent to Recognize.
//
// TestClassification 校验 Python 风格的方法名 Classification 与 Recognize 等价。
func TestClassification(t *testing.T) {
	engine, meta, _ := loadFixture(t)
	for _, sample := range meta.Samples {
		got, err := engine.Classification(readSampleImage(t, sample))
		if err != nil {
			t.Fatalf("%s: recognition failed: %v", sample.Name, err)
		}
		if got != sample.Text {
			t.Fatalf("%s: got %q, want %q", sample.Name, got, sample.Text)
		}
	}
}

// TestClassificationProb checks recognition with confidence: the text matches
// Recognize, every character carries a probability in (0,1], and joining the
// characters gives back the text.
//
// TestClassificationProb 校验带置信度的识别: 文本和 Recognize 一致, 每个字符都有 (0,1] 的概率, 且逐字拼起来就是文本。
func TestClassificationProb(t *testing.T) {
	engine, meta, _ := loadFixture(t)
	for _, sample := range meta.Samples {
		text, chars, err := engine.ClassificationProb(readSampleImage(t, sample))
		if err != nil {
			t.Fatalf("%s: recognition failed: %v", sample.Name, err)
		}
		if text != sample.Text {
			t.Fatalf("%s: got %q, want %q", sample.Name, text, sample.Text)
		}
		if len(chars) != len([]rune(text)) {
			t.Fatalf("%s: text has %d characters but %d confidences were returned",
				sample.Name, len([]rune(text)), len(chars))
		}
		var joined strings.Builder
		for _, cp := range chars {
			if cp.Prob <= 0 || cp.Prob > 1 {
				t.Fatalf("%s: probability %v of %q is outside (0,1]", sample.Name, cp.Prob, cp.Char)
			}
			joined.WriteString(cp.Char)
		}
		if joined.String() != text {
			t.Fatalf("%s: joined characters give %q but the text is %q", sample.Name, joined.String(), text)
		}
	}
}

// TestArgmaxProb checks that argmaxProb returns the softmax probability of the
// largest entry.
//
// TestArgmaxProb 校验 argmaxProb 算的是 softmax 里最大那一项的概率。
func TestArgmaxProb(t *testing.T) {
	cases := []struct {
		row  []float32
		idx  int
		want float64
	}{
		{[]float32{0, 0, 0}, 0, 1.0 / 3},
		{[]float32{1, 2, 3}, 2, 1 / (1 + math.Exp(-1) + math.Exp(-2))},
		{[]float32{-5}, 0, 1},
	}
	for _, c := range cases {
		idx, prob := argmaxProb(c.row)
		if idx != c.idx {
			t.Fatalf("%v: got index %d, want %d", c.row, idx, c.idx)
		}
		if math.Abs(float64(prob)-c.want) > 1e-6 {
			t.Fatalf("%v: got probability %v, want %v", c.row, prob, c.want)
		}
	}
}

// TestRecognizeConcurrent checks that one recognizer may be called from several
// goroutines.
//
// TestRecognizeConcurrent 校验同一个识别器可以被并发调用。
func TestRecognizeConcurrent(t *testing.T) {
	engine, meta, _ := loadFixture(t)
	if len(meta.Samples) == 0 {
		t.Skip("no reference samples")
	}
	sample := meta.Samples[0]
	img := readSampleImage(t, sample)
	const n = 8
	var wg sync.WaitGroup
	errs := make([]error, n)
	got := make([]string, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			got[i], errs[i] = engine.Recognize(img)
		}(i)
	}
	wg.Wait()
	for i := 0; i < n; i++ {
		if errs[i] != nil {
			t.Fatalf("goroutine %d: %v", i, errs[i])
		}
		if got[i] != sample.Text {
			t.Fatalf("goroutine %d: got %q, want %q", i, got[i], sample.Text)
		}
	}
}

// TestCharsetAndHeight checks the charset and height reported by the loaded
// model against the exported metadata.
//
// TestCharsetAndHeight 校验加载后的字符集和高度与导出时一致。
func TestCharsetAndHeight(t *testing.T) {
	engine, meta, _ := loadFixture(t)
	if engine.Height() != meta.Height {
		t.Fatalf("Height() = %d, want %d", engine.Height(), meta.Height)
	}
	if got := len(engine.Charset()); got != len(meta.Charset) {
		t.Fatalf("len(Charset()) = %d, want %d", got, len(meta.Charset))
	}
	if got := engine.Charset()[0]; got != "" {
		t.Fatalf("Charset()[0] = %q, want the CTC blank (an empty string)", got)
	}
}

// TestBackboneWithoutHead checks that a backbone without a classification head
// reports ErrNoHead instead of crashing on nil weights.
//
// TestBackboneWithoutHead 校验只有主干(没有分类头) 时会明确报 ErrNoHead,
// 而不是崩在 nil 权重上。
func TestBackboneWithoutHead(t *testing.T) {
	backboneBytes, err := os.ReadFile(filepath.Join("backbone", "model", "backbone.bin"))
	if err != nil {
		t.Skipf("the backbone was never exported: %v", err)
	}
	if _, err := New(backboneBytes); err != ErrNoHead {
		t.Fatalf("New(backbone) = %v, want ErrNoHead", err)
	}
}

// TestNewRejectsBadWeights checks that malformed weights are rejected.
//
// TestNewRejectsBadWeights 校验非法权重会被拒绝。
func TestNewRejectsBadWeights(t *testing.T) {
	if _, err := New(nil); err == nil {
		t.Fatal("empty weights must be rejected")
	}
	if _, err := New([]byte("not-a-model")); err == nil {
		t.Fatal("a wrong magic number must be rejected")
	}
	if _, err := NewFromSet(nil, nil); err == nil {
		t.Fatal("an empty backbone and head must be rejected")
	}
	if _, err := NewFromSet([]byte("not-a-model"), []byte("neither")); err == nil {
		t.Fatal("a garbage backbone must be rejected")
	}
}

// TestRecognizeBadImage checks that non-image bytes produce a decode error
// rather than a panic.
//
// TestRecognizeBadImage 校验非图片字节给出的是解码错误而不是 panic。
func TestRecognizeBadImage(t *testing.T) {
	engine, _, _ := loadFixture(t)
	if _, err := engine.Recognize([]byte("not an image")); err == nil {
		t.Fatal("non-image data must be rejected")
	}
}

// BenchmarkRecognize measures the cost of recognizing one image.
//
// BenchmarkRecognize 测单张图的识别耗时。
func BenchmarkRecognize(b *testing.B) {
	engine, meta, _ := loadFixture(b)
	if len(meta.Samples) == 0 {
		b.Skip("no reference samples")
	}
	img := readSampleImage(b, meta.Samples[0])
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := engine.Recognize(img); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkRecognizeParallel measures throughput with many goroutines on the
// same recognizer.
//
// BenchmarkRecognizeParallel 测同一识别器在多 goroutine 下的吞吐。
func BenchmarkRecognizeParallel(b *testing.B) {
	engine, meta, _ := loadFixture(b)
	if len(meta.Samples) == 0 {
		b.Skip("no reference samples")
	}
	img := readSampleImage(b, meta.Samples[0])
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, err := engine.Recognize(img); err != nil {
				b.Fatal(err)
			}
		}
	})
}
