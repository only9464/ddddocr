package digits_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/only9464/ddddocr-go/digits"
)

// TestRecognizeSamples recognizes the sample images saved at export time; the
// expected texts come from the same onnxruntime run.
//
// TestRecognizeSamples 端到端识别导出时留下的样图, 期望值与 onnxruntime 一致。
func TestRecognizeSamples(t *testing.T) {
	metaRaw, err := os.ReadFile(filepath.Join("testdata", "refs.json"))
	if err != nil {
		t.Skipf("no reference data: %v", err)
	}
	var meta struct {
		Samples []struct {
			File string `json:"file"`
			Text string `json:"text"`
		} `json:"samples"`
	}
	if err := json.Unmarshal(metaRaw, &meta); err != nil {
		t.Fatalf("cannot parse refs.json: %v", err)
	}
	engine, err := digits.New()
	if err != nil {
		t.Skipf("the weights were never exported (backbone/model/backbone.bin + digits/model/head.bin): %v", err)
	}
	for _, s := range meta.Samples {
		path := filepath.Join("testdata", s.File)
		text, err := engine.RecognizeFile(path)
		if err != nil {
			t.Fatalf("%s: %v", s.File, err)
		}
		if text != s.Text {
			t.Fatalf("%s: got %q, want %q", s.File, text, s.Text)
		}
	}
}

// BenchmarkRecognize measures recognizing one image (a 4-digit captcha by
// default).
//
// BenchmarkRecognize 测单张图的识别耗时(默认 4 位数字验证码)。
func BenchmarkRecognize(b *testing.B) {
	engine, err := digits.New()
	if err != nil {
		b.Skipf("the weights were never exported: %v", err)
	}
	data, err := os.ReadFile(filepath.Join("testdata", "captcha0.jpg"))
	if err != nil {
		b.Skipf("no sample image: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := engine.Recognize(data); err != nil {
			b.Fatalf("recognition failed: %v", err)
		}
	}
}
