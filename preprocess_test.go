package ddddocr

import (
	"image"
	"image/color"
	"math"
	"testing"
)

// TestPreprocessMatchesReference checks that the Go preprocessing matches the
// tensor ddddocr (PIL) fed to the model at export time. A difference of about
// one uint8 is tolerated: PIL normalizes its filter taps with fixed-point
// integers, while this code uses float32, which leaves half an LSB of jitter.
//
// TestPreprocessMatchesReference 校验 Go 的预处理与 ddddocr(PIL) 导出时喂给模型的张量一致。允许 ±1 个 uint8 量级的差: PIL 的卷积核系数用定点整数归一化, 与
// float32 归一化会有半个 LSB 的出入。
func TestPreprocessMatchesReference(t *testing.T) {
	engine, meta, inputs := loadFixture(t)
	const tol = 4.0 / 255 // measured maximum is 2/255
	for _, sample := range meta.Samples {
		want := referenceInput(t, sample, inputs)
		img, err := decodeImage(readSampleImage(t, sample))
		if err != nil {
			t.Fatalf("%s: cannot decode: %v", sample.Name, err)
		}
		got := preprocess(img, engine.Height())
		if got.h != sample.Height || got.w != sample.Width {
			t.Fatalf("%s: preprocessed to %dx%d, want %dx%d",
				sample.Name, got.h, got.w, sample.Height, sample.Width)
		}
		var maxDiff, sum float64
		for i := range want.data {
			d := math.Abs(float64(got.data[i] - want.data[i]))
			sum += d
			if d > maxDiff {
				maxDiff = d
			}
		}
		mean := sum / float64(len(want.data))
		t.Logf("%s: max|diff| = %.4f/255, mean = %.6f", sample.Name, maxDiff*255, mean)
		if maxDiff > tol {
			t.Fatalf("%s: max difference %.4f/255 exceeds the tolerance %.1f/255",
				sample.Name, maxDiff*255, tol*255)
		}
		if mean > 0.002 {
			t.Fatalf("%s: mean difference %.6f is too large", sample.Name, mean)
		}
	}
}

// TestPreprocessShapes checks that images of many sizes and color models all
// become [1,64,W] with values in [0,1].
//
// TestPreprocessShapes 校验各种尺寸/色彩模型的图片都能变成 [1,64,W] 且落在 [0,1]。
func TestPreprocessShapes(t *testing.T) {
	cases := []struct {
		name string
		img  image.Image
		w, h int
	}{
		{"rgba", solid(image.Rect(0, 0, 70, 30), color.NRGBA{R: 200, G: 30, B: 30, A: 255}), 70, 30},
		{"gray", solid(image.Rect(0, 0, 30, 70), color.Gray{Y: 128}), 30, 70},
		{"single", solid(image.Rect(0, 0, 1, 1), color.Black), 1, 1},
		{"wide", solid(image.Rect(0, 0, 500, 100), color.White), 500, 100},
	}
	for _, c := range cases {
		out := preprocess(c.img, 64)
		wantW := int(float64(c.w) * (64.0 / float64(c.h)))
		if out.h != 64 || out.w != wantW {
			t.Fatalf("%s: got %dx%d, want 64x%d", c.name, out.h, out.w, wantW)
		}
		if len(out.data) != 64*out.w {
			t.Fatalf("%s: got %d values, want %d", c.name, len(out.data), 64*out.w)
		}
		for i, v := range out.data {
			if v < 0 || v > 1 {
				t.Fatalf("%s: value %d is %v, outside [0,1]", c.name, i, v)
			}
		}
	}
}

// solid returns an image filled with one color.
//
// solid 返回一张填满单一颜色的图片。
func solid(r image.Rectangle, c color.Color) image.Image {
	img := image.NewRGBA(r)
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			img.Set(x, y, c)
		}
	}
	return img
}
