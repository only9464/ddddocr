package ddddocr

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"math"

	// Formats the standard library cannot decode (BMP/TIFF/WebP). These three
	// packages only register decoders, are pure Go, and are the sole external
	// dependency of this module (see go.mod).
	//
	// 标准库解不了的格式(BMP/TIFF/WebP)。这三个包只注册解码器, 是纯 Go 实现,
	// 也是本库唯一的外部依赖(见 go.mod)。
	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"
)

// defaultHeight is the model input height (input1 = [1,1,64,W]). The weights
// header also records a height, and that one wins.
//
// 模型输入高度 (input1 = [1,1,64,W])。权重头里也会记录 height, 那里优先。
const defaultHeight = 64

// maxWidth keeps an extremely wide image from eating all memory.
//
// maxWidth 防止整行长图把内存吃爆。
const maxWidth = 4096

// decodeImage decodes JPEG, PNG, GIF, BMP, TIFF or WebP.
//
// decodeImage 解码 JPEG/PNG/GIF/BMP/TIFF/WebP。
func decodeImage(data []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("ddddocr: cannot decode the image: %w", err)
	}
	return img, nil
}

// preprocess turns an image of any size into the model input [1,height,W].
//
// The result matches ddddocr's PIL pipeline bit for bit:
//  1. scale the height to height and the width proportionally (int(w*height/h));
//  2. resize the R, G and B planes separately with Lanczos3, rounding the
//     result of each pass (x and y) back to uint8: PIL keeps 8-bit
//     intermediates, and not rounding yields different values at high-contrast
//     edges (PIL clamps the Lanczos overshoot);
//  3. convert to gray with the ITU-R 601 fixed-point formula and normalize to
//     [0,1].
//
// preprocess 把任意尺寸的图片变成模型输入 [1,height,W]。
//
// 与 ddddocr 的预处理逐位对齐: 高度缩放到 height, 宽度按比例; 对 R/G/B 三个通道分别做 Lanczos3 缩放, 每一趟(x 方向、y 方向) 都把结果量化回 uint8 —— PIL 的中间结果是 8 位的, 不量化会在高反差边缘得到不同的值; 最后按 ITU-R 601 定点公式转灰度, 再归一化到 [0,1]。
func preprocess(img image.Image, height int) tensor3 {
	if height <= 0 {
		height = defaultHeight
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return newTensor3(1, height, 1)
	}
	targetW := int(float64(w) * (float64(height) / float64(h)))
	if targetW < 1 {
		targetW = 1
	}
	if targetW > maxWidth {
		targetW = maxWidth
	}

	r, g, bl := rgbPlanes(img)
	xk := lanczosWeights(w, targetW)
	yk := lanczosWeights(h, height)
	r = resizePlane(r, w, h, targetW, height, xk, yk)
	g = resizePlane(g, w, h, targetW, height, xk, yk)
	bl = resizePlane(bl, w, h, targetW, height, xk, yk)

	out := newTensor3(1, height, targetW)
	for i := range out.data {
		out.data[i] = float32(luma(r[i], g[i], bl[i])) / 255
	}
	return out
}

// rgbPlanes extracts the R, G and B planes of an image, dropping alpha the same
// way PIL's convert("L") does.
//
// rgbPlanes 取出图片的 R/G/B 三个平面(丢掉 alpha, 与 PIL 的 convert("L") 一致)。
func rgbPlanes(img image.Image) (r, g, b []uint8) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	r = make([]uint8, w*h)
	g = make([]uint8, w*h)
	b = make([]uint8, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := color.NRGBAModel.Convert(img.At(bounds.Min.X+x, bounds.Min.Y+y)).(color.NRGBA)
			i := y*w + x
			r[i], g[i], b[i] = c.R, c.G, c.B
		}
	}
	return r, g, b
}

// luma is the fixed-point formula PIL uses for RGB to L (0.299/0.587/0.114,
// rounded).
//
// luma 是 PIL RGB→L 用的定点公式 (0.299/0.587/0.114, 四舍五入)。
func luma(r, g, b uint8) uint8 {
	return uint8((19595*uint32(r) + 38470*uint32(g) + 7471*uint32(b) + 0x8000) >> 16)
}

// kernel is the list of (input index, weight) pairs that produce one output
// coordinate.
//
// kernel 是某个输出坐标对应的一串 (输入下标, 权重)。
type kernel struct {
	taps []int
	wts  []float32
}

// lanczosWeights precomputes the one-dimensional resampling weights (Lanczos3,
// with PIL's edge handling: clamp the taps, then normalize).
//
// lanczosWeights 预先算好一维重采样权重(Lanczos3, 边界处按 PIL 的做法裁剪后归一化)。
func lanczosWeights(in, out int) []kernel {
	const a = 3.0
	scale := float64(in) / float64(out)
	support := a
	if scale > 1 {
		support = a * scale
	}
	res := make([]kernel, out)
	for o := 0; o < out; o++ {
		center := (float64(o)+0.5)*scale - 0.5
		lo := int(math.Ceil(center - support))
		hi := int(math.Floor(center + support))
		if lo < 0 {
			lo = 0
		}
		if hi > in-1 {
			hi = in - 1
		}
		taps := make([]int, 0, hi-lo+1)
		wts := make([]float32, 0, hi-lo+1)
		var sum float64
		for i := lo; i <= hi; i++ {
			d := float64(i) - center
			if scale > 1 {
				d /= scale
			}
			v := lanczos(d, a)
			taps = append(taps, i)
			wts = append(wts, float32(v))
			sum += v
		}
		if sum != 0 {
			for i := range wts {
				wts[i] = float32(float64(wts[i]) / sum)
			}
		}
		res[o] = kernel{taps: taps, wts: wts}
	}
	return res
}

func lanczos(x, a float64) float64 {
	if x == 0 {
		return 1
	}
	if x < 0 {
		x = -x
	}
	if x >= a {
		return 0
	}
	px := math.Pi * x
	return a * math.Sin(px) * math.Sin(px/a) / (px * px)
}

// resizePlane resamples one channel with a separable Lanczos filter, rounding
// to uint8 after each pass exactly like PIL.
//
// resizePlane 对单个通道做可分离的 Lanczos 重采样, 与 PIL 一样每趟都量化到 uint8。
func resizePlane(src []uint8, inW, inH, outW, outH int, xk, yk []kernel) []uint8 {
	tmp := make([]uint8, outW*inH)
	for y := 0; y < inH; y++ {
		row := src[y*inW : (y+1)*inW]
		dst := tmp[y*outW : (y+1)*outW]
		for x := 0; x < outW; x++ {
			k := xk[x]
			var sum float32
			for i, tap := range k.taps {
				sum += float32(row[tap]) * k.wts[i]
			}
			dst[x] = clamp8(sum)
		}
	}
	out := make([]uint8, outW*outH)
	acc := make([]float32, outW)
	for y := 0; y < outH; y++ {
		k := yk[y]
		for x := range acc {
			acc[x] = 0
		}
		for i, tap := range k.taps {
			srcRow := tmp[tap*outW : (tap+1)*outW]
			wt := k.wts[i]
			for x := 0; x < outW; x++ {
				acc[x] += float32(srcRow[x]) * wt
			}
		}
		dst := out[y*outW : (y+1)*outW]
		for x, v := range acc {
			dst[x] = clamp8(v)
		}
	}
	return out
}

// clamp8 rounds and clamps to [0,255], matching PIL writing each pass back to
// uint8.
//
// clamp8 四舍五入并裁到 [0,255], 对应 PIL 每趟写回 uint8 的行为。
func clamp8(v float32) uint8 {
	n := int32(v + 0.5)
	if n < 0 {
		return 0
	}
	if n > 255 {
		return 255
	}
	return uint8(n)
}
