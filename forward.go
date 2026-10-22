package ddddocr

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
)

// tensor3 is an intermediate tensor in CHW layout.
//
// tensor3 是 CHW 布局的中间张量。
type tensor3 struct {
	c, h, w int
	data    []float32
}

func newTensor3(c, h, w int) tensor3 {
	return tensor3{c: c, h: h, w: w, data: make([]float32, c*h*w)}
}

// conv3 runs a 2D convolution with bias; the weights are laid out as
// [oc][ic][kh][kw].
//
// conv3 做 2D 卷积(含偏置), 权重布局 [oc][ic][kh][kw]。
func conv3(x tensor3, weights, bias []float32, oc, kh, kw, stride, pad int) tensor3 {
	ph, pw := x.h+2*pad, x.w+2*pad
	src := x.data
	if pad > 0 {
		src = make([]float32, x.c*ph*pw)
		for c := 0; c < x.c; c++ {
			dst := c * ph * pw
			srow := c * x.h * x.w
			for y := 0; y < x.h; y++ {
				copy(src[dst+(y+pad)*pw+pad:dst+(y+pad)*pw+pad+x.w], x.data[srow+y*x.w:srow+(y+1)*x.w])
			}
		}
	}
	ho := (ph-kh)/stride + 1
	wo := (pw-kw)/stride + 1
	out := newTensor3(oc, ho, wo)
	kSize := kh * kw
	rowLen := x.c * kSize

	workers := convWorkers(oc, ho, wo, rowLen)
	if workers <= 1 {
		convRange(out, src, weights, bias, 0, oc, x.c, ph, pw, ho, wo, kh, kw, kSize, stride)
		return out
	}
	var wg sync.WaitGroup
	chunk := (oc + workers - 1) / workers
	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if end > oc {
			end = oc
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(from, to int) {
			defer wg.Done()
			convRange(out, src, weights, bias, from, to, x.c, ph, pw, ho, wo, kh, kw, kSize, stride)
		}(start, end)
	}
	wg.Wait()
	return out
}

// convWorkers only pays off on large convolutions; on small ones going
// parallel is slower than staying serial.
//
// convWorkers 只在大卷积上开并行的收益才明显, 小卷积串行反而更快。
func convWorkers(oc, ho, wo, rowLen int) int {
	macs := oc * ho * wo * rowLen
	if macs < 400_000 {
		return 1
	}
	procs := runtime.GOMAXPROCS(0)
	n := 4
	if macs > 4_000_000 {
		n = procs
	}
	// Do not spawn more goroutines than CPUs: on small machines the context
	// switches would eat the gain.
	//
	// 小机器上别开比核数还多的 goroutine: 切换开销会吃掉并行收益。
	if n > procs {
		n = procs
	}
	if n > oc {
		n = oc
	}
	return n
}

// convRange computes the output channels from <= o < to.
//
// The loop order is "output channel -> input channel -> kernel tap -> output
// pixel": each weight is loaded once outside the innermost loop and reused for a
// whole output row, so the accumulator is that row in L1 rather than a register,
// and the additions of neighbouring pixels are independent. It is far faster
// than walking 9*ic weights per pixel.
//
// convRange 计算 from<=o<to 这些输出通道。
//
// 循环顺序是「输出通道 → 输入通道 → 卷积核 → 输出像素」: 一个权重只在最内层循环外读一次, 就能反复用在整行输出上, 累加器是 L1 里的一行输出而不是寄存器, 每个像素的加法链互相独立。这比「逐像素把 9*ic 个权重读一遍」快得多。
func convRange(out tensor3, src, weights, bias []float32, from, to int,
	ic, ph, pw, ho, wo, kh, kw, kSize, stride int) {
	for o := from; o < to; o++ {
		oBase := o * ho * wo
		row := out.data[oBase : oBase+ho*wo]
		for i := range row {
			row[i] = 0
		}
		for c := 0; c < ic; c++ {
			wc := (o*ic + c) * kSize
			sBase := c * ph * pw
			for ky := 0; ky < kh; ky++ {
				for kx := 0; kx < kw; kx++ {
					wv := weights[wc+ky*kw+kx]
					if wv == 0 {
						continue
					}
					for y := 0; y < ho; y++ {
						sp := sBase + (y*stride+ky)*pw + kx
						op := y * wo
						if stride == 1 {
							x := 0
							for ; x+4 <= wo; x += 4 {
								row[op+x] += wv * src[sp+x]
								row[op+x+1] += wv * src[sp+x+1]
								row[op+x+2] += wv * src[sp+x+2]
								row[op+x+3] += wv * src[sp+x+3]
							}
							for ; x < wo; x++ {
								row[op+x] += wv * src[sp+x]
							}
							continue
						}
						for x := 0; x < wo; x++ {
							row[op+x] += wv * src[sp+x*stride]
						}
					}
				}
			}
		}
		b := bias[o]
		for i := range row {
			row[i] += b
		}
	}
}

// swish computes x * sigmoid(x) in place and returns x.
//
// swish 计算 x * sigmoid(x), 复用入参内存。
func swish(x tensor3) tensor3 {
	for i, v := range x.data {
		x.data[i] = v / (1 + float32(math.Exp(float64(-v))))
	}
	return x
}

// add3 adds two tensors element-wise.
//
// add3 做逐元素相加。
func add3(a, b tensor3) tensor3 {
	out := newTensor3(a.c, a.h, a.w)
	for i, v := range a.data {
		out.data[i] = v + b.data[i]
	}
	return out
}

// toSeq turns [C,H,W] into a sequence [W][C*H], matching the Transpose[3,0,1,2]
// plus Reshape in the ONNX graph.
//
// toSeq 把 [C,H,W] 变成序列 [W][C*H], 对应图里的 Transpose[3,0,1,2] + Reshape。
func toSeq(x tensor3) [][]float32 {
	seq := make([][]float32, x.w)
	hw := x.h * x.w
	for t := 0; t < x.w; t++ {
		row := make([]float32, x.c*x.h)
		for c := 0; c < x.c; c++ {
			base := c*hw + t
			for y := 0; y < x.h; y++ {
				row[c*x.h+y] = x.data[base+y*x.w]
			}
		}
		seq[t] = row
	}
	return seq
}

// matvec computes dst = m @ v, with m a rows×n row-major matrix.
//
// matvec 计算 dst = m @ v, m 为 rows×n 的行主序矩阵。
func matvec(dst, m, v []float32, n int) {
	for r := range dst {
		row := m[r*n : r*n+n]
		var sum float32
		for i := 0; i < n; i++ {
			sum += row[i] * v[i]
		}
		dst[r] = sum
	}
}

// addMatvec computes dst += m @ v.
//
// addMatvec 计算 dst += m @ v。
func addMatvec(dst, m, v []float32, n int) {
	for r := range dst {
		row := m[r*n : r*n+n]
		var sum float32
		for i := 0; i < n; i++ {
			sum += row[i] * v[i]
		}
		dst[r] += sum
	}
}

// lstm runs an ONNX-style bidirectional LSTM with gate order i, o, f, c.
//
// lstm 跑 ONNX 风格的双向 LSTM(门顺序 i, o, f, c)。
func lstm(seq [][]float32, w, r, b []float32, hidden, dirs int) [][]float32 {
	steps := len(seq)
	in := len(seq[0])
	gate := 4 * hidden
	out := make([][]float32, steps)
	for i := range out {
		out[i] = make([]float32, dirs*hidden)
	}
	gates := make([]float32, gate)
	hPrev := make([]float32, hidden)
	cPrev := make([]float32, hidden)

	for d := 0; d < dirs; d++ {
		wOff := d * gate * in
		rOff := d * gate * hidden
		bOff := d * 8 * hidden
		for i := range hPrev {
			hPrev[i] = 0
			cPrev[i] = 0
		}
		for step := 0; step < steps; step++ {
			idx := step
			if d == 1 {
				idx = steps - 1 - step
			}
			matvec(gates, w[wOff:wOff+gate*in], seq[idx], in)
			addMatvec(gates, r[rOff:rOff+gate*hidden], hPrev, hidden)
			for g := 0; g < gate; g++ {
				gates[g] += b[bOff+g] + b[bOff+gate+g]
			}
			for j := 0; j < hidden; j++ {
				iGate := sigmoid(gates[j])
				oGate := sigmoid(gates[hidden+j])
				fGate := sigmoid(gates[2*hidden+j])
				cand := float32(math.Tanh(float64(gates[3*hidden+j])))
				cPrev[j] = fGate*cPrev[j] + iGate*cand
				hPrev[j] = oGate * float32(math.Tanh(float64(cPrev[j])))
			}
			copy(out[idx][d*hidden:(d+1)*hidden], hPrev)
		}
	}
	return out
}

func sigmoid(v float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-v))))
}

// gemm computes seq @ W^T + b, with W laid out as [out][in].
//
// gemm 计算 seq @ W^T + b, W 布局为 [out][in]。
func gemm(seq [][]float32, w, b []float32) [][]float32 {
	outDim := len(b)
	in := len(seq[0])
	out := make([][]float32, len(seq))
	for t, row := range seq {
		res := make([]float32, outDim)
		for o := 0; o < outDim; o++ {
			wr := w[o*in : o*in+in]
			var sum float32
			for i := 0; i < in; i++ {
				sum += wr[i] * row[i]
			}
			res[o] = sum + b[o]
		}
		out[t] = res
	}
	return out
}

// argmaxRow returns the index of the largest value in one row of logits.
//
// argmaxRow 返回一行 logits 的最大值下标。
func argmaxRow(row []float32) int {
	best, bestI := row[0], 0
	for i := 1; i < len(row); i++ {
		if row[i] > best {
			best, bestI = row[i], i
		}
	}
	return bestI
}

// argmaxRows returns the argmax of every row.
//
// argmaxRows 逐行取最大值的下标。
func argmaxRows(seq [][]float32) []int {
	idx := make([]int, len(seq))
	for t, row := range seq {
		idx[t] = argmaxRow(row)
	}
	return idx
}

// argmaxProb returns the argmax of one row of logits together with the softmax
// probability of that index.
//
// Subtracting the maximum before the exponential makes the largest softmax term
// exactly 1, so the probability is 1/Σexp(v-max) and the full row never has to
// be produced.
//
// argmaxProb 返回一行 logits 的最大值下标, 以及该下标 softmax 之后的概率。
//
// 先减去最大值再取指数, softmax 中最大项恒为 1, 所以概率就是 1/Σexp(v-max),
// 不需要真的算出一整行概率。
func argmaxProb(row []float32) (int, float32) {
	i := argmaxRow(row)
	maxv := float64(row[i])
	var sum float64
	for _, v := range row {
		sum += math.Exp(float64(v) - maxv)
	}
	return i, float32(1 / sum)
}

// forwardSeq runs the exported program and returns the classification logits of
// every time step.
//
// forwardSeq 按导出的程序跑一遍前向, 返回每个时间步的分类 logits。
func (m *model) forwardSeq(x tensor3) ([][]float32, error) {
	prog := m.header.Program
	saved := make([]tensor3, len(prog))
	cur := x
	var seq [][]float32
	for i, st := range prog {
		switch st.Op {
		case "conv":
			if len(st.Kernel) != 2 {
				return nil, fmt.Errorf("ddddocr: step %d: conv has no kernel", i)
			}
			cur = conv3(cur, m.tensor(st.W), m.tensor(st.B), st.Out, st.Kernel[0], st.Kernel[1], st.Stride, st.Pad)
			saved[i] = cur
		case "swish":
			cur = swish(cur)
			saved[i] = cur
		case "sigmoid":
			for k, v := range cur.data {
				cur.data[k] = sigmoid(v)
			}
			saved[i] = cur
		case "add":
			if st.Ref < 0 || st.Ref >= len(saved) {
				return nil, fmt.Errorf("ddddocr: step %d: residual source %d is out of range", i, st.Ref)
			}
			cur = add3(cur, saved[st.Ref])
			saved[i] = cur
		case "to_seq":
			seq = toSeq(cur)
		case "lstm":
			if len(seq) == 0 {
				return nil, fmt.Errorf("ddddocr: step %d: lstm before to_seq", i)
			}
			seq = lstm(seq, m.tensor(st.W), m.tensor(st.R), m.tensor(st.B), st.Hidden, st.Dirs)
		case "gemm":
			if len(seq) == 0 {
				return nil, fmt.Errorf("ddddocr: step %d: gemm before to_seq", i)
			}
			if st.W == "" || st.B == "" {
				return nil, ErrNoHead
			}
			seq = gemm(seq, m.tensor(st.W), m.tensor(st.B))
		case "argmax":
			if len(seq) == 0 {
				return nil, fmt.Errorf("ddddocr: step %d: argmax before to_seq", i)
			}
			return seq, nil
		default:
			return nil, fmt.Errorf("ddddocr: step %d: unknown op %q", i, st.Op)
		}
	}
	if seq == nil {
		return nil, errors.New("ddddocr: the program produced no output")
	}
	return seq, nil
}

// recognize runs the exported program and returns the class index of every time
// step.
//
// recognize 按导出的程序跑一遍前向, 返回每个时间步的类别下标。
func (m *model) recognize(x tensor3) ([]int, error) {
	seq, err := m.forwardSeq(x)
	if err != nil {
		return nil, err
	}
	return argmaxRows(seq), nil
}

// recognizeProb is recognize plus the softmax probability of every picked
// index.
//
// recognizeProb 与 recognize 相同, 但额外返回每个时间步最大值的概率。
func (m *model) recognizeProb(x tensor3) ([]int, []float32, error) {
	seq, err := m.forwardSeq(x)
	if err != nil {
		return nil, nil, err
	}
	idx := make([]int, len(seq))
	probs := make([]float32, len(seq))
	for t, row := range seq {
		idx[t], probs[t] = argmaxProb(row)
	}
	return idx, probs, nil
}
