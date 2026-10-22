package ddddocr

import (
	"errors"
	"image"
	"os"
	"strings"
)

var (
	// ErrNoWeights reports that no weights are available, for example when a
	// charset package was built without exporting them.
	//
	// ErrNoWeights 表示没有找到权重文件(子包还没导出权重时会返回它)。
	ErrNoWeights = errors.New("ddddocr: no weights found, export them with tools/export.py")

	// ErrNoHead reports that the weights carry no classification head, which
	// happens when only the backbone is supplied.
	//
	// ErrNoHead 表示权重里没有分类头(只给了 backbone 时)。
	ErrNoHead = errors.New("ddddocr: weights contain no classification head, use NewFromSet(backbone, head)")
)

// Recognizer recognizes captchas with weights that are already loaded.
//
// The weights are read-only after construction and every call allocates its own
// scratch buffers, so a single Recognizer may be shared by any number of
// goroutines.
//
// Recognizer 是一个加载好权重、可并发复用的识别器。
// 权重在构造后只读, 每次识别用的缓冲都是本次调用内新分配的, 所以同一个
// Recognizer 可以被任意多个 goroutine 同时使用。
type Recognizer struct {
	m *model
}

// New returns a Recognizer from a single-file weight blob that already carries
// its own classification head (the --single-out format). Use NewFromSet with a
// backbone plus a head instead when you want to keep the two apart.
//
// New 用一个「自带分类头」的完整权重文件构造识别器(单文件模式)。
// 拆分模式下请用 NewFromSet。
func New(weights []byte) (*Recognizer, error) {
	if len(weights) == 0 {
		return nil, ErrNoWeights
	}
	m, err := loadModel(weights)
	if err != nil {
		return nil, err
	}
	if len(m.header.Charset) == 0 {
		return nil, ErrNoHead
	}
	if len(m.header.Program) == 0 {
		return nil, errors.New("ddddocr: weights contain no forward program")
	}
	return &Recognizer{m: m}, nil
}

// NewFromSet returns a Recognizer built from a shared backbone plus one
// classification head.
//
// The backbone is the 20.4 MB convolution/BiLSTM part shared by every charset;
// the head holds only the classification matrix and its charset, so its size
// follows the number of classes (45 KB for digits, 33.7 MB for the full
// charset).
//
// NewFromSet 用「共享 backbone + 某个分类头」构造识别器。
// backbone 是 20.4 MB 的卷积/BiLSTM 部分, 所有字符集共用; head 只有分类头矩阵和字符集, 体积随字符数变化(数字 45 KB, 全字符集 33.7 MB)。
func NewFromSet(backbone, head []byte) (*Recognizer, error) {
	if len(backbone) == 0 {
		return nil, ErrNoWeights
	}
	if len(head) == 0 {
		return nil, ErrNoHead
	}
	m, err := loadModel(backbone)
	if err != nil {
		return nil, err
	}
	h, err := loadModel(head)
	if err != nil {
		return nil, err
	}
	if err := m.applyHead(h); err != nil {
		return nil, err
	}
	if len(m.header.Program) == 0 {
		return nil, errors.New("ddddocr: backbone weights contain no forward program")
	}
	return &Recognizer{m: m}, nil
}

// Charset returns the charset of the classification head; index 0 is the CTC
// blank. The returned slice is a copy, so callers may modify it freely.
//
// Charset 返回模型分类头对应的字符集, 下标 0 是 CTC 的 blank。
// 返回的是副本, 调用方可以随意修改。
func (r *Recognizer) Charset() []string {
	if r == nil || r.m == nil {
		return nil
	}
	out := make([]string, len(r.m.header.Charset))
	copy(out, r.m.header.Charset)
	return out
}

// Height returns the input image height the model expects (64 for the common
// models).
//
// Height 返回模型要求的输入图片高度(common 模型是 64)。
func (r *Recognizer) Height() int {
	if r == nil || r.m == nil {
		return 0
	}
	return r.m.header.Height
}

// Recognize recognizes an encoded image (raw bytes of JPEG, PNG, GIF, BMP, TIFF
// or WebP) and returns the CTC-decoded text.
//
// Recognize 识别一张图片(JPEG/PNG/GIF/BMP/TIFF/WebP 的原始字节), 返回 CTC
// 去重后的文本。
func (r *Recognizer) Recognize(data []byte) (string, error) {
	idx, err := r.RecognizeIndices(data)
	if err != nil {
		return "", err
	}
	return r.decode(idx), nil
}

// Classification is an alias for Recognize.
//
// The name matches ddddocr.DdddOcr.classification of the Python package, so code
// ported from Python can keep using it; the behavior is identical.
//
// Classification 是 Recognize 的别名。
// 方法名和 Python 版 ddddocr.DdddOcr.classification 对齐, 从 Python 迁过来的代码可以直接照抄; 两者行为完全一样。
func (r *Recognizer) Classification(data []byte) (string, error) {
	return r.Recognize(data)
}

// CharProb is one recognized character together with its confidence.
//
// CharProb 是识别结果里的一个字符和它的置信度。
type CharProb struct {
	Char string  // 字符本身 / the character itself
	Prob float32 // 该时间步 softmax 之后的概率, 0~1 / softmax probability at that step, 0~1
}

// ClassificationProb matches Python's classification(img, probability=True):
// besides the text it returns the confidence of every character in it, with
// duplicates and blanks already removed by the CTC rule.
//
// The Python version returns the whole time-step × charset probability matrix;
// this one returns only the characters that survive, which is far smaller and
// enough to implement "retry when confidence is low" logic.
//
// ClassificationProb 对应 Python 版的 classification(img, probability=True):
// 除了文本, 还返回文本中每个字符的置信度(同样按 CTC 规则去掉了重复和空白)。
// Python 版返回的是「时间步 × 字符集」的整张概率矩阵, 这里只给最终留下的那些字符, 体积小得多, 需要做「低置信度就重试」这类判断时够用。
func (r *Recognizer) ClassificationProb(data []byte) (string, []CharProb, error) {
	if r == nil || r.m == nil {
		return "", nil, ErrNoWeights
	}
	img, err := decodeImage(data)
	if err != nil {
		return "", nil, err
	}
	idx, probs, err := r.m.recognizeProb(preprocess(img, r.Height()))
	if err != nil {
		return "", nil, err
	}
	text, chars := r.decodeWithProb(idx, probs)
	return text, chars, nil
}

// RecognizeImage recognizes an already decoded image.Image.
//
// RecognizeImage 识别一个已解码的 image.Image。
func (r *Recognizer) RecognizeImage(img image.Image) (string, error) {
	idx, err := r.indicesOf(img)
	if err != nil {
		return "", err
	}
	return r.decode(idx), nil
}

// RecognizeFile recognizes a local image file.
//
// RecognizeFile 识别一个本地图片文件。
func (r *Recognizer) RecognizeFile(name string) (string, error) {
	data, err := os.ReadFile(name)
	if err != nil {
		return "", err
	}
	return r.Recognize(data)
}

// RecognizeIndices returns the class index of every time step before CTC
// deduplication. It is meant for debugging and for comparing against a
// reference implementation.
//
// RecognizeIndices 返回每个时间步的类别下标(未去重, 调试/对比模型时用)。
func (r *Recognizer) RecognizeIndices(data []byte) ([]int, error) {
	if r == nil || r.m == nil {
		return nil, ErrNoWeights
	}
	img, err := decodeImage(data)
	if err != nil {
		return nil, err
	}
	return r.indicesOf(img)
}

// indicesOf runs the forward pass and returns the per-step class indices.
//
// indicesOf 抽取图片特征后跑一遍前向, 返回逐时间步的下标。
func (r *Recognizer) indicesOf(img image.Image) ([]int, error) {
	return r.m.recognize(preprocess(img, r.Height()))
}

// decode applies the CTC rule: drop repeated indices and the blank (index 0).
//
// decode 做 CTC 去重: 跳过相邻重复与 blank(下标 0)。
func (r *Recognizer) decode(idx []int) string {
	text, _ := r.decodeWithProb(idx, nil)
	return text
}

// decodeWithProb applies the CTC rule and, when probs is non-nil, also collects
// the confidence of every character it keeps.
//
// decodeWithProb 做 CTC 去重; probs 非 nil 时顺带收集每个保留字符的置信度。
func (r *Recognizer) decodeWithProb(idx []int, probs []float32) (string, []CharProb) {
	charset := r.m.header.Charset
	var b strings.Builder
	var out []CharProb
	prev := -1
	for t, i := range idx {
		if i == prev || i == 0 || i < 0 || i >= len(charset) {
			prev = i
			continue
		}
		prev = i
		b.WriteString(charset[i])
		if t < len(probs) {
			out = append(out, CharProb{Char: charset[i], Prob: probs[t]})
		}
	}
	return b.String(), out
}
