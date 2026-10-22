package ddddocr

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
)

// weightsMagic starts every weight file; the four bytes after it are the
// little-endian length of the JSON header that follows.
//
// weightsMagic 是权重文件的魔数, 后面紧跟 uint32(小端) 的 JSON 头长度。
const weightsMagic = "OCRW1\x00\x00\x00"

// weightsFormat and weightsVersion are declared in the JSON header and written
// by tools/export.py. The exporting script also writes a kind field
// (single/backbone/head) for humans; readers ignore it, because the presence of
// a charset already tells a backbone and a head apart.
//
// weightsFormat/weightsVersion 是 JSON 头里声明的格式与版本, 由 tools/export.py 写出。
// 导出脚本还会写一个 kind 字段(single/backbone/head) 供人阅读, 读取时用不上:有没有字符集就足以区分主干和分类头。
const (
	weightsFormat  = "ddddocr-go-weights"
	weightsVersion = 1
)

// Tensor names of the two classification head arrays; tools/export.py writes
// them in split mode.
//
// 分类头里的两个张量名, 由 tools/export.py 在拆分模式下写出。
const (
	headWeightName = "gemm_w"
	headBiasName   = "gemm_b"
)

// weightTensor describes one float32 block inside a weight file.
//
// weightTensor 描述权重文件里的一段 float32 数据。
type weightTensor struct {
	Name   string `json:"name"`
	Shape  []int  `json:"shape"`
	Offset int    `json:"offset"`
	Bytes  int    `json:"bytes"`
}

// programStep is one step of the exported forward program; which fields are
// used depends on Op.
//
// programStep 是前向程序的一步, 字段按 op 取用。
type programStep struct {
	Op     string `json:"op"`               // conv / swish / add / to_seq / lstm / gemm / argmax
	W      string `json:"w,omitempty"`      // weight tensor / 权重张量名
	B      string `json:"b,omitempty"`      // bias tensor / 偏置张量名
	R      string `json:"r,omitempty"`      // LSTM recurrent weight / LSTM 循环权重张量名
	Stride int    `json:"stride,omitempty"` // conv stride / conv 步长
	Pad    int    `json:"pad,omitempty"`    // conv padding / conv 四周补零
	Kernel []int  `json:"kernel,omitempty"` // conv kernel [kh, kw] / conv 卷积核大小
	Out    int    `json:"out,omitempty"`    // conv output channels / conv 输出通道数
	Hidden int    `json:"hidden,omitempty"` // LSTM hidden size / LSTM 隐藏层维度
	Dirs   int    `json:"dirs,omitempty"`   // LSTM directions (1 or 2) / LSTM 方向数
	Ref    int    `json:"ref,omitempty"`    // residual source step of add / add 的残差来源步号
}

// weightHeader is the JSON header of a weight file.
//
// weightHeader 是权重文件的 JSON 头。
type weightHeader struct {
	Format  string         `json:"format"`
	Version int            `json:"version"`
	Height  int            `json:"height"`
	Charset []string       `json:"charset"`
	Program []programStep  `json:"program"`
	Tensors []weightTensor `json:"tensors"`
}

// model is a loaded set of weights: the header plus every tensor decoded into
// float32 slices.
//
// model 是加载后的权重组。
type model struct {
	header  weightHeader
	tensors map[string][]float32
}

// loadModel parses the byte blob of a weight file.
//
// loadModel 解析权重字节。
func loadModel(raw []byte) (*model, error) {
	if len(raw) < len(weightsMagic)+4 {
		return nil, fmt.Errorf("ddddocr: weights blob is too small (%d bytes)", len(raw))
	}
	if string(raw[:len(weightsMagic)]) != weightsMagic {
		return nil, errors.New("ddddocr: weights have a wrong magic number, re-export them with tools/export.py")
	}
	headLen := int(binary.LittleEndian.Uint32(raw[len(weightsMagic) : len(weightsMagic)+4]))
	start := len(weightsMagic) + 4
	if headLen <= 0 || len(raw) < start+headLen {
		return nil, errors.New("ddddocr: weights header length is out of range")
	}
	var header weightHeader
	if err := json.Unmarshal(raw[start:start+headLen], &header); err != nil {
		return nil, fmt.Errorf("ddddocr: parsing the weights header: %w", err)
	}
	if header.Format != weightsFormat {
		return nil, fmt.Errorf("ddddocr: unknown weights format %q, re-export with a newer tools/export.py", header.Format)
	}
	if header.Version > weightsVersion {
		return nil, fmt.Errorf("ddddocr: weights version %d is newer than the supported version %d, upgrade the library",
			header.Version, weightsVersion)
	}
	if header.Height <= 0 {
		header.Height = defaultHeight
	}
	blob := raw[start+headLen:]
	m := &model{header: header, tensors: make(map[string][]float32, len(header.Tensors))}
	for _, t := range header.Tensors {
		if t.Offset < 0 || t.Bytes < 0 || t.Offset+t.Bytes > len(blob) {
			return nil, fmt.Errorf("ddddocr: weights for %q are out of range", t.Name)
		}
		data := blob[t.Offset : t.Offset+t.Bytes]
		values := make([]float32, len(data)/4)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
		m.tensors[t.Name] = values
	}
	return m, nil
}

// applyHead merges a classification head into a backbone: it names the head
// tensors for the program's gemm step and swaps in the head's charset. A head
// may hold any subset of characters, so one backbone can serve any number of
// "digits only", "letters only" or "just these few characters" heads.
//
// applyHead 把分类头(head) 并进 backbone: 给程序里的 gemm 步骤补上权重名,再把字符集换成 head 里的那一套。head 可以是任意字符子集, 因此同一条
// backbone 可以配无数个"只要数字/只要字母/只用其中几个字"的头。
func (m *model) applyHead(h *model) error {
	if len(h.header.Charset) == 0 {
		return errors.New("ddddocr: the classification head has no charset")
	}
	w := h.tensor(headWeightName)
	b := h.tensor(headBiasName)
	if len(b) == 0 {
		return fmt.Errorf("ddddocr: the classification head is missing %s", headBiasName)
	}
	if len(b) != len(h.header.Charset) {
		return fmt.Errorf("ddddocr: the classification head %s has %d rows but the charset has %d entries",
			headBiasName, len(b), len(h.header.Charset))
	}
	if len(w) == 0 || len(w)%len(b) != 0 {
		return fmt.Errorf("ddddocr: the classification head has a bad shape (%d float32 values for %d classes)",
			len(w), len(b))
	}
	if in := len(w) / len(b); in != 1024 {
		return fmt.Errorf("ddddocr: the classification head input dimension is %d, want 1024", in)
	}

	patched := false
	for i := range m.header.Program {
		if m.header.Program[i].Op != "gemm" {
			continue
		}
		m.header.Program[i].W = headWeightName
		m.header.Program[i].B = headBiasName
		patched = true
		break
	}
	if !patched {
		return errors.New("ddddocr: the backbone has no gemm step to attach a classification head to")
	}
	m.tensors[headWeightName] = w
	m.tensors[headBiasName] = b
	m.header.Charset = h.header.Charset
	return nil
}

// tensor returns one weight array, or nil when the name is empty or unknown.
//
// tensor 取一段权重, 名字为空或不存在时返回 nil。
func (m *model) tensor(name string) []float32 {
	if name == "" {
		return nil
	}
	return m.tensors[name]
}
