// Package backbone embeds the shared ddddocr backbone: convolutions plus a
// BiLSTM, about 20.4 MB of weights.
//
// A recognizer is backbone (20.4 MB) plus one classification head (its size
// grows with the charset). The backbone is identical for every charset, so a
// program only ever carries one copy; import the head package you need:
//
//	github.com/only9464/ddddocr-go/digits // 0-9         (45 KB head)
//	github.com/only9464/ddddocr-go/alnum  // 0-9A-Za-z   (259 KB head)
//	github.com/only9464/ddddocr-go/full   // 8210 chars  (33.7 MB head)
//
// For a custom charset (say 0-9 plus a few Chinese characters) export a head
// with tools/export.py and embed it yourself:
//
//	//go:embed my-head.bin
//	var myHead []byte
//
//	engine, err := ddddocr.NewFromSet(backbone.Bytes(), myHead)
//
// Package backbone 内嵌 ddddocr 的共享主干: 卷积 + BiLSTM, 约 20.4 MB。
//
// 识别模型 = backbone(20.4 MB) + 分类头(随字符数变化)。主干对所有字符集都一样, 所以一个程序里只会有一份; 想要什么字符集, 就只 import 对应的头包。想自定义字符集
// (比如只认 0-9 加几个汉字), 用 tools/export.py 生成一个头, 然后自己 embed 它。
package backbone

import (
	"embed"
	"errors"

	"github.com/only9464/ddddocr-go/internal/modelbin"
)

//go:embed model
var modelFS embed.FS

// ErrMissing reports that backbone.bin has not been exported yet.
//
// ErrMissing 表示还没有导出 backbone.bin。
var ErrMissing = errors.New("ddddocr: backbone weights are not embedded, run tools/export.py first")

// Bytes returns the embedded backbone weights.
//
// Bytes 返回内嵌的主干权重。
func Bytes() ([]byte, error) {
	data, err := modelbin.Read(modelFS, "model/backbone.bin")
	if err != nil {
		return nil, ErrMissing
	}
	return data, nil
}

// Size returns the size of the backbone weights in bytes, or 0 when they were
// not exported.
//
// Size 返回主干权重的字节数(没导出时为 0)。
func Size() int64 { return modelbin.Size(modelFS, "model/backbone.bin") }
