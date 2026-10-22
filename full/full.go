// Package full recognizes the complete ddddocr charset (8210 classes: Chinese,
// English, digits and symbols).
//
// Size: the shared 20.4 MB backbone plus a 33.7 MB head. When digits or letters
// are all you need, use digits or alnum instead and save those 33.7 MB.
//
//	engine, err := full.New()
//	text, err := engine.RecognizeFile("captcha.jpg")
//
// Package full 是 ddddocr 的完整字符集模型(8210 类, 中英文/符号都能认)。
// 体积: backbone 20.4 MB(所有字符集共用) + 本包的头 33.7 MB。只需要数字或字母数字时请用 digits / alnum, 能省下 33.7 MB。
package full

import (
	"embed"

	ddddocr "github.com/only9464/ddddocr-go"
	"github.com/only9464/ddddocr-go/internal/charset"
)

//go:embed model
var modelFS embed.FS

// engine caches the recognizer shared by the whole process.
//
// engine 缓存整个进程共用的识别器。
var engine charset.Engine

// New returns the recognizer for the built-in full charset model. The weights
// are loaded once per process and the result may be used by many goroutines.
//
// New 返回内置全字符集模型的识别器。权重只加载一次, 返回的识别器可并发使用。
func New() (*ddddocr.Recognizer, error) { return engine.Load(modelFS, "model/head.bin") }

// Head returns the embedded classification head; it reports ddddocr.ErrNoHead
// when the head has not been exported.
//
// Head 返回内嵌的分类头字节; 还没导出时返回 ddddocr.ErrNoHead。
func Head() ([]byte, error) { return charset.Head(modelFS, "model/head.bin") }
