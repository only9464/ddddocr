// Package alnum recognizes digits and ASCII letters: the classification head
// keeps blank plus 0-9, A-Z and a-z, 63 classes in total.
//
// Size: the shared 20.4 MB backbone plus a 259 KB head. It suits the common
// mixed letter/digit captcha.
//
//	engine, err := alnum.New()
//	text, err := engine.RecognizeFile("captcha.jpg")
//
// Package alnum 是「数字 + 大小写字母」的 ddddocr 模型: 分类头保留空白 + 0-9 +
// A-Z + a-z 共 63 类。体积: backbone 20.4 MB(所有字符集共用) + 本包的头 259 KB,
// 适合常见的字母数字混合验证码。
package alnum

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

// New returns the recognizer for the built-in digit-and-letter model. The
// weights are loaded once per process and the result may be used by many
// goroutines.
//
// New 返回内置「数字 + 大小写字母」模型的识别器。
func New() (*ddddocr.Recognizer, error) { return engine.Load(modelFS, "model/head.bin") }

// Head returns the embedded classification head; it reports ddddocr.ErrNoHead
// when the head has not been exported.
//
// Head 返回内嵌的分类头字节; 还没导出时返回 ddddocr.ErrNoHead。
func Head() ([]byte, error) { return charset.Head(modelFS, "model/head.bin") }
