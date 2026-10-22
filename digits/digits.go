// Package digits recognizes digits only: the classification head keeps blank
// plus 0-9, 11 classes in total.
//
// Size: the shared 20.4 MB backbone plus a 45 KB head. It is the model to use
// for purely numeric captchas (4 or 6 digits and so on).
//
//	engine, err := digits.New()
//	text, err := engine.RecognizeFile("captcha.jpg")
//
// Package digits 是「只认数字」的 ddddocr 模型: 分类头只保留空白 + 0-9 共 11 类。
// 体积: backbone 20.4 MB(所有字符集共用) + 本包的头 45 KB, 适合纯数字验证码。
package digits

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

// New returns the recognizer for the built-in digit model. The weights are
// loaded once per process and the result may be used by many goroutines.
//
// New 返回内置数字模型的识别器。权重在整个进程内只加载一次, 返回的识别器可并发使用。
func New() (*ddddocr.Recognizer, error) { return engine.Load(modelFS, "model/head.bin") }

// Head returns the embedded classification head; it reports ddddocr.ErrNoHead
// when the head has not been exported.
//
// Head 返回内嵌的分类头字节; 还没导出时返回 ddddocr.ErrNoHead。
func Head() ([]byte, error) { return charset.Head(modelFS, "model/head.bin") }
