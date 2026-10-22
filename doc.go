// Package ddddocr runs the ddddocr "common" model (CTC text recognition) in
// pure Go.
//
// Highlights:
//   - pure Go: no cgo, no onnxruntime, no Python. The only external dependency
//     is golang.org/x/image, used solely to decode the image formats the
//     standard library lacks (BMP/TIFF/WebP);
//   - the weights are embedded into the executable, so nothing has to be
//     downloaded or shipped alongside it;
//   - the weights are split into a 20.4 MB shared backbone plus a small
//     classification head per charset, so importing one charset only embeds
//     that charset's head.
//
// Supported image formats: JPEG, PNG and GIF from the standard library, plus
// BMP, TIFF and WebP from x/image.
//
// This package is only the engine and carries no weights. For the quickest
// start import one of the charset packages:
//
//	github.com/only9464/ddddocr-go/digits // 0-9, 45 KB head
//	github.com/only9464/ddddocr-go/alnum  // 0-9A-Za-z, 259 KB head
//	github.com/only9464/ddddocr-go/full   // 8210 classes, 33.7 MB head
//
// Minimal usage:
//
//	import "github.com/only9464/ddddocr-go/digits"
//
//	engine, err := digits.New()
//	if err != nil {
//		log.Fatal(err)
//	}
//	text, err := engine.RecognizeFile("captcha.jpg")
//	if err != nil {
//		log.Fatal(err)
//	}
//	fmt.Println(text)
//
// Custom charset: export your own head with tools/export.py --charset
// custom:0123abc, then combine it with NewFromSet(backbone.Bytes(), yourHead).
//
// When porting from the Python package, engine.Classification(data) behaves
// exactly like Recognize and matches ddddocr.DdddOcr.classification; use
// ClassificationProb to get per-character confidence.
//
// The weight format is produced by tools/export.py from ddddocr's common.onnx;
// see the README.
//
// Package ddddocr 用纯 Go 跑 ddddocr 的 common 模型(CTC 文本识别)。
//
// 特点: 纯 Go, 不需要 cgo、不需要 onnxruntime、不需要 Python。
// 唯一的外部依赖是 golang.org/x/image, 只用来解标准库没有的图片格式(BMP/TIFF/WebP)。
// 权重以二进制形式 embed 进可执行文件, 运行时不需要任何模型文件。
// 权重拆成「20.4 MB 主干 + 很小的分类头」: 主干由 backbone 包提供, 分类头按字符集分包,
// 因此 import 哪个字符集就只嵌入哪个字符集的头。
//
// 支持的图片格式: JPEG/PNG/GIF (标准库) 以及 BMP/TIFF/WebP (x/image)。
//
// 本包只是引擎, 不含权重。开箱即用请 import 上面列出的子包之一。
//
// 自定义字符集: 用 tools/export.py --charset custom:0123abc 导出自己的头,
// 再 NewFromSet(backbone.Bytes(), yourHead) 组合即可。
//
// 从 Python 版 ddddocr 迁过来时, engine.Classification(data) 与 Recognize 完全等价,
// 方法名和 ddddocr.DdddOcr.classification 对齐; 带置信度用 ClassificationProb。
//
// 权重格式由 tools/export.py 从 ddddocr 的 common.onnx 导出, 见 README。
package ddddocr
