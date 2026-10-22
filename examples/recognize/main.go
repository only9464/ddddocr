// Command recognize prints the text of every image named on the command line,
// using the built-in digit model (a 45 KB classification head).
//
//	go run ./examples/recognize captcha0.jpg captcha1.jpg
//
// To use the letters-and-digits or the full charset instead, swap the import for
//
//	"github.com/only9464/ddddocr-go/alnum" // 0-9A-Za-z, 259 KB head
//	"github.com/only9464/ddddocr-go/full"  // 8210 classes, 33.7 MB head
//
// and change digits.New() to alnum.New() or full.New(): Go only links the head
// that gets imported, so the charsets you do not import are never embedded.
//
// 例子: 识别命令行给出的图片, 用内置的数字模型(分类头只有 45 KB)。
package main

import (
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/only9464/ddddocr-go/digits"
)

var errUsage = errors.New("usage: recognize <image> [more images...]")

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "recognize:", err)
		if errors.Is(err, errUsage) {
			os.Exit(2)
		}
		os.Exit(1)
	}
}

// run recognizes every file and prints one line per file. It only returns an
// error for problems that stop the whole run.
//
// run 识别给出的每个文件, 每个文件打印一行; 只有导致整个程序无法继续的问题才返回错误。
func run(files []string) error {
	if len(files) == 0 {
		return errUsage
	}
	engine, err := digits.New()
	if err != nil {
		return fmt.Errorf("loading the model (run python tools/export.py --charset digits first): %w", err)
	}
	fmt.Printf("charset: %d classes, input height: %d\n", len(engine.Charset()), engine.Height())
	for _, name := range files {
		start := time.Now()
		text, err := engine.RecognizeFile(name)
		if err != nil {
			fmt.Printf("%-24s error: %v\n", name, err)
			continue
		}
		fmt.Printf("%-24s -> %-8s (%s)\n", name, text, time.Since(start).Round(time.Millisecond))
	}
	return nil
}
