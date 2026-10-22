// Package charset loads and caches "shared backbone + one classification head"
// pairs for the digits, alnum and full packages, which differ only in charset.
//
// Package charset 把「共享主干 + 某个分类头」的加载与缓存逻辑收在一处, 供 digits / alnum / full 这些字符集包复用 —— 它们除了字符集之外完全一样。
package charset

import (
	"io/fs"
	"sync"

	ddddocr "github.com/only9464/ddddocr-go"
	"github.com/only9464/ddddocr-go/backbone"
	"github.com/only9464/ddddocr-go/internal/modelbin"
)

// Engine caches the recognizer built for one charset package: the first Load
// wins and every later call returns the same instance. Loading weights is a
// one-off read-only cost, and sync.Once keeps it concurrency-safe.
//
// Engine 记住某个字符集包第一次构造出来的识别器, 之后一直返回同一个实例。
// 加载权重是一次性的只读开销, 用 sync.Once 兼顾并发安全。
type Engine struct {
	once sync.Once
	rec  *ddddocr.Recognizer
	err  error
}

// Load builds the recognizer from the embedded shared backbone plus the
// classification head at name inside fsys, then caches the result on e.
//
// It returns ddddocr.ErrNoWeights when the backbone was never exported and
// ddddocr.ErrNoHead when the head is missing.
//
// Load 用内嵌的共享主干加上 fsys 里 name 指向的分类头构造识别器, 结果缓存在 e 上。
// 主干没导出时返回 ddddocr.ErrNoWeights, 分类头缺失时返回 ddddocr.ErrNoHead。
func (e *Engine) Load(fsys fs.FS, name string) (*ddddocr.Recognizer, error) {
	e.once.Do(func() {
		main, err := backbone.Bytes()
		if err != nil {
			e.err = ddddocr.ErrNoWeights
			return
		}
		head, err := modelbin.Read(fsys, name)
		if err != nil {
			e.err = ddddocr.ErrNoHead
			return
		}
		e.rec, e.err = ddddocr.NewFromSet(main, head)
	})
	return e.rec, e.err
}

// Head returns the raw classification head at name inside fsys, for callers
// that want to assemble the weights themselves.
//
// Head 返回 fsys 里 name 指向的分类头字节, 给想自己组合权重的调用方用。
func Head(fsys fs.FS, name string) ([]byte, error) {
	head, err := modelbin.Read(fsys, name)
	if err != nil {
		return nil, ddddocr.ErrNoHead
	}
	return head, nil
}
