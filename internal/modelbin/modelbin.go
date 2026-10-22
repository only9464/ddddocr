// Package modelbin reads the weight files embedded in a sub-package's model
// directory.
//
// Every charset package embeds its model directory with //go:embed model. The
// directory also holds README files, so lookups go by exact file name instead
// of guessing "the first .bin in the directory".
//
// Package modelbin 负责从子包内嵌的目录里读出权重文件。每个字符集包都用
// //go:embed model 内嵌自己的 model 目录, 目录里除权重外还有说明文档, 所以读取时按固定文件名取, 不做"取目录里第一个 .bin"这类猜测。
package modelbin

import (
	"fmt"
	"io/fs"
)

// Read returns the contents of name inside fsys, for example "model/head.bin".
//
// A missing file yields an error wrapping fs.ErrNotExist; an empty file yields
// its own error, because an empty weight blob is never valid.
//
// Read 读取 fsys 里的 name(如 "model/head.bin")。文件不存在时返回包装了
// fs.ErrNotExist 的错误, 文件存在但为空时也报错。
func Read(fsys fs.FS, name string) ([]byte, error) {
	data, err := fs.ReadFile(fsys, name)
	if err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("modelbin: %s is empty", name)
	}
	return data, nil
}

// Size returns the size of name in bytes, or 0 when it cannot be determined
// (for example when the file does not exist).
//
// Size 返回 fsys 里 name 的字节数, 取不到时(比如文件不存在) 返回 0。
func Size(fsys fs.FS, name string) int64 {
	info, err := fs.Stat(fsys, name)
	if err != nil {
		return 0
	}
	return info.Size()
}
