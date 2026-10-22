package alnum_test

import (
	"testing"

	"github.com/only9464/ddddocr-go/alnum"
)

// TestLoad checks that the digit-and-letter head loads and that its charset and
// height match the export.
//
// TestLoad 校验「数字 + 大小写字母」头能加载, 字符集/高度与导出时一致。
func TestLoad(t *testing.T) {
	engine, err := alnum.New()
	if err != nil {
		t.Skipf("the weights were never exported (alnum/model/head.bin): %v", err)
	}
	if got := len(engine.Charset()); got != 63 {
		t.Fatalf("len(Charset()) = %d, want 63", got)
	}
	if got := engine.Height(); got != 64 {
		t.Fatalf("Height() = %d, want 64", got)
	}
	if head, err := alnum.Head(); err != nil || len(head) == 0 {
		t.Fatalf("Head() returned %d bytes, err = %v", len(head), err)
	}
}
