package full_test

import (
	"testing"

	"github.com/only9464/ddddocr-go/full"
)

// TestLoad checks that the full charset head loads and that its charset and
// height match the export.
//
// TestLoad 校验全字符集头能加载, 字符集/高度与导出时一致。
func TestLoad(t *testing.T) {
	engine, err := full.New()
	if err != nil {
		t.Skipf("the weights were never exported (full/model/head.bin): %v", err)
	}
	if got := len(engine.Charset()); got != 8210 {
		t.Fatalf("len(Charset()) = %d, want 8210", got)
	}
	if got := engine.Height(); got != 64 {
		t.Fatalf("Height() = %d, want 64", got)
	}
	if head, err := full.Head(); err != nil || len(head) == 0 {
		t.Fatalf("Head() returned %d bytes, err = %v", len(head), err)
	}
}
