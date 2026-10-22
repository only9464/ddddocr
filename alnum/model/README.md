# 分类头目录

放导出好的 `head.bin`（259 KB，空白 + 0-9 + A-Z + a-z 共 63 类），由导出脚本生成：

```bash
python tools/export.py --charset alnum
```

分类头只是 Gemm 的那一小块矩阵，主干（20.4 MB）在 `backbone/model/backbone.bin`，多个
字符集共用同一份。想省体积就只 import 需要的那个字符集包。

本目录被 `//go:embed model` 整个嵌入，别往里放别的文件——多一个 .bin 就多几 MB 二进制。
