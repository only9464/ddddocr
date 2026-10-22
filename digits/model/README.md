# 分类头目录

放导出好的 `head.bin`（45 KB，空白 + 0-9 共 11 类），由导出脚本生成：

```bash
python tools/export.py --charset digits
```

分类头只是 Gemm 的那一小块矩阵（11 × 1024 个 float32）；真正占体积的 20.4 MB 主干在
`backbone/model/backbone.bin`，所有字符集共用同一份。

本目录被 `//go:embed model` 整个嵌入，别往里放别的文件（比如 `--single-out` 的单文件
权重，那是 54 MB 的完整模型，放进来会让二进制凭空大一倍）。
