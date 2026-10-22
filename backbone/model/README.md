# 主干权重目录

放导出好的 `backbone.bin`（20.4 MB），由仓库根目录的导出脚本生成：

```bash
python tools/export.py --charset digits   # 会顺带写出 backbone + digits 头
```

`backbone.bin` 是所有字符集共用的那 20.4 MB（卷积 + BiLSTM），
分类头在各自的字符集包里（如 `digits/model/head.bin`，45 KB）。

公开仓库建议把这个文件一并提交（远小于 GitHub 100 MB 单文件限制）。

本目录被 `//go:embed model` 整个嵌入，别往里放别的文件——放一个 50 MB 的中间产物，
所有 import `backbone` 的程序就都多 50 MB。
