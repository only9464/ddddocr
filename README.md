# ddddocr-go

**中文** | [English](README.en.md)

**注意：本仓库中所有代码及文档说明全部由deepseek v4.1 flash编写并上传到GitHub仓库，如有侵权，请联系删除**
把 [ddddocr](https://github.com/sml2h3/ddddocr) 的识别模型（`common.onnx`）用**纯 Go** 重新实现，权重在**编译期嵌入**二进制。

- 不需要 cgo、不需要 onnxruntime 动态库、不需要 Python
- 唯一的第三方依赖是纯 Go 的 [`golang.org/x/image`](https://pkg.go.dev/golang.org/x/image)，只用来解标准库没有的图片格式（BMP/TIFF/WebP），约 0.3 MB
- 运行时不需要任何模型文件，交叉编译 / distroless / scratch 镜像都能直接用
- 权重拆成「20.4 MB 主干 + 很小的分类头」，**用哪个字符集才嵌入哪个头**，只认数字的项目不会白白带上 33.7 MB 的全字符集

```go
import "github.com/only9464/ddddocr-go/digits"

engine, err := digits.New()                        // 主干 20.4 MB + 数字头 45 KB，已 embed
if err != nil {
    log.Fatal(err)
}
text, err := engine.RecognizeFile("captcha.jpg")   // "4003"
```

## 体积：按需嵌入

模型的卷积 + BiLSTM 部分（**主干**）约占 20.4 MB，所有字符集完全一样；真正和字符集相关的只有最后那个
Gemm 分类头（`1024 × 字符数` 个 float32）。所以权重被拆成两份。

二进制大小是实测的（windows/amd64、Go 1.26，只 import 一个字符集并调用 `New()` 的空程序；
空 `main` 是 1.8 MB，其中图片解码器 BMP/TIFF/WebP 约占 0.3 MB）：

| 包 | 字符集 | 头体积 | 编出来的二进制 |
| --- | --- | --- | --- |
| `backbone` | —（主干，被下面三个包依赖） | 20.4 MB | — |
| `digits` | blank + `0-9` 共 11 类 | 45 KB | 24.4 MB |
| `alnum` | blank + `0-9A-Za-z` 共 63 类 | 259 KB | 24.6 MB |
| `full` | ddddocr `CHARSET_BETA` 8210 类 | 33.7 MB | 58.0 MB |

Go 的链接器只会把**被真正用到的包和数据**放进二进制，所以换字符集只是换一行 import：

```go
import "github.com/only9464/ddddocr-go/digits"   // 只会嵌入 backbone + 数字头
```

只 import `digits` 时，`alnum`/`full` 的头根本不会被打包——不需要 build tag、不需要自己裁剪。
（注意只有 `import _ "..."` 而从不调用 `New()` 时，链接器会把没人引用的权重数据整体丢掉。）

需要自己的字符集（比如「0-9 + 几个汉字」）时，导出一个自定义头再和主干组合：

```bash
python tools/export.py --charset custom:0123456789加油华为 --head-out ./mymodel/head.bin
```

```go
import (
    ddddocr "github.com/only9464/ddddocr-go"
    "github.com/only9464/ddddocr-go/backbone"
)

head, _ := os.ReadFile("mymodel/head.bin")
engine, err := ddddocr.NewFromSet(must(backbone.Bytes()), head)
```

## 项目结构

```
ddddocr-go/
├── ddddocr 引擎（本包）
│   ├── recognizer.go    公共 API：New / NewFromSet / Recognize* / Classification* / Charset / Height
│   ├── weights.go       权重容器解析（魔数 + JSON 头 + float32 数据）、挂分类头
│   ├── forward.go       前向算子：conv / swish / add / to_seq / lstm / gemm / argmax
│   └── preprocess.go    图片解码（JPEG/PNG/GIF/BMP/TIFF/WebP）→ Lanczos → 灰度 → [0,1]
├── backbone/            内嵌 20.4 MB 共享主干的包
├── digits/ alnum/ full/ 内嵌各自分类头的包（同一套模板，逻辑在 internal/charset）
├── internal/charset/    字符集包的公共逻辑：Once 缓存 + 主干/头组合
├── internal/modelbin/   从 embed.FS 里读权重文件的小工具
├── examples/recognize/  命令行示例
├── tools/export.py      ONNX → 权重容器的导出脚本（开发时用，不进入最终程序）
├── .gitattributes       统一换行符为 LF，权重/图片不被当成文本
└── */model/, */testdata/ 权重与对拍参考数据
```

## 为什么不用 `onnxruntime_go`

GitHub 上现有的几个 "ddddocr Go 版"（`yangbin1322/go-ddddocr`、`FeilongTest/go-ddddocr`、
`Changbaiqi/ddddocr-go`、`KazemiyaMione/ddddocr-go`、`tensafe/goddddocr`）都封装了
`github.com/yalue/onnxruntime_go`，本质仍然是 **cgo + 运行时加载 `onnxruntime.dll`**，
外加一个 54 MB 的 `.onnx` 文件。本项目把 ONNX 图离线"编译"成一条线性的前向程序
（conv / swish / add / to_seq / lstm / gemm / argmax），Go 侧只需要实现这几个算子：

| 组成 | 本项目 | onnxruntime_go 方案 |
| --- | --- | --- |
| cgo | 不需要 | 需要 |
| 运行时依赖 | 无 | onnxruntime 动态库 |
| 第三方库 | 只有 `golang.org/x/image`（纯 Go，解图片格式） | cgo 封装 + 动态库 |
| 模型文件 | 编译进二进制 | 需要随程序分发 `.onnx` |
| 交叉编译 | `GOOS`/`GOARCH` 直接出 | 需要各平台 dll/so/dylib |
| 体积 | 24.4 MB（digits）/ 58.0 MB（full） | ≈20 MB 库 + 54 MB 模型 |

## 安装

```bash
go get github.com/only9464/ddddocr-go/digits   # 数字验证码（推荐）
go get github.com/only9464/ddddocr-go/alnum    # 字母数字
go get github.com/only9464/ddddocr-go/full     # 8210 字符全字符集
```

权重是构建产物，仓库里可以选择性提交（`backbone.bin` 20.4 MB、分类头 45 KB / 259 KB / 33.7 MB，
都没超过 GitHub 100 MB 单文件上限，不需要 LFS）。没提交时 `go build` 依然能过，
只是运行期返回错误——先按下一节导出即可。

`go get` 会顺带拉下唯一的依赖 `golang.org/x/image`（纯 Go，没有 cgo）。

## 重新导出权重

```bash
# onnx + ddddocr 是解析模型必需的；onnxruntime 只在"与参考实现对拍"时用到
pip install ddddocr onnx pillow numpy onnxruntime

# 主干只需导一次（所有字符集共用）
python tools/export.py --charset digits
#   → backbone/model/backbone.bin  (20.4 MB)
#   → digits/model/head.bin        (45 KB)
#   → digits/testdata/             (参考数据，给 go test 用)

python tools/export.py --charset alnum         # → alnum/model/head.bin (259 KB)
python tools/export.py --charset full          # → full/model/head.bin (33.7 MB)

# 想自己提供验证码样图（脚本会用 onnxruntime 对拍）
python tools/export.py --charset digits --samples "path/to/captcha*.jpg"

# 不拆分的单文件模式（一个文件里同时含主干和头，体积翻倍，只用于兼容）
python tools/export.py --charset digits --single-out /tmp/digits.bin
```

脚本流程：解析 ONNX → 抽出前向程序 → 用 numpy 重算并与 onnxruntime **逐时间步对拍**
（对不上直接失败退出）→ 按 `--charset` 裁剪分类头 → 写主干 + 头 + `testdata/` 参考数据。
没有样图或加 `--no-verify` 时会跳过对拍（也就不需要 onnxruntime），只写权重。
`python tools/export.py --help` 可看全部参数。

`--single-out` / `--head-out` 请写到 `*/model/` 之外：这些目录被 `//go:embed model` 整个嵌入，
多放一个文件（尤其是 54 MB 的单文件权重）二进制就会凭空变大。

## 用法

### 从 Python 版迁移：几乎不用改

Python 版的常见写法是「构造一次 → `classification` 一次」，到了 Go 里是这样的：

```python
# Python
import ddddocr

ocr = ddddocr.DdddOcr(beta=True)                     # 8210 字符的 beta 模型
text = ocr.classification(open("captcha.jpg", "rb").read())
```

```go
// Go
engine, err := full.New()                            // beta 模型的等价物
if err != nil {
    log.Fatal(err)
}
text, err := engine.Classification(jpegBytes)         // 和 Recognize 完全等价
```

对应关系：

| Python | Go | 说明 |
| --- | --- | --- |
| `ddddocr.DdddOcr(beta=True)` | `full.New()` | 同为 `common.onnx` + `CHARSET_BETA`（8210 类），权重和字符集一一对应 |
| `ddddocr.DdddOcr()`（默认） | 无 | 默认用的是 `common_old.onnx`（旧字符集），本项目没导出它；需要时用 `--model common_old.onnx` 自己导一份 |
| `ocr.classification(bytes)` | `engine.Classification(bytes)` 或 `engine.Recognize(bytes)` | 同名别名的实现就是 Recognize，行为一致 |
| `ocr.classification(path)` | `engine.RecognizeFile(path)` | Python 里路径字符串也能直接传，Go 用独立方法 |
| `ocr.classification(PIL.Image)` | `engine.RecognizeImage(img)` | 入参类型是 `image.Image` |
| `ocr.classification(bytes, probability=True)` | `engine.ClassificationProb(bytes)` | 返回文本 + 每个字符的置信度；不返「时间步 × 字符集」整张矩阵 |
| `ocr.set_ranges("0123456789")` | `digits.New()` | 按字符集裁剪分类头，比运行期过滤更快也更省内存 |
| `ocr.get_charset()` | `engine.Charset()` | 返回值下标 0 是 CTC 的 blank |
| `show_ad` / `use_gpu` / `switch_device` | — | 纯 Go 库没有广告，也没有 GPU |
| `png_fix=True` / `color_filter_*` | — | 未实现；透明背景的 PNG 可以先用 `image/draw` 铺白底再识别 |
| `ocr.detection()` / `slide_match()` | — | 目标检测和滑块匹配没有移植 |

和 Python 版同一个套路：**构造一次，全局复用**。区别只在 Go 里构造和识别都会返回 `error`
（Python 是抛异常），各自 `if err != nil` 处理即可。

### 数字验证码

```go
import "github.com/only9464/ddddocr-go/digits"

engine, err := digits.New()
text, err := engine.RecognizeFile("captcha.jpg")     // 本地文件
text, err := engine.Recognize(jpegBytes)             // []byte（JPEG/PNG/GIF/BMP/TIFF/WebP）
text, err := engine.RecognizeImage(img)              // image.Image
```

### 字母数字 / 全字符集

```go
import "github.com/only9464/ddddocr-go/alnum"

engine, err := alnum.New()
text, err := engine.Recognize(data)
```

### 自带权重（自制头或单文件权重）

```go
raw, _ := os.ReadFile("my-weights.bin")
engine, err := ddddocr.New(raw)      // 单文件权重（--single-out 产物）
```

格式就是导出脚本写的容器格式：8 字节魔数 `OCRW1\0\0\0` + uint32 头长度 + JSON 头 + float32 数据。

### 低级接口

```go
idx, _ := engine.RecognizeIndices(data)  // 每个时间步的类别下标（CTC 未去重）
text, chars, _ := engine.ClassificationProb(data)
                                         // chars: []ddddocr.CharProb{{Char:"4", Prob:0.99}, ...}
engine.Charset()                         // 分类头字符集，下标 0 是 blank
engine.Height()                          // 输入高度（common 模型 = 64）
```

`Recognizer` 加载后是只读的：权重不写、每次识别用的中间缓冲都是调用内新分配的，
所以**同一个实例可以被任意多个 goroutine 并发调用**，不需要再加锁。

实测（i5-12500H，12 核 16 线程，windows/amd64，70×30 的 4 位数字验证码，含预处理）：

| 场景 | 耗时 |
| --- | --- |
| 单张，占满多核（前向内部按输出通道并行） | ≈ 91 ms |
| 单张，`GOMAXPROCS=1`（单核） | ≈ 255 ms |
| 16 路并发时的平均吞吐 | ≈ 41 ms/张 |
| 单次识别内存分配 | ≈ 6.9 MB / 5.3k 次 |

## 测试

```bash
go test ./...                    # 没有权重时相关用例自动 skip
go test -race ./...              # 并发用例（识别器无锁共享）
go test -bench=. ./digits        # 单张识别基准
```

- `TestRecognizeMatchesReference`：用导出时保存的预处理输入，把 Go 前向的每个时间步下标与
  onnxruntime 逐一对拍（算子改错立刻发现）。
- `TestPreprocessMatchesReference`：把 Go 的「解码 + Lanczos + 灰度」结果与 PIL 管的张量对齐，
  容差 4/255（实测偏差 ≤ 2/255），保证预处理不是"看起来对"。
- `TestRecognizeImageFormats`：同一张验证码存成 jpg/png/bmp/tiff/webp，结果必须一致
  （顺便守住 BMP/TIFF/WebP 的解码器注册）。
- `TestRecognizeSampleImages`：端到端跑样图，期望值来自 `refs.json`，不硬编码。
- `TestClassification*`：Python 风格的 `Classification` 别名、带置信度的 `ClassificationProb`
  与 `Recognize` 结果一致（逐字符概率在 (0,1] 内、拼回去就是文本）。
- `TestRecognizeConcurrent`：多 goroutine 同时识别同一实例，配合 `-race`。

仓库带 `.gitattributes`（`* text=auto eol=lf` + 权重的 `binary` 标记），Windows 上不需要再折腾
`core.autocrlf`；CI 会同时检查 `gofmt`。

## 已知限制

- 只移植了**识别**模型；`ddddocr` 的目标检测（`common_det.onnx`）与滑块匹配没有移植。
- Python 版的 `png_fix` / `color_filter_*` / `set_ranges` 这些后处理开关没做：字符集限制用
  「import 对应的头」代替（`digits`/`alnum`/`--charset custom`），透明背景自己铺底。
- 分类头是 ddddocr 的 `CHARSET_BETA`（8210 类），不是 `CHARSET_OLD`；用错字符集会解出乱码。
- 图片格式只认 JPEG/PNG/GIF/BMP/TIFF/WebP；别的格式（ICO、JPEG 2000…）可以用
  `image.RegisterFormat` 自己注册，`Recognize` 走的就是标准库的 `image.Decode`。
- 纯标量 float32，没有 SIMD：同一张图 Python + onnxruntime 约 6 ms，本项目单核约 255 ms。
  卷积已按「权重只读一次、输出行常驻 L1」的顺序实现，仍受限于 Go 不会自动向量化；
  追求极限可以上 cgo/汇编，但那会破坏"无依赖交叉编译"这个卖点。
- 权重是 float32。想要更小可以做 int8 量化（约 1/4 体积），主干与头需要一起量化。
- 批量识别直接开 goroutine 即可（实例并发安全），无需再包一层锁。

## 发布到自己的仓库 / fork

```bash
sed -i 's#github.com/only9464/ddddocr-go#github.com/<你的用户名>/ddddocr-go#' \
    go.mod $(grep -rl 'github.com/only9464/ddddocr-go' --include='*.go' .)
git init && git add . && git commit -m "ddddocr-go: pure Go ddddocr inference"
git push -u origin main && git tag v0.1.0 && git push origin v0.1.0
```

之后在别的项目里：`go get github.com/<你的用户名>/ddddocr-go/digits@v0.1.0`。

## 许可

本项目代码 MIT。模型权重与字符集来自 [sml2h3/ddddocr](https://github.com/sml2h3/ddddocr)（MIT），
详见 [NOTICE](NOTICE)。
