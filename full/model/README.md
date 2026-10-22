# 分类头目录

放导出好的 `head.bin`（约 33.7 MB，8210 类全字符集），由导出脚本生成：

```bash
python tools/export.py --charset full
```

注意体积分布：主干 20.4 MB + 这个头 33.7 MB。如果只需要数字或字母数字，用 `digits`
（45 KB）或 `alnum`（259 KB）的头就够，省下 33.7 MB。

需要别的字符组合（比如「0-9 + 几个汉字」）可以自己导一个小头：

```bash
python tools/export.py --charset custom:0123456789东大讲座 --head-out my-head.bin
```

`--head-out` 不要指回本目录：`//go:embed model` 会把目录里的**所有**文件打进二进制，
多放一个 .bin 就直接多几 MB。
