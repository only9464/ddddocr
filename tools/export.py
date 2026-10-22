#!/usr/bin/env python3
"""把 ddddocr 的 common.onnx 拆成「共享主干 + 分类头」，供纯 Go 直接执行。

为什么拆?
    模型 = 卷积 + BiLSTM(主干, 20.4 MB) + Gemm 分类头(1024 × 字符数)。
    全字符集 8210 类时分类头有 33.7 MB, 而多数场景只认数字或字母数字,
    所以主干只存一份, 分类头按字符集单独导出 —— 用哪个 import 哪个,
    Go 只把被 import 的包链接进二进制。

流程:
1. 解析 ONNX 图, 抽出一份线性前向程序(conv / swish / add / to_seq / lstm / gemm / argmax);
2. numpy 按程序重算, 与 onnxruntime 逐时间步对拍;
3. 按 --charset 裁剪 Gemm 行, 写出 backbone + head 两个 .bin;
4. 可选写出 Go 测试用的参考数据。

用法:
    pip install ddddocr onnx onnxruntime pillow numpy
    python tools/export.py --charset digits
    python tools/export.py --charset alnum
    python tools/export.py --charset full
    python tools/export.py --charset custom:0123456789 --head-out /tmp/head.bin

权重格式(小端):
    magic   8B   "OCRW1\\0\\0\\0"
    len     4B   uint32, JSON 头的字节数
    header  NB   UTF-8 JSON: {format, version, kind, height, charset, program, tensors}
    data    ...  各张量的 float32 原始字节, 按 offset 排列
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from PIL import Image

MAGIC = b"OCRW1\x00\x00\x00"
HEIGHT = 64
ROOT = Path(__file__).resolve().parents[1]

# 预设字符集。index 0 必须留空, 那是 CTC 的 blank。
PRESETS = {
    "digits": "0123456789",
    "alnum": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
}


# --------------------------------------------------------------------- 基础工具
def find_model(explicit: str | None) -> str:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SystemExit(f"找不到模型文件: {path}")
        return str(path)
    env = os.environ.get("DDDDOCR_MODEL")
    if env and Path(env).exists():
        return env
    for base in {Path(sys.prefix), Path(sys.executable).resolve().parent}:
        for candidate in base.rglob("ddddocr/common.onnx"):
            return str(candidate)
    raise SystemExit("找不到 ddddocr/common.onnx, 请 pip install ddddocr 或用 --model 指定")


def load_charset_json(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("charset") or data.get("CHARSET_BETA")
    if not isinstance(data, list) or not data:
        raise SystemExit(f"字符集 JSON 格式不对: {path}")
    return [str(v) for v in data]


def load_charset(explicit_json: str | None) -> list[str]:
    if explicit_json:
        return load_charset_json(explicit_json)
    try:
        from ddddocr.charsets import CHARSET_BETA  # type: ignore

        return list(CHARSET_BETA)
    except Exception:
        pass
    for base in {Path(sys.prefix), Path(sys.executable).resolve().parent}:
        for candidate in base.rglob("charsets_beta.json"):
            return load_charset_json(str(candidate))
    raise SystemExit("拿不到 CHARSET_BETA: 请 pip install ddddocr, 或用 --charset-json 指定")


def resolve_charset(spec: str, full_charset: list[str]) -> tuple[str, list[str]]:
    """把 --charset 解析成 (名字, keep_chars)。"""
    if spec == "full":
        chars = list(full_charset)
        if chars[0] != "":
            raise SystemExit("字符集第 0 项应该是空(CTC blank), 检查 CHARSET_BETA")
        return "full", chars

    if spec.startswith("custom:"):
        text = spec[len("custom:") :]
        wanted = [""]
        for ch in text:
            if ch not in wanted:
                wanted.append(ch)
        missing = [ch for ch in wanted if ch and ch not in full_charset]
        if missing:
            raise SystemExit(f"这些字符不在 ddddocr 字符集里: {''.join(missing)}")
        return "custom", wanted

    if spec not in PRESETS:
        raise SystemExit(f"未知的 --charset: {spec} (可用 digits / alnum / full / custom:xxx)")
    return spec, ["", *PRESETS[spec]]


def keep_indices(chars: list[str], full_charset: list[str]) -> list[int]:
    if len(chars) == len(full_charset) and all(a == b for a, b in zip(chars, full_charset)):
        return list(range(len(full_charset)))
    return [full_charset.index(ch) for ch in chars]


# --------------------------------------------------------------------- 抽取程序
def build_program(graph: onnx.GraphProto) -> list[dict]:
    inits = {i.name: i for i in graph.initializer}
    produced: dict[str, int] = {}
    program: list[dict] = []
    current: str | None = None

    sigmoid_in = {node.output[0]: node.input[0] for node in graph.node if node.op_type == "Sigmoid"}
    swish_sigmoids: set[str] = set()
    for node in graph.node:
        if node.op_type != "Mul":
            continue
        a, b = node.input[0], node.input[1]
        for x, s in ((a, b), (b, a)):
            if s in sigmoid_in and sigmoid_in[s] == x:
                swish_sigmoids.add(s)

    for node in graph.node:
        op = node.op_type
        if op == "Conv":
            assert node.input[0] == current or node.name.startswith("Conv_0"), (
                f"conv 输入不是当前流: {node.name} {node.input[0]} != {current}"
            )
            w, b = node.input[1], node.input[2]
            attrs = {a.name: a for a in node.attribute}
            strides = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]
            pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]
            kernel = tuple(numpy_helper.to_array(inits[w]).shape)
            program.append(
                {
                    "op": "conv",
                    "w": w,
                    "b": b,
                    "stride": strides[0],
                    "pad": pads[0],
                    "kernel": [kernel[2], kernel[3]],
                    "out": kernel[0],
                }
            )
            current = node.output[0]
        elif op == "Sigmoid":
            program.append({"op": "swish" if node.output[0] in swish_sigmoids else "sigmoid"})
            current = node.output[0]
        elif op == "Mul":
            a, b = node.input[0], node.input[1]
            fold = any(s in swish_sigmoids and sigmoid_in[s] == x for s, x in ((a, b), (b, a)))
            if fold:
                produced[node.output[0]] = len(program) - 1
                current = node.output[0]
                continue
            continue
        elif op == "Add":
            a, b = node.input[0], node.input[1]
            if a in produced and b == current:
                ref = produced[a]
            elif b in produced and a == current:
                ref = produced[b]
            else:
                raise SystemExit(f"Add 无法识别残差连接: {node.name} {a} {b} current={current}")
            program.append({"op": "add", "ref": ref})
            current = node.output[0]
        elif op == "Transpose":
            perm = list(a.ints for a in node.attribute if a.name == "perm")[0]
            if perm == [3, 0, 1, 2]:
                program.append({"op": "to_seq"})
                current = node.output[0]
            else:
                continue
        elif op == "LSTM":
            attrs = {a.name: a for a in node.attribute}
            direction = attrs["direction"].s.decode() if "direction" in attrs else "forward"
            program.append(
                {
                    "op": "lstm",
                    "w": node.input[1],
                    "r": node.input[2],
                    "b": node.input[3],
                    "hidden": attrs["hidden_size"].i,
                    "dirs": 2 if direction == "bidirectional" else 1,
                }
            )
            current = node.output[0]
        elif op == "Gemm":
            program.append({"op": "gemm", "w": node.input[1], "b": node.input[2]})
            current = node.output[0]
        elif op == "ArgMax":
            program.append({"op": "argmax"})
            current = node.output[0]
        elif op in {"Reshape", "Shape", "Constant", "Gather", "Unsqueeze", "Concat", "ConstantOfShape"}:
            continue
        else:
            raise SystemExit(f"未处理的算子: {op} ({node.name})")

        produced[node.output[0]] = len(program) - 1 if program else 0

    # 图里的 ArgMax 输出没被 graph output 使用(输出是 reshape 后的 logits),
    # 但识别要拿到每个时间步的类别下标, 所以在程序尾部显式补一个 argmax。
    program.append({"op": "argmax"})
    return program


# ------------------------------------------------------------------ numpy 执行器
def swish(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int, pad: int) -> np.ndarray:
    c, h, wd = x.shape
    o, _, kh, kw = w.shape
    if pad:
        xp = np.zeros((c, h + 2 * pad, wd + 2 * pad), dtype=np.float32)
        xp[:, pad : pad + h, pad : pad + wd] = x
    else:
        xp = x
    ho = (h + 2 * pad - kh) // stride + 1
    wo = (wd + 2 * pad - kw) // stride + 1
    out = np.zeros((o, ho, wo), dtype=np.float32)
    for ky in range(kh):
        for kx in range(kw):
            patch = xp[:, ky : ky + stride * (ho - 1) + 1 : stride, kx : kx + stride * (wo - 1) + 1 : stride]
            out += np.einsum("oc,chw->ohw", w[:, :, ky, kx], patch, optimize=True)
    out += b[:, None, None]
    return out


def lstm(seq: np.ndarray, w: np.ndarray, r: np.ndarray, b: np.ndarray, hidden: int) -> np.ndarray:
    dirs = w.shape[0]
    t = seq.shape[0]
    outs = np.zeros((t, dirs * hidden), dtype=np.float32)
    for d in range(dirs):
        wd, rd, bd = w[d], r[d], b[d]
        wb, rb = bd[: 4 * hidden], bd[4 * hidden :]
        h_prev = np.zeros(hidden, dtype=np.float32)
        c_prev = np.zeros(hidden, dtype=np.float32)
        steps = range(t) if d == 0 else range(t - 1, -1, -1)
        for step in steps:
            gates = wd @ seq[step] + rb + rd @ h_prev + wb
            i, o, f, c = np.split(gates, 4)
            i = 1.0 / (1.0 + np.exp(-i))
            o = 1.0 / (1.0 + np.exp(-o))
            f = 1.0 / (1.0 + np.exp(-f))
            g = np.tanh(c)
            c_prev = f * c_prev + i * g
            h_prev = o * np.tanh(c_prev)
            outs[step, d * hidden : (d + 1) * hidden] = h_prev
    return outs


def ctc(indices: list[int]) -> list[int]:
    out, prev = [], None
    for idx in indices:
        if idx != prev and idx != 0:
            out.append(idx)
        prev = idx
    return out


def run_program(program: list[dict], inits: dict[str, np.ndarray], x: np.ndarray, keep: list[int]) -> list[int]:
    cur = x[0]
    saved: list[np.ndarray] = []
    for step in program:
        op = step["op"]
        if op == "conv":
            cur = conv2d(cur, inits[step["w"]], inits[step["b"]], step["stride"], step["pad"])
        elif op == "swish":
            cur = swish(cur)
        elif op == "sigmoid":
            cur = 1.0 / (1.0 + np.exp(-cur))
        elif op == "add":
            cur = cur + saved[step["ref"]]
        elif op == "to_seq":
            c, h, wd = cur.shape
            cur = cur.transpose(2, 0, 1).reshape(wd, c * h)
        elif op == "lstm":
            cur = lstm(cur, inits[step["w"]], inits[step["r"]], inits[step["b"]], step["hidden"])
        elif op == "gemm":
            cur = cur @ inits[step["w"]][keep].T + inits[step["b"]][keep]
        elif op == "argmax":
            cur = np.argmax(cur, axis=-1).astype(np.int64).reshape(-1)
        else:
            raise SystemExit(f"未知 op: {op}")
        saved.append(cur)
    return [int(v) for v in cur.reshape(-1)]


def preprocess(path: str) -> np.ndarray:
    img = Image.open(path)
    width, height = img.size
    target_w = int(width * (HEIGHT / height))
    img = img.resize((target_w, HEIGHT), Image.LANCZOS).convert("L")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr[None, None, :, :]


# --------------------------------------------------------------------- 写权重文件
def dump(path: Path, header: dict, blobs: list[np.ndarray]) -> None:
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(np.uint32(len(header_bytes)).tobytes())
        fh.write(header_bytes)
        for arr in blobs:
            fh.write(arr.tobytes())
    total = sum(t["bytes"] for t in header["tensors"])
    print(f"写出 {path}: header {len(header_bytes)}B, 权重 {total / 1e6:.2f} MB, 张量 {len(blobs)} 个")


def pack(entries: list[tuple[str, np.ndarray]]) -> tuple[list[dict], list[np.ndarray], int]:
    tensors, blobs, offset = [], [], 0
    for name, arr in entries:
        arr = np.ascontiguousarray(arr.astype(np.float32))
        tensors.append({"name": name, "shape": list(arr.shape), "offset": offset, "bytes": int(arr.size * 4)})
        blobs.append(arr)
        offset += int(arr.size * 4)
    return tensors, blobs, offset


# ------------------------------------------------------------------------ 主流程
def main() -> None:
    parser = argparse.ArgumentParser(description="导出 ddddocr 模型为纯 Go 权重(主干 + 分类头)")
    parser.add_argument("--charset", default="digits",
                        help="digits / alnum / full / custom:0123abc  (默认 digits)")
    parser.add_argument("--model", help="common.onnx 路径, 默认从已安装的 ddddocr 里找")
    parser.add_argument("--charset-json", help="没装 ddddocr 时用这个 charsets_beta.json 取字符集")
    parser.add_argument("--backbone-out", default=str(ROOT / "backbone" / "model" / "backbone.bin"),
                        help="主干输出路径(所有字符集共用)")
    parser.add_argument("--head-out", help="分类头输出路径, 默认 <charset>/model/head.bin")
    parser.add_argument("--single-out", help="改成导出单文件(主干+头合一), 给不方便拆分的场景用")
    parser.add_argument("--testdata", help="参考数据目录, 默认 <charset>/testdata; 传空字符串跳过")
    parser.add_argument("--samples", help="验证码样图 glob, 默认 <testdata>/captcha*.jpg")
    parser.add_argument("--no-verify", action="store_true", help="跳过与 onnxruntime 的对拍")
    args = parser.parse_args()

    full_charset = load_charset(args.charset_json)
    name, keep_chars = resolve_charset(args.charset, full_charset)
    keep = keep_indices(keep_chars, full_charset)
    head_out = Path(args.head_out or (ROOT / name / "model" / "head.bin"))
    test_dir = None if args.testdata == "" else Path(args.testdata or (ROOT / name / "testdata"))
    model_path = find_model(args.model)

    print(f"模型: {model_path}")
    print(f"字符集: {name} ({len(keep_chars)} 类, 其中 blank 1 个)")
    if len(keep) <= 24:
        print(f"keep_index: {keep}")

    graph = onnx.load(model_path).graph
    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    program = build_program(graph)
    print(f"程序步数: {len(program)}")

    samples = glob.glob(args.samples) if args.samples else []
    if not samples and test_dir is not None:
        samples = sorted(glob.glob(str(test_dir / "captcha*.jpg")))
    verify = bool(samples) and not args.no_verify
    if not verify:
        print("提示: 没有样图或指定了 --no-verify, 跳过与 onnxruntime 的对拍"
              "(可用 --samples 'D:/pics/*.jpg' 指验证码图片)")

    refs = []
    if verify:
        import onnxruntime  # 只有要拿参考输出/对拍时才需要它

        sess = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        for path in samples:
            x = preprocess(path)
            ort = np.asarray(sess.run(None, {iname: x})[0])
            logits = ort[:, 0, :] if ort.ndim == 3 else ort
            ort_list = [int(v) for v in np.argmax(logits, axis=-1)]
            got = run_program(program, inits, x, list(range(len(full_charset))))
            if got != ort_list:
                print("  ort  :", ort_list)
                print("  numpy:", got)
                raise SystemExit(f"{os.path.basename(path)}: 程序与 onnxruntime 不一致")
            masked = [int(v) for v in np.argmax(logits[:, keep], axis=-1)]
            mine = run_program(program, inits, x, keep)
            if mine != masked:
                print("  masked ort:", masked)
                print("  mine      :", mine)
                raise SystemExit(f"{os.path.basename(path)}: 裁剪后的结果与 onnxruntime 不一致")
            print(f"{os.path.basename(path)}: 本字符集 → "
                  f"{''.join(keep_chars[i] for i in ctc(masked))!r} | "
                  f"全字符集 → {''.join(full_charset[i] for i in ctc(ort_list))!r}")
            refs.append({"name": os.path.basename(path), "path": path,
                         "input": x[0, 0].astype(np.float32), "indices": masked, "full": ort_list})

    # -------- 拆分: 主干(所有 conv/lstm) + 分类头(gemm 的 W/B) --------
    backbone_entries: list[tuple[str, np.ndarray]] = []
    head_entries: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    backbone_program = [dict(step) for step in program]
    for step in backbone_program:
        if step["op"] == "gemm":
            head_entries = [
                ("gemm_w", inits[step["w"]][keep]),
                ("gemm_b", inits[step["b"]][keep]),
            ]
            step.pop("w", None)
            step.pop("b", None)
            continue
        for key in ("w", "b", "r"):
            tname = step.get(key)
            if not tname or tname in seen:
                continue
            seen.add(tname)
            backbone_entries.append((tname, inits[tname]))
    if not head_entries:
        raise SystemExit("程序里没有 gemm 步, 无法拆出分类头")

    height = HEIGHT
    common = {"format": "ddddocr-go-weights", "version": 1, "height": height}

    if args.single_out:
        tensors, blobs, _ = pack(backbone_entries + head_entries)
        single_program = [dict(step) for step in program]
        for step in single_program:
            if step["op"] == "gemm":
                step["w"], step["b"] = "gemm_w", "gemm_b"
        header = {**common, "kind": "single", "charset": keep_chars,
                  "program": single_program, "tensors": tensors}
        dump(Path(args.single_out), header, blobs)
    else:
        tensors, blobs, _ = pack(backbone_entries)
        header = {**common, "kind": "backbone", "charset": [],
                  "program": backbone_program, "tensors": tensors}
        dump(Path(args.backbone_out), header, blobs)

        tensors, blobs, _ = pack(head_entries)
        header = {**common, "kind": "head", "charset": keep_chars,
                  "program": [], "tensors": tensors}
        dump(head_out, header, blobs)

    if test_dir is None or not refs:
        return

    # -------- Go 测试用参考数据 --------
    test_dir.mkdir(parents=True, exist_ok=True)
    flats, metas, off = [], [], 0
    for idx, r in enumerate(refs):
        flat = np.ascontiguousarray(r["input"]).reshape(-1)
        flats.append(flat)
        filename = f"captcha{idx}.jpg"
        metas.append({
            "name": r["name"],
            "file": filename,
            "height": int(r["input"].shape[0]),
            "width": int(r["input"].shape[1]),
            "input_offset": off,
            "input_floats": int(flat.size),
            "indices": r["indices"],
            "text": "".join(keep_chars[i] for i in ctc(r["indices"])),
            "full_text": "".join(full_charset[i] for i in ctc(r["full"])),
        })
        off += int(flat.size) * 4
        dst = test_dir / filename
        if os.path.abspath(r["path"]) != os.path.abspath(dst):
            shutil.copyfile(r["path"], dst)

    with open(test_dir / "inputs.f32", "wb") as fh:
        for flat in flats:
            fh.write(flat.tobytes())
    with open(test_dir / "refs.json", "w", encoding="utf-8") as fh:
        json.dump({"charset": keep_chars, "height": height, "samples": metas},
                  fh, ensure_ascii=False, indent=2)
    print(f"参考数据已写入 {test_dir}")


if __name__ == "__main__":
    main()
