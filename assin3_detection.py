#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Assignment 3-2 -- Object detection on two images with two confidence thresholds.

基于课件 Lecture 4 "Coding Your Own Object Detection Program" 的范例改造：

  1) 把 /dev/video0 摄像头输入改成静态图片输入          (Tips 1)
  2) 阈值可配置，一条命令跑完 2 张图 x 2 个阈值共 4 组实验
  3) 逐条输出 ClassID / Confidence / Left / Top / Right / Bottom /
     Width / Height / Area / Center，并可用 --class 只看某一类别 (Tips 2)
  4) 保存带检测框的图片 (detected image) 和结果文本，直接贴进报告

典型用法（一次跑完作业要求的全部 4 组实验）：

    python3 assin3_detection.py \
        --input image1.jpg image2.jpg \
        --threshold 0.3 0.7 \
        --outdir results

只跑一组也可以：

    python3 assin3_detection.py --input image1.jpg --threshold 0.3

只打印某一类的结果（例如人）：

    python3 assin3_detection.py -i image1.jpg -t 0.3 -c person

适配环境：Jetson Nano 2GB / JetPack 4.6.1 (L4T R32.7.1) /
jetson-inference 的 L4T-R32.7.1 分支。API 已对照该分支源码核对。
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# 1. 导入 jetson-inference / jetson-utils
#
#    课件这里有个坑：它注释掉了旧写法
#        from jetson_inference import detectNet
#        from jetson_utils import videoSource, videoOutput
#    改成了
#        import jetson.inference
#        import jetson.utils
#    但下面仍然直接调用裸函数名 detectNet(...) / videoSource(...)，
#    照抄会报 NameError: name 'detectNet' is not defined。
#
#    正确写法是带命名空间调用：jetson.inference.detectNet(...)。
#    下面同时兼容新旧两种模块名，保证两种安装方式都能跑。
# ---------------------------------------------------------------------------
try:
    import jetson.inference as ji
    import jetson.utils as ju
except ImportError:
    try:
        import jetson_inference as ji
        import jetson_utils as ju
    except ImportError:
        sys.exit(
            "找不到 jetson-inference / jetson-utils，请先确认：\n"
            "  1) 已刷 JetPack 4.6.x（Nano 2GB 最高支持 4.6.1 / L4T R32.7.1）\n"
            "  2) 已编译安装 jetson-inference 的 L4T-R32.7.1 分支\n"
            "  3) 使用 python3 运行（Ubuntu 18.04 自带 Python 3.6）"
        )


def get_class_name(net, class_id):
    """把 ClassID 转成可读的类别名；不同版本方法名略有差异，故做兜底。"""
    for name in ("GetClassDesc", "GetClassLabel", "GetClassSynset"):
        fn = getattr(net, name, None)
        if fn is not None:
            try:
                return str(fn(class_id))
            except Exception:
                pass
    return "class_%d" % class_id


def describe(net, det):
    """取出作业要求的 10 个字段（另加类别名），老版本缺属性时自行计算。"""
    left, top = float(det.Left), float(det.Top)
    right, bottom = float(det.Right), float(det.Bottom)
    width = float(getattr(det, "Width", right - left))
    height = float(getattr(det, "Height", bottom - top))
    area = float(getattr(det, "Area", width * height))
    center = getattr(det, "Center", ((left + right) / 2.0, (top + bottom) / 2.0))
    return {
        "ClassID": int(det.ClassID),
        "ClassName": get_class_name(net, det.ClassID),
        "Confidence": float(det.Confidence),
        "Left": left,
        "Top": top,
        "Right": right,
        "Bottom": bottom,
        "Width": width,
        "Height": height,
        "Area": area,
        "Center": center,
    }


FIELDS = ("ClassID", "ClassName", "Confidence", "Left", "Top", "Right",
          "Bottom", "Width", "Height", "Area", "Center")


def format_line(row):
    parts = []
    for key in FIELDS:
        value = row[key]
        parts.append("%s=%s" % (key, "%.3f" % value if isinstance(value, float)
                                else str(value)))
    return "  ".join(parts)


def process_one(net, image_path, threshold, outdir, class_filter):
    """处理 (一张图片, 一个阈值) 的一组实验。"""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    tag = "%s_t%.2f" % (stem, threshold)
    img_out = os.path.join(outdir, tag + ".jpg")
    txt_out = os.path.join(outdir, tag + ".txt")

    # 每个组合都必须重新读图：net.Detect() 会直接在 img 上画框，
    # 复用上一组的 img 会导致两次实验的框叠加在同一张图上。
    img = ju.loadImage(image_path)

    # 模型只加载一次，用 SetThreshold 切换阈值（2GB 内存友好，也更快）。
    net.SetThreshold(threshold)

    # overlay 默认就是 box,labels,conf：画框 + 类别名 + 置信度
    detections = net.Detect(img, overlay="box,labels,conf")

    # 保存带检测框的图片，即作业要求上传的 detected image
    ju.saveImage(img_out, img)

    rows = [describe(net, d) for d in detections]
    if class_filter:
        rows = [r for r in rows
                if r["ClassName"].lower() == class_filter.lower()
                or str(r["ClassID"]) == class_filter]

    lines = [
        "Image       : %s" % image_path,
        "Threshold   : %.2f" % threshold,
        "Total detections (all classes) : %d" % len(detections),
        "Shown detections (filter=%s)   : %d" % (class_filter or "ALL", len(rows)),
        "Detected image : %s" % img_out,
        "-" * 78,
    ]
    for i, row in enumerate(rows):
        lines.append("[%d] %s" % (i + 1, format_line(row)))
    lines.append("-" * 78)
    report = "\n".join(lines)

    print("\n" + report)
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("[saved] %s , %s" % (img_out, txt_out))
    return len(detections)


def main():
    parser = argparse.ArgumentParser(
        description="Assignment 3-2: detect objects in images at several thresholds",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--input", nargs="+", required=True,
                        help="一张或多张输入图片路径")
    parser.add_argument("-t", "--threshold", nargs="+", type=float,
                        default=[0.5],
                        help="一个或多个置信度阈值，例如 0.3 0.7（一个小于 0.5，一个大于 0.5）")
    parser.add_argument("-o", "--outdir", default="results",
                        help="带框图片与结果文本的保存目录（默认 results/）")
    parser.add_argument("-n", "--network", default="ssd-mobilenet-v2",
                        help="模型名（默认 ssd-mobilenet-v2，91 类 COCO）")
    parser.add_argument("-c", "--class", dest="class_filter", default="",
                        help="只打印某一类的结果，例如 person 或 1（默认打印全部）")
    opt = parser.parse_args()

    for path in opt.input:
        if not os.path.isfile(path):
            sys.exit("找不到输入图片：%s" % path)
    for th in opt.threshold:
        if not 0.0 < th < 1.0:
            sys.exit("阈值必须在 (0, 1) 之间：%s" % th)

    os.makedirs(opt.outdir, exist_ok=True)

    # 模型只创建一次：TensorRT 首次运行会构建引擎，在 2GB 上要等几分钟，
    # 重复创建既慢又吃内存。
    print("[info] 正在加载模型 %s ..." % opt.network)
    try:
        net = ji.detectNet(opt.network, threshold=opt.threshold[0])
    except TypeError:
        # 个别版本要求按位置传 argv（传空列表，避免把本脚本的参数喂给网络）
        net = ji.detectNet(opt.network, [], opt.threshold[0])
    print("[info] 模型加载完成，类别数：%d" % net.GetNumClasses())

    summary = []
    for path in opt.input:
        for th in opt.threshold:
            count = process_one(net, path, th, opt.outdir, opt.class_filter)
            summary.append((os.path.basename(path), th, count))

    print("\n================ 实验汇总（直接用于填表） ================")
    print("%-28s %-10s %s" % ("Image", "Threshold", "Number of detected objects"))
    for name, th, count in summary:
        print("%-28s %-10.2f %d" % (name, th, count))
    print("=========================================================\n")
    print("提示：把 results/ 下每个 .txt 的内容粘进报告。")
    print("'Missed or incorrect detections' 一栏需要对照原图人工核对（见操作指南）。")


if __name__ == "__main__":
    main()
