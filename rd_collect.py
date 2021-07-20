#!/usr/bin/python3
# Copyright 2013, Mozilla Corporation
# Copyright 2017-2020 Robert-André Mauchin
# Loosely based on a script written by Josh Aas
# https://github.com/bdaehlie/web_image_formats
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#     software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

import os
import errno
import subprocess
import sys
import glob
import re
import shlex
import string
import json
from multiprocessing import Pool
from timeit import Timer
import numpy as np

# Tests
vmaf = "vmaf --json --model version=vmaf_v0.6.1 --feature psnr --feature psnr_hvs --feature float_ssim --feature float_ms_ssim --feature ciede"

# Path to tmp dir to be used by the tests
tmpdir = "/tmp/"

#############################################################################


def split(cmd):
    lex = shlex.shlex(cmd)
    lex.quotes = '"'
    lex.whitespace_split = True
    lex.commenters = ""
    return list(lex)


def run_silent(cmd):
    FNULL = open(os.devnull, "w")
    rv = subprocess.call(split(cmd), stdout=FNULL, stderr=FNULL)
    if rv != 0:
        sys.stderr.write("Failure from subprocess:\n")
        sys.stderr.write("\t" + cmd + "\n")
        sys.stderr.write("Aborting!\n")
        sys.exit(rv)
    return rv


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


def create_dir(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def path_for_file_in_tmp(path):
    return tmpdir + str(os.getpid()) + os.path.basename(path)


def get_img_width(path):
    cmd = "identify -format %%w %s" % (path)
    proc = subprocess.Popen(
        split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
    )
    out, err = proc.communicate()
    if proc.returncode != 0:
        sys.stderr.write("Failed process: identify\n")
        sys.exit(proc.returncode)
    lines = out.split(os.linesep)
    return int(lines[0].strip())


def get_img_height(path):
    cmd = "identify -format %%h %s" % (path)
    proc = subprocess.Popen(
        split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
    )
    out, err = proc.communicate()
    if proc.returncode != 0:
        sys.stderr.write("Failed process: identify\n")
        sys.exit(proc.returncode)
    lines = out.split(os.linesep)
    return int(lines[0].strip())


def convert_img(inn, out):
    # PNG24: needed otherwise grayscale image lose their sRGB colorspace
    cmd = "convert %s PNG24:%s" % (inn, out)
    run_silent(cmd)


def detect_alpha(img):
    cmd = "identify -format '%%[channels]' %s" % (img)
    proc = subprocess.Popen(
        split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
    )
    out, err = proc.communicate()
    if proc.returncode != 0:
        sys.stderr.write("Failed process: %s\n" % (cmd))
        sys.exit(proc.returncode)

    channels = out.split(os.linesep)[0]
    if channels == "srgba":
        return True;
    return False


def remove_alpha(inn, out):
    # PNG24: needed otherwise grayscale image lose their sRGB colorspace
    cmd = "convert %s -alpha off PNG24:%s" % (inn, out)
    run_silent(cmd)


def convertff_img(inn, out):
    # 10le -strict -1
    cmd = (
        "ffmpeg -y -i %s -pix_fmt yuv444p -vf scale=in_range=full:out_range=full %s"
        % (inn, out)
    )
    run_silent(cmd)


def get_score(y4m1, y4m2, target_json):
    cmd = "%s -r %s -d %s -o %s" % (vmaf, y4m1, y4m2, target_json)
    proc = subprocess.Popen(
        split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
    )
    out, err = proc.communicate()
    if proc.returncode != 0:
        sys.stderr.write("Failed process: %s\n" % (cmd))
        sys.exit(proc.returncode)

    with open(target_json) as f:
        data = f.read()
        scores = json.loads(data)
        psnrhvs_score = scores["pooled_metrics"]["psnr_hvs"]["mean"]
        ssim_score = scores["pooled_metrics"]["float_ssim"]["mean"]
        msssim_score = scores["pooled_metrics"]["float_ms_ssim"]["mean"]
        ciede2000_score = scores["pooled_metrics"]["ciede2000"]["mean"]
        vmaf_score = scores["pooled_metrics"]["vmaf"]["mean"]

    return ssim_score, msssim_score, ciede2000_score, psnrhvs_score, vmaf_score


def get_butteraugli(png1, png2):
    cmd = "butteraugli_main %s %s" % (png1, png2)
    proc = subprocess.Popen(
        split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
    )
    out, err = proc.communicate()
    if proc.returncode != 0:
        sys.stderr.write("Failed process: %s\n" % (cmd))
        sys.exit(proc.returncode)

    butteraugli_score = float(out.split(os.linesep)[0])
    return butteraugli_score


def get_dssim(png1, png2):
    cmd = "dssim %s %s" % (png1, png2)
    proc = subprocess.Popen(
        split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
    )
    out, err = proc.communicate()
    if proc.returncode != 0:
        sys.stderr.write("Failed process: %s\n" % (cmd))
        sys.exit(proc.returncode)
    line = out.split(os.linesep)[0]
    dssim_score = float(re.search("^\d+\.?\d*", line).group(0))
    return dssim_score


def get_ssimulacra(png1, png2):
    cmd = "ssimulacra_main %s %s" % (png1, png2)
    proc = subprocess.Popen(
        split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
    )
    out, err = proc.communicate()
    if proc.returncode != 0:
        sys.stderr.write("Failed process: %s\n" % (cmd))
        sys.exit(proc.returncode)
    ssimulacra_score = float(out.split(os.linesep)[0])
    return ssimulacra_score


# Returns tuple containing:
#   (target_file_size, encode_time, decode_time)
def get_lossless_results(subset_name, origpng, format, format_recipe):

    target = (
        format.upper()
        + "_out/"
        + subset_name
        + "/"
        + os.path.splitext(os.path.basename(origpng))[0]
        + "/"
        + os.path.splitext(os.path.basename(origpng))[0]
        + "-lossless"
    )
    create_dir(target)
    target_dec = path_for_file_in_tmp(target)

    target += "." + format_recipe["encode_extension"]
    cmd = string.Template(format_recipe["lossless_cmd"]).substitute(locals())
    wrapped = wrapper(run_silent, cmd)
    encode_time = Timer(wrapped).timeit(5)

    target_dec += "." + format_recipe["decode_extension"]
    cmd = string.Template(format_recipe["decode_cmd"]).substitute(locals())
    wrapped = wrapper(run_silent, cmd)
    decode_time = Timer(wrapped).timeit(5)

    target_file_size = os.path.getsize(target)

    try:
        os.remove(target_dec)
    except FileNotFoundError:
        pass

    return (target_file_size, encode_time, decode_time)


# Returns tuple containing:
#   (target_file_size, encode_time, decode_time, yssim_score, rgbssim_score,
#   psnrhvsm_score, msssim_score)
def get_lossy_results(
    subset_name, origpng, width, height, format, format_recipe, quality
):

    origy4m = path_for_file_in_tmp(origpng) + ".y4m"
    convertff_img(origpng, origy4m)

    target = (
        format.upper()
        + "_out/"
        + subset_name
        + "/"
        + os.path.splitext(os.path.basename(origpng))[0]
        + "/"
        + os.path.splitext(os.path.basename(origpng))[0]
        + "-q"
        + str(quality)
    )
    create_dir(target)
    target_dec = path_for_file_in_tmp(target)

    target += "." + format_recipe["encode_extension"]
    cmd = string.Template(format_recipe["encode_cmd"]).substitute(locals())
    wrapped = wrapper(run_silent, cmd)
    encode_time = Timer(wrapped).timeit(1)

    target_json = target_dec + ".json"

    target_dec += "." + format_recipe["decode_extension"]
    cmd = string.Template(format_recipe["decode_cmd"]).substitute(locals())
    wrapped = wrapper(run_silent, cmd)
    decode_time = Timer(wrapped).timeit(1)

    if format_recipe["decode_extension"] != "png":
        target_png = path_for_file_in_tmp(target_dec) + ".png"
        convert_img(target_dec, target_png)
    else:
        target_png = target_dec

    # libavif bug?
    if not detect_alpha(origpng):
        remove_alpha(target_png, target_png)

    target_y4m = path_for_file_in_tmp(target_dec) + ".y4m"
    convertff_img(target_dec, target_y4m)

    ssim_score, msssim_score, ciede2000_score, psnrhvs_score, vmaf_score = get_score(
        origy4m, target_y4m, target_json
    )

    butteraugli_score = get_butteraugli(origpng, target_png)
    dssim_score = get_dssim(origpng, target_png)
    ssimulacra_score = get_ssimulacra(origpng, target_png)

    target_file_size = os.path.getsize(target)

    try:
        os.remove(origy4m)
        os.remove(target_dec)
        os.remove(target_json)
        os.remove(target_y4m)
    except FileNotFoundError:
        pass

    return (
        target_file_size,
        encode_time,
        decode_time,
        ssim_score,
        msssim_score,
        ciede2000_score,
        psnrhvs_score,
        vmaf_score,
        butteraugli_score,
        dssim_score,
        ssimulacra_score,
    )


def process_image(args):
    [format, format_recipe, subset_name, origpng] = args

    result_file = (
        "results/"
        + subset_name
        + "/"
        + format
        + "/lossy/"
        + os.path.splitext(os.path.basename(origpng))[0]
        + "."
        + format
        + ".out"
    )
    if os.path.isfile(result_file) and not os.stat(result_file).st_size == 0:
        return

    try:
        isfloat = (
            isinstance(format_recipe["quality_start"], float)
            or isinstance(format_recipe["quality_end"], float)
            or isinstance(format_recipe["quality_step"], float)
        )

        if isfloat:
            start = float(format_recipe["quality_start"])
            end = float(format_recipe["quality_end"])
            step = float(format_recipe["quality_step"])
        else:
            start = int(format_recipe["quality_start"])
            end = int(format_recipe["quality_end"])
            step = int(format_recipe["quality_step"])
    except ValueError:
        print("There was an error parsing the format recipe.")
        return

    if (
        not "encode_extension" in format_recipe
        or not "decode_extension" in format_recipe
        or not "encode_cmd" in format_recipe
        or not "lossless_cmd" in format_recipe
        or not "decode_cmd" in format_recipe
    ):
        print("There was an error parsing the format recipe.")
        return

    orig_file_size = os.path.getsize(origpng)
    width = get_img_width(origpng)
    height = get_img_height(origpng)
    pixels = width * height

    # Lossless
    print("Processing image {}, quality lossless".format(os.path.basename(origpng)))

    path = (
        "results/"
        + subset_name
        + "/"
        + format
        + "/lossless/"
        + os.path.splitext(os.path.basename(origpng))[0]
        + "."
        + format
        + ".out"
    )
    create_dir(path)
    file = open(path, "w")

    file.write(
        "file_name:orig_file_size:compressed_file_size:pixels:bpp:compression_ratio:encode_time:decode_time\n"
    )

    results = get_lossless_results(subset_name, origpng, format, format_recipe)
    bpp = results[0] * 8 / pixels
    compression_ratio = orig_file_size / results[0]
    file.write(
        "%s:%d:%d:%d:%f:%f:%f:%f\n"
        % (
            os.path.splitext(os.path.basename(origpng))[0],
            orig_file_size,
            results[0],
            pixels,
            bpp,
            compression_ratio,
            results[1],
            results[2],
        )
    )

    file.close()

    # Lossy
    path = (
        "results/"
        + subset_name
        + "/"
        + format
        + "/lossy/"
        + os.path.splitext(os.path.basename(origpng))[0]
        + "."
        + format
        + ".out"
    )
    create_dir(path)
    file = open(path, "w")

    file.write(
        "file_name:quality:orig_file_size:compressed_file_size:pixels:bpp:compression_ratio:encode_time:decode_time:ssim_score:msssim_score:ciede2000_score:psnrhvs_score:vmaf_score:butteraugli_score:dssim_score:ssimulacra_score\n"
    )
    if isfloat:
        quality_list = list(np.arange(start, end, step))
    else:
        quality_list = list(range(start, end, step))

    i = 0
    while i < len(quality_list):
        quality = quality_list[i]
        print(
            "Processing image {}, quality {}".format(os.path.basename(origpng), quality)
        )
        results = get_lossy_results(
            subset_name, origpng, width, height, format, format_recipe, quality
        )
        bpp = results[0] * 8 / pixels
        compression_ratio = orig_file_size / results[0]
        file.write(
            "%s:%f:%d:%d:%d:%f:%f:%f:%f:%f:%f:%f:%f:%f:%f:%f:%f\n"
            % (
                os.path.splitext(os.path.basename(origpng))[0],
                quality,
                orig_file_size,
                results[0],
                pixels,
                bpp,
                compression_ratio,
                results[1],
                results[2],
                results[3],
                results[4],
                results[5],
                results[6],
                results[7],
                results[8],
                results[9],
                results[10],
            )
        )
        i += 1

    file.close()


def main(argv):
    if sys.version_info[0] < 3 and sys.version_info[1] < 5:
        raise Exception("Python 3.5 or a more recent version is required.")

    data = {}
    try:
        with open("recipes.json") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        raise Exception("Could not find recipes.json")

    supported_formats = list(data["recipes"].keys())

    if len(argv) != 4:
        print(
            "rd_collect.py: Generate compressed images from Y4Ms and calculate quality and speed metrics for a given format"
        )
        print("Arg 1: format to test {}".format(supported_formats))
        print("Arg 2: name of the subset to test (e.g. 'subset1')")
        print("Arg 3: path to the subset to test (e.g. 'subset1/')")
        return

    format = argv[1]
    subset_name = argv[2]
    if format not in supported_formats:
        print(
            "Image format not supported. Supported formats are: {}.".format(
                supported_formats
            )
        )
        return

    pool = Pool(processes=16)
    pool.map(
        process_image,
        [
            (format, data["recipes"][format], subset_name, origpng)
            for origpng in glob.glob(argv[3] + "/*.png")
        ],
    )


if __name__ == "__main__":
    main(sys.argv)
