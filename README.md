# HPC-project-2024

This is the final project for the 2024 course of "High Performance Computing" of the Master degree "Quantum Engineering" at Politecnico di Torino.

Written in CUDA, the end result should be a CLI tool. Given a complex function (among a relative wide range of possibilities) and, optionally, the values of some options (width, height,fps, center, function parameters, etc.), such tool produces a video of the stream plot of the chosen function.

The general program flow is the following
1. Initial particle positions are generated using [Lloyd's algorithm](https://en.m.wikipedia.org/wiki/Lloyd's_algorithm)
2. The evolution of the particles is written on multiple canvases in order to parallelize writing on different OpenMP threads or CUDA thread blocks
3. For each frame, the same pixel of each canvas are combined into a single pixel which is drawn into the frame buffer
4. Frames are combined into a Webp container to build a Webp video


## Supported resolutions

<table style="text-align: center">
    <thead>
        <tr><th>Name</th><th>Pixels</th><th>Ratio</th></tr>
    </thead>
    <tbody>
        <tr><td>VGA</td><td>640x480</td><td rowspan="4">4:3</td></tr>
        <tr><td>SVGA</td><td>800x600</td></tr>
        <tr><td>XGA</td><td>1024x768</td></tr>
        <tr><td>QXGA</td><td>2048x1536</td></tr>
        <tr><td>qHD</td><td>960x540</td><td rowspan="5">16:9</td></tr>
        <tr><td>FHD</td><td>1920x1080</td></tr>
        <tr><td>WQHD</td><td>2560x1440</td></tr>
        <tr><td>WQXGA+</td><td>3200x1800</td></tr>
        <tr><td>UHD</td><td>3840x2160</td></tr>
        <tr><td>FHD+</td><td>1920x1280</td><td>3:2</td></tr>
        <tr><td>UW-FHD</td><td>2560x1080</td><td rowspan="2">21:9</td></tr>
        <tr><td>UW-QHD</td><td>3440x1440</td></tr>
        <tr><td>WXQA</td><td>1280x800</td><td rowspan="6">8:5</td></tr>
        <tr><td>WXGA+</td><td>1440x900</td></tr>
        <tr><td>WSXGA+</td><td>1680x1050</td></tr>
        <tr><td>WUXGA</td><td>1920x1200</td></tr>
        <tr><td>WQXGA</td><td>2560x1600</td></tr>
        <tr><td>WQUXGA</td><td>3840x2400</td></tr>
    </tbody>
</table>

## Library used

This tool makes use of two libraries:
- `getopt` &mdash; Preinstalled on Linux. When compiling for Windows, a port of `getopt` (taken from [Chunde's repository](https://github.com/Chunde/getopt-for-windows)) is injected.
- `ffmpeg` &mdash; Once installed, the environment variable `FFMPEG_PATH` must be specified with the path to the installation directory of `ffmpeg`. When building for Windows, be sure to add the `${FFMPEG_PATH}/bin` directory in the PATH, so that dll can be used by the executable.

## Debugging tools

```bash
ncu -o <out_file> <command>
compute-sanitizer --tool memcheck --log-file <out_file> <command>
```