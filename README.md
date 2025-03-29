# Alpha
## Introduction
Alpha is a simple & efficient AI framework written in C. It attempts to provide an alternative to the existing AI frameworks, while attempting to bring the C programming language to the forefront of the AI industry.
## Features
Alpha is efficient, but not as efficient as other mature AI frameworks such as PyTorch or TensorFlow yet. However, it is capable of harnessing the power of both CPUs and GPUs, using OpenMP. If OpenMP is available, Alpha will use it to parallelize the code.
### Offloading
By default, OpenMP will use all available cores, but it can easily be configured to use the GPU. The following example will compile Alpha, or any project using Alpha as a submodule with NVIDIA offloading support using OpenMP and GCC:
```sh
cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_C_FLAGS="-fopenmp -foffload=nvptx-none -foffload-options=-misa=sm_80" ..
```
Replace `sm_80` with the desired GPU architecture. In my case, I have an NVIDIA GeForce RTX 4060, so `sm_80` is fine for me, although `sm_86` would be better (my system seems to support `sm_80` and not `sm_86`).
Note that you will need the CUDA toolkit installed, as well as the NVIDIA driver for this to work. OpenMP should also have offloading support for other GPUs and platforms, but I have not tested it yet. If you are able to successfully compile Alpha with offloading support for other GPUs, submitting a pull request with the commands & procedures you used would be greatly appreciated.