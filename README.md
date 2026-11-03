# prot_features_cuda

Computing amino-acid composition features of protein sequences on the GPU with CUDA. This was the final
project for CMSC 603 (High Performance Distributed Systems, VCU); the proposal is `Description.pdf` /
`Description.docx`, the final write-up is `Document/Document.pdf` inside `submission/Document.zip`, and the
slides are `presentation.pptx`.

## What the program does

`Final_work.cu` reads a tab-separated file named `Read.txt` from the current directory, whose header is
`PDB ID<TAB>Sequence` and whose rows are one protein per line (identifier, sequence of capital letters).
For every sequence it counts how many times each letter occurs (26 counters per sequence, indexed by
`letter - 'A'`), first on the CPU in a plain loop and then on the GPU with one thread per sequence
(`cuda_features` kernel, 256 threads per block), and prints the elapsed time of each. The feature values
themselves are not printed; the code that printed them is commented out.

Limits baked into the source: at most `MAX_ELEMENTS = 10000` sequences are read (the rest of the file is
ignored) and each sequence is copied into a fixed `MAX_LENGTH = 10000` character slot, so longer sequences
overflow it. Neither the host nor the device feature arrays are zeroed before the counters are
incremented, so the counts rely on the freshly allocated memory being zero.

## Files

- `Final_work.cu` – the final program. `submission/main.cu` is an identical copy; `prot_features.cu` is an
  earlier revision without the CPU timing.
- `submission/` – what was handed in: `main.cu`, `Read.txt` (104,369 PDB sequences, ~32 MB), `ReadMe.txt`,
  `Makefile.mak` (left over from a KNN assignment template; it refers to `main.cpp` and a `libarff/`
  directory that are not part of this project, so it does not work) and `Document.zip` (the LaTeX write-up).
  `submission.zip` is the same directory zipped.
- `Makefile` – builds `prot_features` from `Final_work.cu` with `nvcc`.

## Building and running

```sh
make                       # nvcc -O2 -o prot_features Final_work.cu
cd submission && ../prot_features   # Read.txt must be in the current directory
```

The program prints `without GPU <ms>`, the kernel launch configuration and `with GPU <ms>`.
