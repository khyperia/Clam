exec clang src/mandelbox.cl -Weverything -target nvptx -cl-std=CL1.2 -include clc/clc.h -S -o output.ptx
