output: mandelbox.obj crappy_reimpl.obj main.obj
	clang -O2 *.obj -lm -o output

mandelbox.obj: ../src/mandelbox.cl
	clang -O2 -Weverything -cl-std=CL1.2 -Xclang -finclude-default-header -c ../src/mandelbox.cl -o mandelbox.obj

crappy_reimpl.obj: crappy_reimpl.cpp
	clang -O2 -Weverything -Wno-missing-prototypes -c crappy_reimpl.cpp -o crappy_reimpl.obj

main.obj: main.cpp
	clang -O2 -Weverything -c main.cpp -o main.obj
