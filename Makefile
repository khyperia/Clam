CC=g++
CFLAGS=-c --std=c++11 -Wall
LDFLAGS=-lOpenCL -lGL -lGLEW -lglut
SOURCES=main.cpp kernel.cpp context.cpp interop.cpp helper.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=clam2

all: $(SOURCES) $(EXECUTABLE)

clean:
	rm $(EXECUTABLE) *.o

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
