CC=g++
CFLAGS=-c --std=c++11 -Wall -Wextra
LDFLAGS=-lOpenCL -lGL -lglut
SOURCES=main.cpp kernel.cpp context.cpp interop.cpp helper.cpp
OBJECTS=$(SOURCES:.cpp=.o)
DEPENDS=$(SOURCES:.cpp=.d)
EXECUTABLE=clam2

all: $(SOURCES) $(EXECUTABLE)

clean:
	rm $(EXECUTABLE) $(OBJECTS) $(DEPENDS)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@ -MMD

-include $(DEPENDS)
