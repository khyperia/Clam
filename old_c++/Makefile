CC=clang++
OBJDIR=obj

WARNINGFLAGS=-Wall -Wextra -Weverything -Wno-c++98-compat -Wno-missing-variable-declarations -Wno-global-constructors -Wno-exit-time-destructors -Wno-missing-prototypes

# My system doesn't have opencl.pc or glut.pc
PKGCONFIGPACKS=gl libpng lua

PKGCONFIGCFLAGS=$(shell pkg-config --cflags $(PKGCONFIGPACKS))
CFLAGS=-c -O2 --std=c++11 $(WARNINGFLAGS) $(PKGCONFIGCFLAGS)

PKGCONFIGLIBS=$(shell pkg-config --libs $(PKGCONFIGPACKS))
LDFLAGS=-lOpenCL -lglut $(PKGCONFIGLIBS)

SOURCES=$(wildcard *.cpp)
OBJECTS=$(patsubst %.cpp,$(OBJDIR)/%.o,$(SOURCES))
DEPENDS=$(patsubst %.cpp,$(OBJDIR)/%.d,$(SOURCES))

EXECUTABLE=bin/clam2

all: $(SOURCES) $(EXECUTABLE)

clean:
	rm $(EXECUTABLE) $(OBJECTS) $(DEPENDS)

$(EXECUTABLE): $(OBJECTS)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

$(OBJDIR)/%.o: %.cpp Makefile
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD $< -o $@

-include $(DEPENDS)
