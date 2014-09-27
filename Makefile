CC=clang++
OBJDIR=obj
CFLAGS=-c -O2 --std=c++11 -Wall -Wextra -I$(OBJDIR) $(EXTRACFLAGS)
LDFLAGS=-lOpenCL -lGL -lglut -lpng -llua $(EXTRALDFLAGS)
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

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< -o $@ -MMD

-include $(DEPENDS)
