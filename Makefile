CC=clang++
OBJDIR=obj
DISABLEFLAGS=-Wno-c++98-compat -Wno-missing-variable-declarations -Wno-global-constructors -Wno-exit-time-destructors -Wno-missing-prototypes
CFLAGS=-c -O2 --std=c++11 -Wall -Wextra -Weverything $(DISABLEFLAGS) -I$(OBJDIR)
LDFLAGS=-lOpenCL -lGL -lglut -lpng -llua
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
