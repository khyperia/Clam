CC=g++
OBJDIR=obj
CFLAGS=-c --std=c++11 -Wall -Wextra -I$(OBJDIR) $(EXTRACFLAGS)
ifdef WIN32
	LDFLAGS=-lOpenCL -lopengl32 -lglut $(EXTRALDFLAGS)
else
	LDFLAGS=-lOpenCL -lGL -lglut $(EXTRALDFLAGS)
endif
SOURCES=main.cpp kernel.cpp context.cpp interop.cpp helper.cpp
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
