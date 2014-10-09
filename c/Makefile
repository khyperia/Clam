CC=c99
OBJDIR=obj
BINDIR=bin

WARNINGFLAGS=-Wall -Wextra

# My system doesn't have opencl.pc or glut.pc
PKGCONFIGPACKS=gl libpng lua

PKGCONFIGCFLAGS=$(shell pkg-config --cflags $(PKGCONFIGPACKS))
CFLAGS=-g -c -O2 $(WARNINGFLAGS) $(PKGCONFIGCFLAGS)

PKGCONFIGLIBS=$(shell pkg-config --libs $(PKGCONFIGPACKS))
LDFLAGS=-lOpenCL -lglut $(PKGCONFIGLIBS)

SOURCES=$(wildcard *.c)
OBJECTS=$(patsubst %.c,$(OBJDIR)/%.o,$(SOURCES))
DEPENDS=$(patsubst %.c,$(OBJDIR)/%.d,$(SOURCES))

MASTERFILES=master socketHelper helper
SLAVEFILES=slave slaveSocket glclContext socketHelper helper openclHelper

all: $(BINDIR)/clam2_master $(BINDIR)/clam2_slave

clean:
	rm -r $(OBJDIR) $(BINDIR)

bin/clam2_master: $(OBJECTS)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) $(addsuffix .o,$(addprefix $(OBJDIR)/,$(MASTERFILES))) -o $@

bin/clam2_slave: $(OBJECTS)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) $(addsuffix .o,$(addprefix $(OBJDIR)/,$(SLAVEFILES))) -o $@

$(OBJDIR)/%.o: %.c Makefile
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD $< -o $@

-include $(DEPENDS)
