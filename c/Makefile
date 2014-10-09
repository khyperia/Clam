CC=c99
OBJDIR=obj
BINDIR=bin

WARNINGFLAGS=-Wall -Wextra

# My system doesn't have opencl.pc or glut.pc
PKGCONFIGCFLAGS=$(shell pkg-config --cflags gl libpng lua)
CFLAGS=-g -c -O2 $(WARNINGFLAGS) $(PKGCONFIGCFLAGS)

LDFLAGS_MASTER=-lglut $(shell pkg-config --libs gl libpng lua)
LDFLAGS_SLAVE=-lOpenCL -lglut $(shell pkg-config --libs gl)

SOURCES=$(wildcard *.c)
OBJECTS=$(patsubst %.c,$(OBJDIR)/%.o,$(SOURCES))
DEPENDS=$(patsubst %.c,$(OBJDIR)/%.d,$(SOURCES))

MASTERFILES=master luaHelper socketHelper helper
SLAVEFILES=slave slaveSocket glclContext openclHelper socketHelper helper

all: $(BINDIR)/clam2_master $(BINDIR)/clam2_slave

clean:
	rm -r $(OBJDIR) $(BINDIR)

bin/clam2_master: $(OBJECTS)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS_MASTER) $(addsuffix .o,$(addprefix $(OBJDIR)/,$(MASTERFILES))) -o $@

bin/clam2_slave: $(OBJECTS)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS_SLAVE) $(addsuffix .o,$(addprefix $(OBJDIR)/,$(SLAVEFILES))) -o $@

$(OBJDIR)/%.o: %.c Makefile
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD $< -o $@

-include $(DEPENDS)
