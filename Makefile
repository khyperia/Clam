CC=c99
OBJDIR=obj
BINDIR=bin

WARNINGFLAGS=-Wall -Wextra

# My system doesn't have opencl.pc or glut.pc
PKGCONFIGCFLAGS=$(shell pkg-config --cflags gl libpng lua)
CFLAGS=-g -c -O2 $(WARNINGFLAGS) $(PKGCONFIGCFLAGS)

LDFLAGS_MASTER=-lglut $(shell pkg-config --libs gl libpng lua)
LDFLAGS_SLAVE=-lOpenCL -lglut $(shell pkg-config --libs gl)
LDFLAGS_ECHO=-lpthread

SOURCES=$(wildcard *.c)
# OBJECTS=$(patsubst %.c,$(OBJDIR)/%.o,$(SOURCES))
DEPENDS=$(patsubst %.c,$(OBJDIR)/%.d,$(SOURCES))

MASTERFILES=master luaHelper pngHelper socketHelper helper
SLAVEFILES=slave slaveSocket glclContext openclHelper socketHelper helper
ECHOFILES=echo socketHelper helper

toobjs=$(addsuffix .o,$(addprefix $(OBJDIR)/,$(1)))

all: $(BINDIR)/clam2_master $(BINDIR)/clam2_slave $(BINDIR)/clam2_echo

clean:
	rm -r $(OBJDIR) $(BINDIR)

bin/clam2_master: $(call toobjs,$(MASTERFILES))
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS_MASTER) $(call toobjs,$(MASTERFILES)) -o $@

bin/clam2_slave: $(call toobjs,$(SLAVEFILES))
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS_SLAVE) $(call toobjs,$(SLAVEFILES)) -o $@

bin/clam2_echo: $(call toobjs,$(ECHOFILES))
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS_ECHO) $(call toobjs,$(ECHOFILES)) -o $@

$(OBJDIR)/%.o: %.c Makefile
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD $< -o $@

-include $(DEPENDS)
