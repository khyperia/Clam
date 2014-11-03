CC=c99
CCP=c++ -std=c++98
OBJDIR=obj
BINDIR=bin
SCRIPTDIR=script

HOSTNAME=$(shell hostname)
IVS_SLAVENAME=ivs.research.mtu.edu
IVS_MASTERNAME=ccsr.ee


WARNINGFLAGS=-Wall -Wextra
ifeq ($(HOSTNAME), $(IVS_SLAVENAME))
# Building on ivs.research.mtu.edu
PKGCONFIGCFLAGS="$(pkg-config --cflags gl libpng) -I/home/kuhl/public-vrlab/lua/src"
LDFLAGS_MASTER="-L/home/kuhl/public-vrlab/lua/src -llua -lglut -lrt $(pkg-config --libs gl libpng)"
LDFLAGS_SO="-L/home/kuhl/public-vrlab/lua/src -llua"
else
# Common between ccsr.ee and standard
LDFLAGS_MASTER=-rdynamic -lglut $(shell pkg-config --libs gl libpng lua)
LDFLAGS_SO=$(shell pkg-config --libs lua)
ifeq ($(HOSTNAME), $(IVS_MASTERNAME))
# Building on ccsr.ee
PKGCONFIGCFLAGS=-I/export/apps/cuda-5.0/include/
else
# Standard build
PKGCONFIGCFLAGS=$(shell pkg-config --cflags gl libpng lua)
LDFLAGS_SLAVE=-lOpenCL -lglut -lGL
LDFLAGS_ECHO=-lpthread
endif
endif

CFLAGS=-O2 $(WARNINGFLAGS) $(PKGCONFIGCFLAGS)

PLUGIN_SOURCES=$(wildcard $(SCRIPTDIR)/*.c) $(wildcard $(SCRIPTDIR)/*.cpp)
PLUGINS:=$(PLUGIN_SOURCES)
PLUGINS:=$(patsubst %.cpp,%.so,$(PLUGINS))
PLUGINS:=$(patsubst %.c,%.so,$(PLUGINS))
SOURCES=$(wildcard *.c) $(PLUGIN_SOURCES)
DEPENDS=$(SOURCES)
DEPENDS:=$(patsubst %.cpp,$(OBJDIR)/%.d,$(DEPENDS))
DEPENDS:=$(patsubst %.c,$(OBJDIR)/%.d,$(DEPENDS))

MASTERFILES=master luaHelper socketHelper helper
SLAVEFILES=slave slaveSocket glclContext openclHelper socketHelper helper
ECHOFILES=echo socketHelper helper

toobjs=$(addsuffix .o,$(addprefix $(OBJDIR)/,$(1)))

.PHONY: all
all: master slave echo

.PHONY: clean
clean:
	rm -r $(OBJDIR) $(BINDIR) $(PLUGINS)

.PHONY: master
master: bin/clam2_master $(PLUGINS)
.PHONY: slave
slave: bin/clam2_slave
.PHONY: echo
echo: bin/clam2_echo

bin/clam2_master: $(call toobjs,$(MASTERFILES))
	@mkdir -p $(@D)
	$(CC) $(call toobjs,$(MASTERFILES)) -o $@ $(LDFLAGS_MASTER)

bin/clam2_slave: $(call toobjs,$(SLAVEFILES))
	@mkdir -p $(@D)
	$(CC) $(call toobjs,$(SLAVEFILES)) -o $@ $(LDFLAGS_SLAVE)

bin/clam2_echo: $(call toobjs,$(ECHOFILES))
	@mkdir -p $(@D)
	$(CC) $(call toobjs,$(ECHOFILES)) -o $@ $(LDFLAGS_ECHO)

$(OBJDIR)/%.o: %.c Makefile
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) -MMD $< -o $@

%.so: %.c Makefile $(wildcard %.mk)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -shared -o $@ -fPIC $< $(LDFLAGS_SO)

%.so: %.cpp Makefile $(wildcard %.mk)
	@mkdir -p $(@D)
	$(CCP) $(CFLAGS) -shared -o $@ -fPIC $< $(LDFLAGS_SO)

-include $(patsubst %.so,%.mk,$(PLUGINS))
-include $(DEPENDS)
