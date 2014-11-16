CC=c99
CCP=c++ -std=c++98
OBJDIR=obj
BINDIR=bin
SCRIPTDIR=script

HOSTNAME=$(shell hostname)
IVS_SLAVENAME=ivs.research.mtu.edu
IVS_MASTERNAME=ccsr.ee.mtu.edu

WARNINGFLAGS=-Wall -Wextra
ifeq ($(HOSTNAME), $(IVS_SLAVENAME))
# Building on ivs.research.mtu.edu
PKGCONFIGCFLAGS=-I/export/apps/cuda-5.0/include/
LDFLAGS_SLAVE=-lOpenCL -lGL -lSDL2 -lSDL2_net
else
ifeq ($(HOSTNAME), $(IVS_MASTERNAME))
# Building on ccsr.ee.mtu.edu
LDFLAGS_MASTER=-rdynamic -L/home/echauck/lua/src -llua -lrt -lSDL2 -lSDL2_net $(shell pkg-config --libs gl libpng)
PKGCONFIGCFLAGS=$(shell pkg-config --cflags gl libpng sdl2) -I/home/kuhl/public-vrlab/vrpn -I/home/echauck/lua/src
LDFLAGS_SO=-L/home/echauck/lua/src -L/home/kuhl/public-vrlab/vrpn/build -llua
else
# Standard build
LDFLAGS_MASTER=-rdynamic $(shell pkg-config --libs gl libpng lua SDL2_net)
PKGCONFIGCFLAGS=$(shell pkg-config --cflags gl libpng lua sdl2 SDL2_net)
LDFLAGS_SLAVE=-lOpenCL -lGL $(shell pkg-config --libs sdl2 SDL2_net)
LDFLAGS_SO=$(shell pkg-config --libs lua)
endif
endif

CFLAGS=-O2 $(WARNINGFLAGS) $(PKGCONFIGCFLAGS)
ifeq ("","$(wildcard $(SCRIPTDIR)/pluginBlacklist.txt)")
$(warning "$(SCRIPTDIR)/pluginBlacklist.txt doesn't exist, touching it. Fill it in with .c/.cpp plugin files that you don't want compiled")
$(shell touch $(SCRIPTDIR)/pluginBlacklist.txt)
endif

PLUGIN_SOURCES=$(wildcard $(SCRIPTDIR)/*.c) $(wildcard $(SCRIPTDIR)/*.cpp)
PLUGIN_BLACKLIST=$(addprefix $(SCRIPTDIR)/,$(shell cat $(SCRIPTDIR)/pluginBlacklist.txt))
PLUGIN_SOURCES:=$(filter-out $(PLUGIN_BLACKLIST),$(PLUGIN_SOURCES))
PLUGINS=$(PLUGIN_SOURCES)
PLUGINS:=$(patsubst %.cpp,%.so,$(PLUGINS))
PLUGINS:=$(patsubst %.c,%.so,$(PLUGINS))
SOURCES=$(wildcard *.c) $(PLUGIN_SOURCES)
DEPENDS=$(SOURCES)
DEPENDS:=$(patsubst %.cpp,$(OBJDIR)/%.d,$(DEPENDS))
DEPENDS:=$(patsubst %.c,$(OBJDIR)/%.d,$(DEPENDS))

MASTERFILES=master masterSocket luaHelper socketHelper helper
SLAVEFILES=slave slaveSocket glclContext openclHelper socketHelper helper

toobjs=$(addsuffix .o,$(addprefix $(OBJDIR)/,$(1)))

.PHONY: all
all: master slave

.PHONY: clean
clean:
	rm -r $(OBJDIR) $(BINDIR) $(PLUGINS)

.PHONY: master
master: bin/clam2_master $(PLUGINS)
.PHONY: slave
slave: bin/clam2_slave

bin/clam2_master: $(call toobjs,$(MASTERFILES))
	@mkdir -p $(@D)
	$(CC) $(call toobjs,$(MASTERFILES)) -o $@ $(LDFLAGS_MASTER)

bin/clam2_slave: $(call toobjs,$(SLAVEFILES))
	@mkdir -p $(@D)
	$(CC) $(call toobjs,$(SLAVEFILES)) -o $@ $(LDFLAGS_SLAVE)

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
