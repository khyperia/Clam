Clam2
=====

A dynamic, scriptable OpenCL/Lua-based cluster computing rendering engine.

--

Clam2 provides a framework for an easily-programmed scripting environment that renders images using OpenCL. A master computer, running a Lua script, calls builtin functions to do operations like create, transfer, and delete buffers, and call OpenCL kernels that operate on those buffers. These commands are sent over the network to multiple slave computers, who perform those commands on their respective GPUs. A specially named buffer represents the screen, and writing to it is essentially to writing to the screen (in reality, OpenGL interop is required, and a fullscreen quad is drawn using the OpenCL buffer as a texture). In this way, complex structures can be rendered and displayed in realtime using the extreme power of GPUs, in an easy to program way.

--

# How to run:

Note: Only Linux supported at this time.

The master glut window must be focused for key input to work (NOT the master terminal window, nor the slave render window(s))

Get the code with

    git clone https://github.com/khyperia/Clam2.git
    cd Clam2

Note: If a plugin fails to build, and its functionality is not needed, it is possible to blacklist it by placing the name of the source file (i.e. with .c extension) in the file script/pluginBlacklist.txt. This file should already exist (via touch) if the makefile has been ran once.

##### Running it locally:

    make
    # Modifications to the Makefile may be necessary, due to system differences - primarily in PKGCONFIGCFLAGS and LDFLAGS_*
    bin/clam2_slave &
    bin/clam2_master script/[scriptname].lua

You may want to run the slave and master executables in separate terminal windows to not confuse output of each program.

See the other ways of running the program for other parameters (the defaults are okay for localhost)

##### Running it remotely:

See below for a list of environment variables and parameters. Sensible defaults are shown here, except for "??" values which need to be filled in.

On the slave computer(s):

    make bin/clam2_slave
    export CLAM2_PORT=23456
    export CLAM2_WINPOS=1024x1024+0+0
    export CLAM2_RENDERPOS=-512,-512
    export CLAM2_FULLSCREEN=false
    bin/clam2_slave

On the master computer:

    make bin/clam2_master
    export CLAM2_SLAVES=?? # Tilde-seperated list of IP:PORT pairs, e.g. 127.0.0.1:23456~foo.local:23456
    bin/clam2_master script/??.lua

On any "echo"/"relay" computers (which forward connections through themselves, if direct connection is not available):

    make bin/clam2_echo
    bin/clam2_echo $HOST_PORT $DEST_IP_1 $DEST_PORT_1 ... $DEST_IP_n $DEST_PORT_n
    # This will open a server on $HOST_PORT that takes $n repeated connections to itself, routing each one to the specified ip port pairs
    # Note there is NOT a colon between IP and PORT, but rather simply a space.

##### Running it on Michigan Tech's Interactive Visualization Studio:

On ccsr.ee:

    ./ivs.bash script/[scriptname].lua

You may have to build lua and modify some lines in the Makefile (specifically inside the if statement if HOSTNAME equals IVS_MASTERNAME) to point at the directory you built it in. Ensure you build lua with PIC enabled, as in:

    make linux MYCFLAGS=-fPIC

Note also it may be necessary to blacklist some plugins, as described above (in script/pluginBlacklist.txt)

--

# What script/[scriptname].lua means:

Clam2 is a dynamic scripting framework, and is useless on it's own. It needs a lua script to function (and those lua scripts in turn need .cl files to function)

### Included lua scripts (at the moment):

##### colors.lua:

A very basic kernel that is essentially just boilerplate to fork for other scripts. Good to start with if someone would like to make their own script. Also good to test the framework is working.

##### mandelbrot.lua:

Renders either the juliabrot/mandelbrot set.

Pressing "h" will dump what keys do what to stdout of the master program.

##### mandelbox.lua:

Renders the 3d mandelbox. Incredibly taxing on GPUs, a small screen size (CLAM2_WINPOS) and beefy GPU is required.

Pressing "h" will dump what keys do what to stdout of the master program.

Editing script/mandelbox.conf.cl changes various parameters of the kernel (recompile of script after editing needed, i.e. pressing 'b').

##### tris.lua:

Renders an object model (read in by assimp, usually .obj) by raytracing. Even though it is represented in a KD-tree, the maximum numbers of triangles possible to render is fairly small, struggling at counts above 5000 or so.

##### pillars.lua:

Unsupported. Similar to mandelbox.lua, except requires a png image parameter to the script (as in 'bin/clam2_master pillars.lua image.png'). It was forked from an old version of mandelbox.lua and, instead of the mandelbox, renders the png as 3d square pillars with height equal to r+g+b.

##### voxels.lua:

Unsupported. Similar to mandelbox.lua, except it uses a voxel octree rather than a fractal formula.

--

# Environment variable parameters to executables

(subject to change, run as specified under 'local' above to get a reliable printing of parameters through "Warning: foo not set" messages)

CLAM2_PORT: Self-explanatory, port to host server on

CLAM2_WINPOS: Where to create the GLUT window (handily in the format of xrandr)

CLAM2_RENDERPOS: Where the top-left corner of the window is with respect to the absolute screen center (in the case of multiple windows spread over multiple computers, this is important)

CLAM2_FULLLSCREEN: Whether or not to tell GLUT to fullscreen the window

CLAM2_SLAVES: Tilde-separated list of IPs for the host to connect to (in the case of echo servers, this will probably be the same IP repeated multiple times)
