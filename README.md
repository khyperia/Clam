Clam2
=====

A dynamic, scriptable OpenCL/Lua-based cluster computing rendering engine.

--

# How to run:

Note: Only Linux supported at this time.

The master glut window must be focused for key input to work (NOT the master terminal window, nor the slave render window(s))

Get the code with

    git clone https://github.com/khyperia/Clam2.git
    cd Clam2

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

--

# What script/[scriptname].lua means:

Clam2 is a dynamic scripting framework, and is useless on it's own. It needs a lua script to function (and those lua scripts in turn need .cl files to function)

### Included lua scripts (at the moment):

##### mandelbrot.lua:

Renders either the juliabrot/mandelbrot set. Eventually a helptext will be included when pressing "h" while the script is running, for now it is here:

wasd: move around
rf: zoom in/out
j: toggle juliabrot/mandelbrot set
u: set the juliabrot C value to the coordinates of the center of the screen
p: take a render (edit the script code to set image size)

##### mandelbox.lua:

Renders the 3d mandelbox. Incredibly taxing on GPUs, a small screen size (CLAM2_WINPOS) and beefy GPU is required.

Pressing "h" will dump what keys do what to stdout of the master program.

Editing script/mandelbox.conf.cl changes various parameters of the kernel (recompile of script after editing needed, i.e. pressing 'b').

##### pillars.lua:

Unsupported. Similar to mandelbox.lua, except requires a png image parameter to the script (as in 'bin/clam2_master pillars.lua image.png'). It was forked from an old version of mandelbox.lua and, instead of the mandelbox, renders the png as 3d square pillars with height equal to r+g+b.

--

# Environment variable parameters to executables

(subject to change, run as specified under 'local' above to get a reliable printing of parameters through "Warning: foo not set" messages)

CLAM2_PORT: Self-explanatory, port to host server on
CLAM2_WINPOS: Where to create the GLUT window (handily in the format of xrandr)
CLAM2_RENDERPOS: Where the top-left corner of the window is with respect to the absolute screen center (in the case of multiple windows spread over multiple computers, this is important)
CLAM2_FULLLSCREEN: Whether or not to tell GLUT to fullscreen the window
CLAM2_UNSAFE_DELAY_FRAMES: You probably don't want to set this to anything other than 0. However, increasing to 1, 2 or 3 may increase framerate due to network lag. Setting it to anything other than zero **WILL BREAK** any lua scripts that download buffers!
CLAM2_SLAVES: Tilde-separated list of IPs for the host to connect to (in the case of echo servers, this will probably be the same IP repeated multiple times)
