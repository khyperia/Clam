require("vector")
require("maketris")
require("rawbuffer")
require("input")

AssumedScreenWidth = 1024

frame = {
    pos = {0, 0, 500},
    look = {0, 0, -1},
    up = {0, -1, 0},
    fov = 1.0,
    focalDistance = 100.0,
    frame = 0.0
}

function special() end

print("Generating tris data file")
local trisfile = maketris("../duck.dae")
print("Done making data file, uploading")
uplrawbuffer("1", trisfile)

function recompile()
    compile("script/tris.cl")
end

recompile()

function update(time)
    frame = update3dCamera(frame, time)
    if iskeydown("b") then
        unsetkey("b")
        recompile()
    end
    if iskeydown("g") then
        unsetkey("g")
        reload()
    end
    kernel("Main", -1, -1, "0", "1",
    special,
    frame.pos[1], frame.pos[2], frame.pos[3],
    frame.look[1], frame.look[2], frame.look[3],
    frame.up[1], frame.up[2], frame.up[3],
    frame.fov / AssumedScreenWidth, {frame.frame})

    frame.frame = frame.frame + 1

    hardsync()
end