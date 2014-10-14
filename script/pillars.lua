require("math")
require("script/vector")

-- TODO: Make this dynamic
AssumedScreenWidth = 1024

frame = {
    pos = {5, 500, 5},
    look = {1, -1, 1},
    up = {0, -1, 0},
    fov = 1.0,
    focalDistance = 100.0,
    frame = 0.0
}

function special()
end

function recompile()
    compile("script/pillars.conf.cl", "script/pillars.cl")
end

recompile()

pilwidth, pilheight = uplbuffer(1, arg[1])

function update(time)
    frame = update3dCamera(frame, time)
    if iskeydown("b") then
        unsetkey("b")
        recompile()
        frame.frame = 0
        print("Recompiled")
    end
    if iskeydown("p") then
        unsetkey("p")
        print("Downloading screenbuffer at frame " .. tostring(frame.frame))
        dlbuffer(0, AssumedScreenWidth)
    end
    
    kernel("Main", -1, -1, "0",
    special,
    frame.pos[1], frame.pos[2], frame.pos[3],
    frame.look[1], frame.look[2], frame.look[3],
    frame.up[1], frame.up[2], frame.up[3],
    frame.fov / AssumedScreenWidth, frame.focalDistance, frame.frame,
    "1", {pilwidth}, {pilheight})

    frame.frame = frame.frame + 1
end
