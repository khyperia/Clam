require("math")
require("script/vector")

pos = {0, 0, 5}
look = {0, 0, -1}
up = {0, 1, 0}
fov = 1.0
focalDistance = 1.0
frame = 0.0
frames = {}

function special()
end

compile("script/mandelbox.conf.cl", "script/mandelbox.cl")

function screenshot(filename, frame, width, height, numframes)
    mkbuffer(filename, width * height * 4 * 4)
    for i=0,numframes do
        kernel("Main", width, height, filename,
        {math.floor(-width / 2)}, {math.floor(-height / 2)}, {width}, {height},
        frame.pos[1], frame.pos[2], frame.pos[3],
        frame.look[1], frame.look[2], frame.look[3],
        frame.up[1], frame.up[2], frame.up[3],
        frame.fov / width, frame.focalDistance, i)
        --print((i / numframes * 100), "% done");
    end
    dlbuffer(filename, width)
    rmbuffer(filename)
end

function video(frames)
    function videoFrameAt(time)
        fractional = time % 1
        whole = math.floor(time)
        frame1 = frames[whole]
        frame2 = frames[whole + 1]
        if frame1 == nil or frame2 == nil then
            return nil
        end
        frame0 = frames[whole - 1]
        if frame0 == nil then frame0 = frame1 end
        frame3 = frames[whole + 2]
        if frame3 == nil then frame3 = frame2 end
        return {
            pos = catmullRom(frame0.pos, frame1.pos, frame2.pos, frame3.pos, fractional),
            look = catmullRom(frame0.look, frame1.look, frame2.look, frame3.look, fractional),
            up = catmullRom(frame0.up, frame1.up, frame2.up, frame3.up, fractional),
            fov = catmullRomF(frame0.fov, frame1.fov, frame2.fov, frame3.fov, fractional),
            focalDistance = catmullRomF(frame0.focalDistance, frame1.focalDistance,
                frame2.focalDistance, frame3.focalDistance, fractional)
        }
    end
    local frameIndex = 0
    while true do
        local frame = videoFrameAt(frameIndex / 10.0)
        if frame == nil then
            print("Frame nil, breaking")
            break
        end
        screenshot("tmp/video" .. tostring(frameIndex), frame, 800, 500, 10)
        print("Took frame", frameIndex)
        frameIndex = frameIndex + 1
    end
end

function getFrame()
    return { pos = pos, look = look, up = up, fov = fov, focalDistance = focalDistance }
end

function update(time)
    if iskeydown("w") then
        pos = addvec(pos, mulvec(look, time * focalDistance))
        frame = 0
    end
    if iskeydown("s") then
        pos = addvec(pos, mulvec(look, -time * focalDistance))
        frame = 0
    end
    if iskeydown("a") then
        pos = addvec(pos, mulvec(cross(up, look), time * focalDistance))
        frame = 0
    end
    if iskeydown("d") then
        pos = addvec(pos, mulvec(cross(look, up), time * focalDistance))
        frame = 0
    end
    if iskeydown(" ") then
        pos = addvec(pos, mulvec(up, -time * focalDistance))
        frame = 0
    end
    if iskeydown("z") then
        pos = addvec(pos, mulvec(up, time * focalDistance))
        frame = 0
    end
    if iskeydown("r") then
        focalDistance = focalDistance * (1 + time * fov)
        frame = 0
    end
    if iskeydown("f") then
        focalDistance = focalDistance / (1 + time * fov)
        frame = 0
    end
    if iskeydown("u") then
        up = rotate(up, look, time)
        frame = 0
    end
    if iskeydown("o") then
        up = rotate(up, look, -time)
        frame = 0
    end
    if iskeydown("j") then
        look = rotate(look, up, time * fov)
        frame = 0
    end
    if iskeydown("l") then
        look = rotate(look, up, -time * fov)
        frame = 0
    end
    if iskeydown("i") then
        look = rotate(look, cross(up, look), time * fov)
        frame = 0
    end
    if iskeydown("k") then
        look = rotate(look, cross(look, up), time * fov)
        frame = 0
    end
    if iskeydown("n") then
        fov = fov * (1 + time)
        frame = 0
    end
    if iskeydown("m") then
        fov = fov / (1 + time)
        frame = 0
    end
    if iskeydown("b") then
        unsetkey("b")
        compile("script/mandelbox.conf.cl", "script/mandelbox.cl")
        frame = 0
        print("Recompiled")
    end
    if iskeydown("p") then
        unsetkey("p")
        screenshot("fractal", getFrame(), 5000, 5000, 100)
    end
    if iskeydown("x") then
        unsetkey("x")
        frames = {}
        print("Cleared frames")
    end
    if iskeydown("c") then
        unsetkey("c")
        local index = 0
        while frames[index] ~= nil do
            index = index + 1
        end
        frames[index] = getFrame()
        print("Added frame", index)
    end
    if iskeydown("v") then
        print("Taking video")
        video(frames)
        print("Done taking video")
    end

    look = normalize(look)
    up = normalize(cross(cross(look, up), look))

    kernel("Main", -1, -1, "",
    special,
    pos[1], pos[2], pos[3],
    look[1], look[2], look[3],
    up[1], up[2], up[3],
    fov / 800, focalDistance, frame)

    frame = frame + 1
end
