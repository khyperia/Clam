require("math")
require("script/vector")
require("script/table_save")

frame = {
    pos = {0, 0, 5},
    look = {0, 0, -1},
    up = {0, 1, 0},
    fov = 1.0,
    focalDistance = 1.0,
    frame = 0.0
}
frames = {}

function special()
end

compile("script/mandelbox.conf.cl", "script/mandelbox.cl")

function screenshot(bufferIndex, frame, width, height, numframes)
    mkbuffer(bufferIndex, width * height * 4 * 4)
    for i=0,numframes do
        kernel("Main", width, height, bufferIndex,
        {math.floor(-width / 2)}, {math.floor(-height / 2)}, {width}, {height},
        frame.pos[1], frame.pos[2], frame.pos[3],
        frame.look[1], frame.look[2], frame.look[3],
        frame.up[1], frame.up[2], frame.up[3],
        frame.fov / width, frame.focalDistance, i)
        print((i / numframes * 100), "% done");
    end
    dlbuffer(bufferIndex, width) -- TODO
    rmbuffer(bufferIndex)
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
        screenshot(tostring(frameIndex), frame, 800, 500, 10)
        print("Took frame", frameIndex)
        frameIndex = frameIndex + 1
    end
end

function update(time)
    -- free keys: gh
    if iskeydown("w") then
        frame.pos = addvec(frame.pos, mulvec(frame.look, time * frame.focalDistance))
        frame.frame = 0
    end
    if iskeydown("s") then
        frame.pos = addvec(frame.pos, mulvec(frame.look, -time * frame.focalDistance))
        frame.frame = 0
    end
    if iskeydown("a") then
        frame.pos = addvec(frame.pos, mulvec(cross(frame.up, frame.look),
        time * frame.focalDistance))
        frame.frame = 0
    end
    if iskeydown("d") then
        frame.pos = addvec(frame.pos, mulvec(cross(frame.look, frame.up),
        time * frame.focalDistance))
        frame.frame = 0
    end
    if iskeydown(" ") then
        frame.pos = addvec(frame.pos, mulvec(frame.up, -time * frame.focalDistance))
        frame.frame = 0
    end
    if iskeydown("z") then
        frame.pos = addvec(frame.pos, mulvec(frame.up, time * frame.focalDistance))
        frame.frame = 0
    end
    if iskeydown("r") then
        frame.focalDistance = frame.focalDistance * (1 + time * frame.fov)
        frame.frame = 0
    end
    if iskeydown("f") then
        frame.focalDistance = frame.focalDistance / (1 + time * frame.fov)
        frame.frame = 0
    end
    if iskeydown("u") then
        frame.up = rotate(frame.up, frame.look, time)
        frame.frame = 0
    end
    if iskeydown("o") then
        frame.up = rotate(frame.up, frame.look, -time)
        frame.frame = 0
    end
    if iskeydown("j") then
        frame.look = rotate(frame.look, frame.up, time * frame.fov)
        frame.frame = 0
    end
    if iskeydown("l") then
        frame.look = rotate(frame.look, frame.up, -time * frame.fov)
        frame.frame = 0
    end
    if iskeydown("i") then
        frame.look = rotate(frame.look, cross(frame.up, frame.look), time * frame.fov)
        frame.frame = 0
    end
    if iskeydown("k") then
        frame.look = rotate(frame.look, cross(frame.look, frame.up), time * frame.fov)
        frame.frame = 0
    end
    if iskeydown("n") then
        frame.fov = frame.fov * (1 + time)
        frame.frame = 0
    end
    if iskeydown("m") then
        frame.fov = frame.fov / (1 + time)
        frame.frame = 0
    end
    if iskeydown("t") then
        unsetkey("t")
        result = table.save(frame, "frameSave.txt")
        if result ~= nil then
            print("Unable to save frameSave.txt")
            print(result)
        else
            print("Saved camera state")
        end
    end
    if iskeydown("y") then
        unsetkey("y")
        frameTemp, err = table.load("frameSave.txt")
        if err == nil then
            frame = frameTemp
            frame.frame = 0
            print("Loaded camera state")
        else
            print("Unable to load frameSave.txt")
            print(err)
        end
    end
    if iskeydown("b") then
        unsetkey("b")
        compile("script/mandelbox.conf.cl", "script/mandelbox.cl")
        frame.frame = 0
        print("Recompiled")
    end
    if iskeydown("p") then
        unsetkey("p")
        screenshot("fractal", frame, 2000, 2000, 200)
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
        frames[index] = frame
        print("Added frame", index)
    end
    if iskeydown("v") then
        print("Taking video")
        video(frames)
        print("Done taking video")
    end

    frame.look = normalize(frame.look)
    frame.up = normalize(cross(cross(frame.look, frame.up), frame.look))

    kernel("Main", -1, -1, "0",
    special,
    frame.pos[1], frame.pos[2], frame.pos[3],
    frame.look[1], frame.look[2], frame.look[3],
    frame.up[1], frame.up[2], frame.up[3],
    frame.fov / 800, frame.focalDistance, frame.frame)

    frame.frame = frame.frame + 1
end
