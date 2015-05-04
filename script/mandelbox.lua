require("math")
require("imagebuffer")
require("input")
require("vector")

function table.shallow_copy(t)
  local t2 = {}
  for k,v in pairs(t) do
    t2[k] = v
  end
  return t2
end

-- TODO: Make this dynamic
AssumedScreenWidth = 1024

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

function recompile()
    compile("script/mandelbox.conf.cl", "script/mandelbox.cl")
end

recompile()
mkbuffer(1, 0) -- RNG buffer

function screenshot(bufferIndex, frame, width, height, numframes)
    mkbuffer(bufferIndex, width * height * 4 * 4)
    mkbuffer(bufferIndex + 1, width * height * 4 * 4)
    local timeout = setsynctimeout(-1);
    i = 0
    while not iskeydown(" ") do
        if iskeydown("p") then
            unsetkey("p")
            print("Downloading intermediate frame at ", i)
            dlbuffer(bufferIndex, width)
        end
        kernel("Main", width, height, tostring(bufferIndex), tostring(bufferIndex + 1),
            {math.floor(-width / 2)}, {math.floor(-height / 2)}, {width}, {height},
            frame.pos[1], frame.pos[2], frame.pos[3],
            frame.look[1], frame.look[2], frame.look[3],
            frame.up[1], frame.up[2], frame.up[3],
            frame.fov / width, frame.focalDistance, i)
        print("frame", i)
        i = i + 1
        softsync()
        pumpevents() -- for iskeydown
    end
    unsetkey(" ")
    setsynctimeout(timeout);
    dlbuffer(bufferIndex, width)
    rmbuffer(bufferIndex + 1)
    rmbuffer(bufferIndex)
end

function video(frames)
    function videoFrameAt(time)
        local fractional = time % 1
        local whole = math.floor(time)
        local frame1 = frames[whole]
        local frame2 = frames[whole + 1]
        if frame1 == nil or frame2 == nil then
            return nil
        end
        local frame0 = frames[whole - 1]
        if frame0 == nil then frame0 = frame1 end
        local frame3 = frames[whole + 2]
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
        screenshot(frameIndex + 2, frame, 800, 500, 20)
        print("Took frame", frameIndex)
        frameIndex = frameIndex + 1
    end
end

function update(time)
    frame = update3dCamera(frame, time)

    if iskeydown("b") then
        unsetkey("b")
        recompile()
        frame.frame = 0
        print("Recompiled")
    end
    if iskeydown("g") then
        unsetkey("g")
        reload()
    end
    if iskeydown("p") then
        unsetkey("p")
        print("Downloading screenbuffer at frame " .. tostring(frame.frame))
        dlbuffer(0, AssumedScreenWidth) -- issues if screen isn't this size
    end
    if iskeydown("1") then
        unsetkey("1")
        screenshot(2, frame, math.pow(2, 1 + 6), math.pow(2, 1 + 6), 200)
    end
    if iskeydown("2") then
        unsetkey("2")
        screenshot(2, frame, math.pow(2, 2 + 6), math.pow(2, 2 + 6), 500)
    end
    if iskeydown("3") then
        unsetkey("3")
        screenshot(2, frame, math.pow(2, 3 + 6), math.pow(2, 3 + 6), 1000)
    end
    if iskeydown("4") then
        unsetkey("4")
        screenshot(2, frame, math.pow(2, 4 + 6), math.pow(2, 4 + 6), 1000)
    end
    if iskeydown("5") then
        unsetkey("5")
        screenshot(2, frame, math.pow(2, 5 + 6), math.pow(2, 5 + 6), 1000)
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
        frames[index] = table.shallow_copy(frame)
        print("Added frame", index)
    end
    if iskeydown("v") then
        print("Taking video")
        video(frames)
        print("Done taking video")
        unsetkey("v")
    end
    if iskeydown("h") then
        unsetkey("h")
        print("wasd, space/z: move")
        print("ijkl(uo|qe): rotate")
        print("rf: speed/focal distance")
        print("nm: field of view")
        print("t: save state, y: load state")
        print("b: recompile kernel, g: recompile lua")
        print("p: screenshot of framebuffer")
        print("c: add keyframe, x: clear keyframes")
        print("v: video through keyframes")
        print("1-6: screenshot 2^(n + 6) pixels square")
        print("h: this message")
        print("\\: toggle VRPN control")
    end

    kernel("Main", -1, -1, "0", "1",
    special,
    frame.pos[1], frame.pos[2], frame.pos[3],
    frame.look[1], frame.look[2], frame.look[3],
    frame.up[1], frame.up[2], frame.up[3],
    frame.fov / AssumedScreenWidth, frame.focalDistance, frame.frame)

    frame.frame = frame.frame + 1

    softsync()
end
