require("imagebuffer")
require("input")

mkbuffer(1, 0)

theta = 0
phi = 1.5707963
frame = 0

function special()
end

function recompile()
    compile("script/strangeAttractor.cl")
end

recompile()

function update(time)
    if iskeydown("w") then
        phi = phi + time
        frame = 0
    end
    if iskeydown("s") then
        phi = phi - time
        frame = 0
    end
    if iskeydown("a") then
        theta = theta + time
        frame = 0
    end
    if iskeydown("d") then
        theta = theta - time
        frame = 0
    end
    if iskeydown("t") then
        unsetkey("t")
        file = io.open("frameSave.txt", "wb")
        file:write("theta="..tostring(theta))
        file:write("phi="..tostring(phi))
        file:close()
        print("Saved state")
    end
    if iskeydown("y") then
        unsetkey("y")
        local f, err = loadfile("frameSave.txt")
        if f == nil then
            print(err)
        else
            f()
            frame = 0
            print("Loaded state")
        end
    end
    if iskeydown("p") then
        unsetkey("p")
        dlbuffer(0, 1024)
        print("Saved screenshot")
    end
    if iskeydown("b") then
        unsetkey("b")
        recompile()
    end
    if iskeydown("g") then
        unsetkey("g")
        reload()
    end

    if frame == 0 then
        print("theta", theta, "phi", phi)
    end

    kernel("main", -1, -1, "0", special, "1", { frame }, theta, phi)
    frame = frame + 1
    softsync()
end
