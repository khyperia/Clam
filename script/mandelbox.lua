require("math")
require("script/vector")

pos = {0, 0, 5}
look = {0, 0, -1}
up = {0, 1, 0}
fov = 1.0
focalDistance = 1.0
frame = 0.0

function special()
end

compile("script/mandelbox.conf.cl", "script/mandelbox.cl")

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
        width = 5000
        height = 5000
        mkbuffer("screenshot", width * height * 4 * 4)
        numframes = 100
        for i=0,numframes do
            kernel("Main", width, height, "screenshot",
                    {math.floor(-width / 2)}, {math.floor(-height / 2)}, {width}, {height},
                    pos[1], pos[2], pos[3],
                    look[1], look[2], look[3],
                    up[1], up[2], up[3],
                    fov / width, focalDistance, i)
            print((i / numframes * 100), "% done");
        end
        dlbuffer("screenshot", width)
        rmbuffer("screenshot")
    end

    look = normalize(look)
    up = normalize(cross(cross(look, up), look))
    
    kernel("Main", -1, -1, "",
           special,
           pos[1], pos[2], pos[3],
           look[1], look[2], look[3],
           up[1], up[2], up[3],
           fov / 1000, focalDistance, frame)

    frame = frame + 1
end
