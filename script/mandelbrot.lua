require("script/plugin")
loadlib("input")

posx = 0.0
posy = 0.0
zoom = 1.0
frame = 0
juliax = -1.5
juliay = 0.5
julia = false

function special()
end

compile("script/mandelbrot.cl")

function update(time)
    if iskeydown("w") then posy = posy - zoom * time; frame = 0 end
    if iskeydown("s") then posy = posy + zoom * time; frame = 0 end
    if iskeydown("a") then posx = posx - zoom * time; frame = 0 end
    if iskeydown("d") then posx = posx + zoom * time; frame = 0 end
    if iskeydown("r") then zoom = zoom / (1 + time); frame = 0 end
    if iskeydown("f") then zoom = zoom * (1 + time); frame = 0 end
    if iskeydown("j") then julia = not julia; unsetkey("j"); frame = 0 end
    if iskeydown("u") then juliax = posx; juliay = posy end

    if iskeydown("p") then
        unsetkey("p")
        width = 1000
        height = 1000
        mkbuffer(1, width * height * 4 * 4)
        for frame=0,8 do
            local julx = 0
            local july = 0
            if julia then julx = juliax; july = juliay end
            kernel("main", width, height, "1",
                {math.floor(-width / 2)}, {math.floor(-height / 2)}, {width}, {height},
                posx, posy, zoom / width, julx, july, {frame})
            print(frame, 8)
        end
        dlbuffer(1, width)
        rmbuffer(1)
        print("Saved screenshot")
    end

    local julx = 0
    local july = 0
    if julia then julx = juliax; july = juliay end
    kernel("main", -1, -1, "0", special, posx, posy, zoom / 1024, julx, july, {frame})
    frame = frame + 1

    softsync()
end
