posx = 0.0
posy = 0.0
zoom = 1.0

function special()
end

compile("script/mandelbrot.cl")

function update(time)
    if iskeydown("w") then posy = posy - zoom * time end
    if iskeydown("s") then posy = posy + zoom * time end
    if iskeydown("a") then posx = posx - zoom * time end
    if iskeydown("d") then posx = posx + zoom * time end
    if iskeydown("r") then zoom = zoom / (1 + time) end
    if iskeydown("f") then zoom = zoom * (1 + time) end

    if iskeydown("p") then
        unsetkey("p")
        width = 10000
        height = 10000
        mkbuffer(1, width * height * 4 * 4)
        kernel("main", width, height, "1",
            {math.floor(-width / 2)}, {math.floor(-height / 2)}, {width}, {height},
            posx, posy, zoom / width)
        dlbuffer(1, width)
        rmbuffer(1)
        print("Saved screenshot")
    end

    kernel("main", -1, -1, "0", special, posx, posy, zoom / 500)
end
