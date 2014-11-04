require("script/plugin")
loadlib("imagebuffer")
loadlib("input")

totalTime = 0

function special()
end

function recompile()
    compile("script/colors.cl")
end

recompile()

function update(time)
    if iskeydown("p") then
        unsetkey("p")
        width = 1000
        height = 1000
        mkbuffer(1, width * height * 4 * 4)
        kernel("main", width, height, "1",
            {math.floor(-width / 2)}, {math.floor(-height / 2)}, {width}, {height},
            totalTime)
        dlbuffer(1, width)
        rmbuffer(1)
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

    kernel("main", -1, -1, "0", special, totalTime)
    totalTime = totalTime + time;
    softsync()
end
