require("script/plugin")
loadlib("makevoxel")
loadlib("rawbuffer")

if false then
    makevoxel("voxel.dat")
end
print("Done making data file, uploading")
uplrawbuffer("1", "voxel.dat")

function update(time)

end
