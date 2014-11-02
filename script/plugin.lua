-- Usage:
-- Write a C script called "script/yourscript.c" containing a function
--  "int yourscript(lua_State* state)" (name of file and name of function must be the same)
-- Run "make", this will compile the .c file into an .so
-- Compilation flags can be edited by making a script/yourscript.mk file containing for example:
-- script/imagebuffer.so: LDFLAGS_SO:=$(LDFLAGS_SO) -llua
-- Call `loadlib("yourscript")` - this will run yourscript() and make available anything
--  registered as a global in yourscript()

require("package")

function loadlib(soname)
    local f, err = package.loadlib("script/" .. soname .. ".so", soname);
    if err ~= nil then
        error("Failed to open library - " .. err)
    end
    f();
end
