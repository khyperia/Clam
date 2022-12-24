Fractals created by this program can be viewed [here](https://khyperia.com/fractals.html)

Note this program is not user-friendly at all, and is very specialized to my own use cases. However, if you do want to
run it yourself, the standard `cargo run --release` is how to do so. Pressing `h` will at least dump a list of
keybindings to console :)

building for wasm:

    cd wasm
    wasm-pack build --target web
    python web/server.py
    # or, in the root
    wasm-pack build wasm --target web
