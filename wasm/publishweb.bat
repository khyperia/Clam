scp pkg/wasm.js pkg/wasm_bg.wasm web/index.html carbon:me/khyperia.com/clam5/
ssh -t carbon "cd me/khyperia.com && sed -i 's/..\/pkg\//.\//' clam5/index.html && make install"
