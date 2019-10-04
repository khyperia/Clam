#!/usr/bin/env zsh
mv *.png ~/sync/photos/Fractals
cd
unison
ssh -t carbon 'cd me/khyperia.com && make convert && make install'
