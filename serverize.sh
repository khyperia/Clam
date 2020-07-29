#!/usr/bin/env bash
mv *.png ~/sync/photos/Fractals
cd
unison
ssh -t carbon 'cd me/khyperia.com && make convert && make install'
