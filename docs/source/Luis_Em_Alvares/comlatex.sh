#!/bin/bash

rm *.aux
rm *.log
rm *.nav
rm *.out
rm *.snm
rm *.toc

file=$1

pdflatex ${file}".tex"

open ${file}".pdf"

