compile: 
	pandoc Memory/*.md --pdf-engine=xelatex -o Compilations/Memory.pdf --bibliography=bibliography.bib --citeproc --template eisvogel --listings --toc
	open Compilations/Memory.pdf
open: 
	open Material