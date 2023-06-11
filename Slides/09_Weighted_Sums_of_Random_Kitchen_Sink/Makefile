all: compile
clean:
	find . -name "*.aux" -type f -delete 
	rm -rf *.nav *.snm *.toc *.out *.lot *.log *.lof *.ind *.ilg *.fls *.fdb_latexmk *.brf *.blg *.bbl *.idx *.loa *.gz

compile: 
	pdflatex -output-directory=. ./article_understanding_the_difficulty_deep_learning.tex

