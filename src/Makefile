DOCNAME=main

all: $(DOCNAME)

$(DOCNAME):
	pdflatex $(DOCNAME).tex
	bibtex $(DOCNAME).aux
	makeglossaries $(DOCNAME)
	pdflatex $(DOCNAME).tex
	pdflatex $(DOCNAME).tex

clean:
	rm *.blg *.bbl *.aux *.log
