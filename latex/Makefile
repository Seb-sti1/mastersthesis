DOCNAME=main

all: $(DOCNAME)

$(DOCNAME):
	pdflatex $(DOCNAME).tex
	bibtex $(DOCNAME).aux
	makeglossaries $(DOCNAME)
	pdflatex $(DOCNAME).tex
	pdflatex $(DOCNAME).tex

clean:
	rm  *.acr *.acn *.alg *.glg *.glo *.gls *.ist *.out *.toc *.blg *.bbl *.aux *.log
