THESISNAME=thesis
PLANNAME=projectplan

all: $(THESISNAME) $(PLANNAME) clean

$(DOCNAME):
	pdflatex $(DOCNAME).tex
	bibtex $(DOCNAME).aux
	makeglossaries $(DOCNAME)
	pdflatex $(DOCNAME).tex
	pdflatex $(DOCNAME).tex
$(THESISNAME):
	pdflatex $(THESISNAME).tex
	bibtex $(THESISNAME).aux
	makeglossaries $(THESISNAME)
	pdflatex $(THESISNAME).tex
	pdflatex $(THESISNAME).tex

$(PLANNAME):
	pdflatex $(PLANNAME).tex
	bibtex $(PLANNAME).aux
	makeglossaries $(PLANNAME)
	pdflatex $(PLANNAME).tex
	pdflatex $(PLANNAME).tex

clean:
	rm  *.acr *.acn *.alg *.glg *.glo *.gls *.ist *.out *.toc *.blg *.bbl *.aux *.log
