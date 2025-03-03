THESISNAME=thesis
PLANNAME=projectplan

all: $(THESISNAME) $(PLANNAME) clean

$(THESISNAME):
	pdflatex $(THESISNAME).tex
	bibtex $(THESISNAME).aux
	makeglossaries $(THESISNAME)
	pdflatex $(THESISNAME).tex
	pdflatex $(THESISNAME).tex
	git config --global --add safe.directory /app && exiftool -XMP:Version=$(shell git rev-parse HEAD) $(THESISNAME).pdf

$(PLANNAME):
	pdflatex $(PLANNAME).tex
	bibtex $(PLANNAME).aux
	makeglossaries $(PLANNAME)
	pdflatex $(PLANNAME).tex
	pdflatex $(PLANNAME).tex
	git config --global --add safe.directory /app && exiftool -XMP:Version=$(shell git rev-parse HEAD) $(PLANNAME).pdf

clean:
	rm  *.acr *.acn *.alg *.glg *.glo *.gls *.ist *.out *.toc *.blg *.bbl *.aux *.log *.pdf_original
