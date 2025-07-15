THESISNAME=thesis
PLANNAME=projectplan
DEFENCENAME=defence
LATEX=xelatex
LATEXBEAMER=pdflatex

all: $(DEFENCENAME) $(THESISNAME) $(PLANNAME) clean

$(THESISNAME):
	$(LATEX) $(THESISNAME).tex
	biber $(THESISNAME)
	makeglossaries $(THESISNAME)
	$(LATEX) $(THESISNAME).tex
	$(LATEX) $(THESISNAME).tex
	git config --global --add safe.directory /app && exiftool -overwrite_original -Producer=$$(git rev-parse HEAD) $(THESISNAME).pdf

$(PLANNAME):
	$(LATEX) $(PLANNAME).tex
	biber $(PLANNAME)
	makeglossaries $(PLANNAME)
	$(LATEX) $(PLANNAME).tex
	$(LATEX) $(PLANNAME).tex
	git config --global --add safe.directory /app && exiftool -overwrite_original -Producer=$$(git rev-parse HEAD) $(PLANNAME).pdf

$(DEFENCENAME):
	TEXINPUTS="./beamer//:" $(LATEXBEAMER) $(DEFENCENAME).tex
	#biber $(DEFENCENAME)
	#makeglossaries $(DEFENCENAME)
	TEXINPUTS="./beamer//:" $(LATEXBEAMER) $(DEFENCENAME).tex
	TEXINPUTS="./beamer//:" $(LATEXBEAMER) $(DEFENCENAME).tex
	git config --global --add safe.directory /app && exiftool -overwrite_original -Producer=$$(git rev-parse HEAD) $(DEFENCENAME).pdf

clean:
	rm  -f *.acr *.acn *.alg *.glg *.glo *.gls *.ist *.out *.toc \
	 	   *.blg *.bbl *.aux *.log *.pdf_original *.bcf *.run.xml *.pdfpc