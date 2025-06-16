THESISNAME=thesis
PLANNAME=projectplan

all: $(THESISNAME) $(PLANNAME) clean

$(THESISNAME):
	pdflatex $(THESISNAME).tex
	biber $(THESISNAME)
	makeglossaries $(THESISNAME)
	pdflatex $(THESISNAME).tex
	pdflatex $(THESISNAME).tex
	git config --global --add safe.directory /app && exiftool -overwrite_original -Producer=$$(git rev-parse HEAD) $(THESISNAME).pdf

$(PLANNAME):
	pdflatex $(PLANNAME).tex
	biber $(PLANNAME)
	makeglossaries $(PLANNAME)
	pdflatex $(PLANNAME).tex
	pdflatex $(PLANNAME).tex
	git config --global --add safe.directory /app && exiftool -overwrite_original -Producer=$$(git rev-parse HEAD) $(PLANNAME).pdf

clean:
	rm  -f *.acr *.acn *.alg *.glg *.glo *.gls *.ist *.out *.toc *.blg *.bbl *.aux *.log *.pdf_original *.bcf *.run.xml