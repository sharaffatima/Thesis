DOCKERBIN := $(shell command -v docker 2>/dev/null)
ALLTARGET = pdf
ifdef DOCKERBIN # use docker if it is present
DOCKERIMG=danteev/texlive
DOCKER=docker run --rm -it -v $(shell pwd):/home -v $(shell pwd):/workdir $(DOCKERIMG)
ALLTARGET = docker-pdf
endif

FILENAME=thesis
MAINFILE=$(FILENAME).tex

all: $(ALLTARGET)

docker-pdf:
	$(DOCKER) make pdf

pdf:
	latexmk -shell-escape -pdf $(MAINFILE)
	makeglossaries $(FILENAME)
	latexmk -shell-escape -pdf $(MAINFILE)

count:
	@$(DOCKER) texcount -relaxed -merge -char $(MAINFILE)

clean:
	$(DOCKER) latexmk -C

setup:
	docker pull $(DOCKERIMG):latest

