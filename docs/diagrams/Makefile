# Exports the draw.io diagrams: PDF for the report, PNG of the English architecture page for the README.

DRAWIO ?= drawio
PDF_DIR := ../relazione/imgs/diagrammi
DIAGRAMS := architettura ball_tree memoria simd fasi

.PHONY: all
all: $(DIAGRAMS:%=$(PDF_DIR)/%.pdf) ../img/architecture.png

$(PDF_DIR)/%.pdf: %.drawio | $(PDF_DIR)
	$(DRAWIO) --export --format pdf --crop --page-index 1 --output $@ $<

../img/architecture.png: architettura.drawio | ../img
	$(DRAWIO) --export --format png --page-index 2 --scale 2 --border 20 --output $@ $<

$(PDF_DIR) ../img:
	mkdir -p $@
