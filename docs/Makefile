# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean-all serve livehtml

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Clean everything including generated API docs
clean-all: clean
	@echo "Removing generated API documentation..."
	@rm -rf api/
	@rm -rf auto_examples/
	@echo "Clean complete."

# Serve documentation locally
serve: html
	@echo "Serving documentation at http://localhost:8000"
	@cd $(BUILDDIR)/html && python -m http.server 8000

# Live reload during development (requires sphinx-autobuild)
livehtml:
	@echo "Starting live documentation server..."
	@sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O) \
		--host 0.0.0.0 --port 8000 \
		--ignore "*.swp" \
		--ignore "*.pdf" \
		--ignore "*.log" \
		--ignore "*.out" \
		--ignore "*.toc" \
		--ignore "*.aux" \
		--ignore "*.idx" \
		--ignore "*.ind" \
		--ignore "*.ilg" \
		--ignore "*.tex"

# Generate API documentation
apidoc:
	@echo "Generating API documentation..."
	@sphinx-apidoc -f -o api/ ../src/python/active_inference/ ../src/python/active_inference/tests/
	@echo "API documentation generated in api/"

# Check for broken links
linkcheck:
	@echo "Checking for broken links..."
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build documentation with warnings as errors
strict:
	@echo "Building documentation with warnings as errors..."
	@$(SPHINXBUILD) -W -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Generate PDF documentation
pdf: latex
	@echo "Building PDF documentation..."
	@cd $(BUILDDIR)/latex && make all-pdf
	@echo "PDF documentation available at $(BUILDDIR)/latex/active-inference-sim-lab.pdf"

# Check spelling (requires sphinxcontrib-spelling)
spelling:
	@echo "Checking spelling..."
	@$(SPHINXBUILD) -b spelling "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build documentation for different formats
all: html pdf epub

# Development targets
dev-install:
	@echo "Installing development dependencies..."
	@pip install sphinx sphinx-rtd-theme myst-parser sphinx-copybutton sphinx-tabs sphinx-autobuild sphinxcontrib-spelling

dev-setup: dev-install apidoc
	@echo "Development environment setup complete."

# CI/CD targets
ci-build: strict linkcheck
	@echo "CI build complete."

ci-deploy: html
	@echo "Preparing documentation for deployment..."
	@touch $(BUILDDIR)/html/.nojekyll
	@echo "Documentation ready for deployment."