.PHONY: test
test:
	pytest -v

.PHONY: readme
readme:
	catnap --help | p2c --tgt _catnap README.md && \
	catnap-create --help | p2c --tgt _catnap_create README.md && \
	catnap-assess --help | p2c --tgt _catnap_assess README.md


