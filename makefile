install:
	pip install -r requirements.txt
	python setup.py install
develop:
	pip install -r requirements.txt
	python setup.py develop
test:
	mkdir -p models
	python tests/data.py
	python tests/graphs.py
	python tests/inference.py
	python tests/models.py
	rm models/test*
	rm models/checkpoint

