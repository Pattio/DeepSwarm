.PHONY: test clean upload

test:
	python -m unittest discover tests

clean:
	rm -rf build *.egg-info dist
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

upload: clean
	python setup.py sdist bdist_wheel
	twine upload dist/*