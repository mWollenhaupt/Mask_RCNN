
# Setup
1. Download and install [anaconda](https://www.anaconda.com/)
2. Create and activate new conda env:
	```bash
	> conda create -n NAME python=3.6 pip
    > conda activate NAME
	```
3. Clone [Mask_RCNN Repository](https://github.com/mWollenhaupt/Mask_RCNN/) and install python dependencies
	```bash 
	> python -m pip install -r requirements.txt
	```
4. Clone [COCO API - Dataset Repository](https://github.com/cocodataset/cocoapi) and modify lines in PythonAPI/setup.py from
	```python
	ext_modules = [
	    Extension(
	        'pycocotools._mask',
	        sources=['../common/maskApi.c', 'pycocotools/_mask.pyx'],
	        include_dirs = [np.get_include(), '../common'],
	        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
	    )
	]
	```
	to
	```python
	ext_modules = [
	    Extension(
	        'pycocotools._mask',
	        sources=['../common/maskApi.c', 'pycocotools/_mask.pyx'],
	        include_dirs = [np.get_include(), '../common'],
	        extra_compile_args={'gcc': ['/Qstd=c99']},
	    )
	]
	```
	run command
	```bash
	python setup.py build_ext install
	```

5. Try tu run Mask_RCNN/samples/demo.ipynb
