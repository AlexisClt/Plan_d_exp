# Plan_d_exp

### Installation

Create venv
```
python3 -m venv env
```

Init env on linux
```
source ./env/bin/activate
```

Init env on WINDOWS
```
.\Wenv\Scripts\activate.bat
```

Install requirements
```
pip install -r requirements.txt
```

### Examples

Some examples are available in the folder `examples`, see the `README` in this folder.

### Development

Launch unit tests using pytest on linux
```
python3 -m pytest tests src
```

Launch unit tests using pytest on WINDOWS
```
python3 -m pytest .\Plan_d_exp\
```


Improving the readability by sorting the imports
```
isort Plan_d_exp
```

mypy for checking type hinting
```
mypy Plan_d_exp
```
