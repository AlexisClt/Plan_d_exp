# Plan_d_exp

### Installation

Create venv
```
python3 -m venv env
```

Init env
```
source ./env/bin/activate
```

Install requirements
```
pip install -r requirements.txt
```

### Development

Launch unit tests using pytest
```
python3 -m pytest tests src
```

Improving the readability by sorting the imports
```
isort Plan_d_exp
```

mypy for checking type hinting
```
mypy Plan_d_exp
```
