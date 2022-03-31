# Adaptive Checkpointing System Prototype
## Prerequisites

1. activate the virutal environment for encapsulated dependency management (e.g., on MacOs)

`source venv/bin/activate`

2. install this project's dependencies

_Note_: they are only tested to work on Linux

`pip3 install -r requirements.txt`

## Project Structure

- Top-level directory layout

```
.
├── src                    # source code files 
├── test                   # tests for source code
```

- Second-level directory layout

```
.
├── src                    # source code files 
│   ├── analysis          # source code for analysing the cluster trace
│   ├── clean         # source code for cleaning the cluster trace
│   └── model                # source code for training the model
└── ...
```

## Tests

### Python

```
pytest test/analysis
pytest test/clean
```

### Scala

``
mvn test
``


## Versions

| Tool            |Version|
|-----------------|---|
| Python          | 3.8.10|
| Scala           | 2.1.2 |
| Maven           | 4.0.0 |
| Spark & PySpark | 3.1.2 |
| Autosklearn     | 0.14.2 |
