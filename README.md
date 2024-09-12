# Polars Plugin for profiling

This is a introduction for writing a plugin for the Polars DataFrame library. This plugin is a simple example of how to write a plugin for Polars. The plugin is a simple profiler that can be used to profile the DataFrame operations.

## Goals

- Understand the nuances of writing ploars plugins
- How to follow good examples
- building an understanding of what can be done extending polars
  
## Steps

- Assumptions 
  - You have a basic understanding of Rust
  - You have a basic understanding of Python
  - You have Rust and Python installed
- Clone the repo
- Create a virtual environment using `python -m venv .env`
- Activate the virtual environment using `source .env/bin/activate`
- Install the dependencies using `pip install -r requirements.txt`
- For development build the environment using `maturin develop`
- Install the developed library using `pip install -e .`
- Run the `test.ipynb` notebook to see the plugin in action

