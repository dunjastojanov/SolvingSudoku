# SolvingSudoku

Contributors:
- Aleksandra Nedić (SW 27/2019)
- Dunja Stojanov (SW 30/2019)

Problem:
Recognising and solving sudoku puzzle from image using convolutional neural networks

Algorithms:
CNN and backtracking

Data:
- http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
- https://icosys.ch/sudoku-dataset

Validation:
- 80% of data used for training the model
- 20% of data used for validation
- Validation of the solved table using py-sudoku library

Python version:
Python 3.7

To start up the project do the following steps:
1. Create virtual environment and activate it.
2. Run the pip command bellow to install all the required packages (Requirements are listed inside the requirements.txt file).

    pip install -r requirements.txt

3. Unpack digits.rar
4. Open project in PyCharm configure Pithon interpreter from your new virtual environment.
5. To solve Sudoku puzzle run script detect_digits.py. To train model run script training.py.

