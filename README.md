# Paintings
 
This is my Master Thesis project. The goal is to approximate a given painting by using one of three optimization algorithms. The three algorithms are:
 
- Stochastic Hillclimber
- Simulated Annealing
- Plant propagation algorithm
 
 
## Installation
 
Use pip3 to install the necessary python3 packages, such as:
 
```bash
pip3 install opencv-python
```
 
## Usage
 
All three algorithms have the same usage commands:
 
```python
python3 algorithm.py filename
```
 
The script expects a file from the **imgs** directory.
 
### Examples
To run the Hillclimber algorithm for the Mona Lisa use:
 
```
python3 HC.py mona.png
```
To run the Simulated annealing algorithm for the Starry night use:
 
```
python3 SA.py starrynight.png
```
 
To run the plant propagation algorithm for the painting of Salvador Dalí use:
 
```
python3 PPA.py dali.png
```
 
All results are stored in output_dir per painting.
 
 

