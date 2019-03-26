<p align="center">
  <img src="https://user-images.githubusercontent.com/9087174/54948416-7f0d9100-4f34-11e9-8ae4-5383e9a75c72.png" width="300">
</p>

<p align="center">
  <strong>Neural Architecture Search Powered by Swarm Intelligence</strong>
</p>


# DeepSwarm 

DeepSwarm is an open-source library which uses Swarm Intelligence to tackle the neural architecture search problem. 

## Example 

```python
from deepswarm.backends import Dataset, TFKerasBackend
from deepswarm.deepswarm import DeepSwarm

dataset = Dataset(training_examples=x_train, training_labels=y_train, testing_examples=x_test, 
    testing_labels=y_test, validation_split=0.1)

backend = TFKerasBackend(dataset=dataset)
deepswarm = DeepSwarm(backend=backend)
topology = deepswarm.find_topology()
trained_topology = deepswarm.train_topology(topology, 50)

```

## Installation 

### Using pip

1. Install the package
   ```sh
   pip install deepswarm
   ```
2. Install one of the implemented backends that you want to use
   ```sh
   pip install tensorflow==1.13.1
   ```

### Using GitHub

1. Clone the repository from the GitHub

   ```sh
   git clone https://github.com/Pattio/DeepSwarm.git
   ```
2. Change the current directory to the repository directory
   ```sh
   cd deepswarm
   ```
3. Create and activate virtual environment (optional)
   ```sh
   python3 -m venv deepswarm-env && source deepswarm-env/bin/activate
   ```
4. Install the external dependencies 
   ```sh
   pip install -r requirements.txt
   ```
5. Install one of the implemented backends that you want to use
   ```sh
   pip install tensorflow==1.13.1
   ```


## Configuration 

## Acknowledgments

DeepSwarm was developed under the supervision of [Dr Wei Pang](https://www.abdn.ac.uk/ncs/people/profiles/pang.wei) in partial fulfilment of the requirements for the degree of Bachelor of Science of the [University of Aberdeen](https://www.abdn.ac.uk).
