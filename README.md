<p align="center">
  <img src="https://user-images.githubusercontent.com/9087174/54948416-7f0d9100-4f34-11e9-8ae4-5383e9a75c72.png" width="300">
</p>

<p align="center">
  <strong>Neural Architecture Search Powered by Swarm Intelligence</strong>
</p>


# DeepSwarm 

DeepSwarm is an open-source library which uses Swarm Intelligence to tackle the neural architecture search problem. 

## Installation 

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

## Configuration 

## Acknowledgments

DeepSwarm was developed under the supervision of [Dr Wei Pang](https://www.abdn.ac.uk/ncs/people/profiles/pang.wei) in partial fulfilment of the requirements for the degree of Bachelor of Science of the [University of Aberdeen](https://www.abdn.ac.uk).
