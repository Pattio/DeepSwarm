<p align="center">
  <img src="https://user-images.githubusercontent.com/9087174/55276558-066c5300-52ed-11e9-8bb6-284948cdef67.png" width="300">
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
   pip install tensorflow-gpu==1.13.1
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
   pip install tensorflow-gpu==1.13.1
   ```


## Configuration 

| Node type      | Attributes  |
| :------------- |:-------------|
| Input | **shape**: tuple which defines the input shape, depending on the backend could be (width, height, channels) or (channels, width, height). |
| Conv2D | **filter_count**: defines how many filters can be used. <br> **kernel_size**: defines what size kernels can be used. For example, if it is set to [1, 3], then only 1x1 and 3x3 kernels will be used. <br> **activation**: defines what activation functions can be used. Allowed values are: ReLU, ELU, LeakyReLU, Sigmoid and Softmax. |
| Dropout | **rate**: defines the allowed dropout rates. For example, if it is set to [0.1, 0.3], then either 10% or 30% of input units will be dropped. |
| BatchNormalization | - |
| Pool2D | **pool_type**: defines the types of allowed pooling nodes. Allowed values are: max (max pooling) and average (average pooling). <br> **pool_size**: defines the allowed pooling window sizes. For example, if it is set to [2], then only 2x2 pooling windows will be used. <br> **stride**: defines the allowed stride sizes. |
| Flatten | - |
| Dense | **output_size**: defines the allowed output space dimensionality. <br> **activation**: defines what activation functions can be used. Allowed values are: ReLU, ELU, LeakyReLU, Sigmoid and Softmax. |
| Output | **output_size**: defines the output size (how many different classes to classify). <br> **activation**: defines what activation functions can be used. Allowed value are ReLU, ELU, LeakyReLU, Sigmoid and Softmax. |

| Setting        | Description |
| :------------- |:-------------|
| save_folder | Specifies the name of the folder which should be used to load the backup. If not specified the search will start from zero. |
| metrics | Specifies what metrics should algorithm use to evaluate the models. Currently available options are: accuracy and loss. |
| max_depth | Specifies the maximum allowed network depth (how deeply the graph can be expanded). The search is performed until the maximum depth is reached. However, it does not mean that the depth of the best architecture will be equal to the max_depth. |
| reuse_patience | Specifies the maximum number of times that weights can be reused without improving the cost. For example, if it is set to 1 it means that when some model X reuses weights from model Y and model X cost did not improve compared to model Y, next time instead of reusing model Y weights, new random weights will be generated.|
| start | Specifies the starting pheromone value for all the new connections. |
| decay | Specifies the local pheromone decay rate in percentage. For example, if it is set to 0.1 it means that during the local pheromone update the pheromone value will be decreased by 10%. |
| evaporation | Specifies the global pheromone evaporation rate in percentage. For example, if it is set to 0.1 it means that during the global pheromone update the pheromone value will be decreased by 10%. |
| greediness | Specifies how greedy should ants be during the edge selection (number is given in percentage). For example, 0.5 means that 50% of the time when ant selects a new edge it should select the one with highest associated probability. |
| ant_count | Specifies how many ants should be generated during each generation (time before the depth is increased). |
| epochs | Specifies for how many epochs each candidate architecture should be trained. |
| batch_size | Specifies the batch size (number of samples used to calculate a single gradient step) used during the training process. |
| patience | Specifies the early stopping number used during the training (after how many epochs when cost is not improving the training process should be stopped). |
| loss | Specifies what loss function should be used during the training. Currently available options are sparse_categorical_crossentropy and categorical_crossentropy. |
| spatial_nodes | Specifies which nodes are placed before the flattening node. Values in this array must correspond to node names. |
| flat_nodes | Specifies which nodes are placed after the flattening node (array should also include the flattening node). Values in this array must correspond to node names. |


## Acknowledgments

DeepSwarm was developed under the supervision of [Dr Wei Pang](https://www.abdn.ac.uk/ncs/people/profiles/pang.wei) in partial fulfilment of the requirements for the degree of Bachelor of Science of the [University of Aberdeen](https://www.abdn.ac.uk).
