import unittest

from deepswarm.nodes import Node, NodeAttribute, NeighbourNode


class TestNodes(unittest.TestCase):

    def setUp(self):
        self.input_node = Node.create_using_type('Input')

    def test_create_using_type(self):
        # Test default values
        self.assertEqual(self.input_node.neighbours, [])
        self.assertEqual(self.input_node.type, 'Input')
        self.assertFalse(self.input_node.is_expanded)
        self.assertNotEqual(self.input_node.attributes, [])
        self.assertNotEqual(self.input_node.available_transitions, [])
        # Test if generated description is correct
        description = self.input_node.name + '(' + 'shape:' + str(self.input_node.shape) + ')'
        self.assertEqual(description, str(self.input_node))

    def test_init(self):
        # Test if you can create node just by using its name
        input_node_new = Node(self.input_node.name)
        self.assertEqual(input_node_new.type, self.input_node.type)

    def test_deepcopy(self):
        # Test if the copied object is an instance of Node
        input_node_copy = self.input_node.create_deepcopy()
        self.assertIsInstance(input_node_copy, Node)
        # Test if unnecessary attributes were removed
        self.assertNotEqual(input_node_copy.available_transitions, self.input_node.attributes)
        # Test if unnecessary attributes are empty arrays
        self.assertEqual(input_node_copy.neighbours, [])
        self.assertEqual(input_node_copy.available_transitions, [])

    def test_available_transition(self):
        # Retrieve first available transition
        available_transition = self.input_node.available_transitions[0]
        # Use its name to initialize Node object
        available_transition_name = available_transition[0]
        available_transition_node = Node(available_transition_name)
        self.assertIsInstance(available_transition_node, Node)
        # Check if the node was properly initialized
        self.assertNotEqual(available_transition_node.attributes, [])
        self.assertNotEqual(available_transition_node.available_transitions, [])
        # Check if available transition contains a heuristic value
        self.assertIsInstance(available_transition[1], float)

    def test_custom_attribute_selection(self):
        # Initialize node which connects to the input node
        node = Node(self.input_node.available_transitions[0][0])
        # For each attribute select first available value
        node.select_custom_attributes(lambda values: values[0][0])
        # Collect selected values
        old_attribute_values = [getattr(node, attribute.name) for attribute in node.attributes]
        # For each attribute if available select second value
        node.select_custom_attributes(lambda values: values[1][0] if len(values) > 1 else values[0][0])
        # Collect newly selected values
        new_attribute_values = [getattr(node, attribute.name) for attribute in node.attributes]
        # Newly selected values should be different from old values
        self.assertNotEqual(old_attribute_values, new_attribute_values)

    def test_adding_neighbour_node(self):
        # Find first available transition
        transition_name, transition_pheromone = self.input_node.available_transitions[0]
        # Initialize node object using transition's name
        node = Node(transition_name)
        # Create NeighbourNode object
        neighbour_node = NeighbourNode(node, transition_pheromone)
        # Check if NeighbourNode object was created properly
        self.assertIsInstance(neighbour_node.node, Node)
        self.assertIsInstance(neighbour_node.heuristic, float)
        self.assertIsInstance(neighbour_node.pheromone, float)
        # Add NeighbourNode object to neighbours list
        self.input_node.neighbours.append(neighbour_node)

    def test_node_attributes_init(self):
        # Create test attribute
        attribute_name = 'filter_count'
        attribute_values = [16, 32, 64]
        attribute = NodeAttribute(attribute_name, attribute_values)
        # Check if attribute name was set correctly
        self.assertEqual(attribute.name, attribute_name)
        # Check if each attribute value was added to the dictionary
        for attribute_value in attribute_values:
            self.assertIn(attribute_value, attribute.dict)
        # Gather all unique pheromone values
        pheromone_values = list(set(attribute.dict.values()))
        # Because NodeAttribute object was just initialized and no changes to
        # pheromone values were performed, all pheromone values must be the same
        # meaning that pheromone_values must contain only 1 element
        self.assertEqual(len(pheromone_values), 1)
