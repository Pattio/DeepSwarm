import unittest
from deepswarm.aco import Graph
from deepswarm.nodes import Node


class TestNodes(unittest.TestCase):

    def setUp(self):
        self.graph = Graph()

    def test_graph_init(self):
        # Test if newly created graph contains the input node
        self.assertEqual(len(self.graph.topology), 1)
        self.assertEqual(self.graph.current_depth, 1)
        input_node = self.graph.input_node
        self.assertIs(self.graph.topology[0][input_node.name], input_node)

    def test_depth_increase(self):
        # Test if depth is increased correctly
        self.assertEqual(self.graph.current_depth, 1)
        self.graph.increase_depth()
        self.assertEqual(self.graph.current_depth, 2)

    def test_path_generation(self):
        # Create rule which selects first available node
        def select_rule(neighbours):
            return neighbours[0].node
        # Generate the path
        path = self.graph.generate_path(select_rule)
        # Test if path is not empty
        self.assertNotEqual(path, [])
        # Test if path start with input node
        self.assertEqual(path[0].type, 'Input')
        # Test if path ends with end node
        self.assertEqual(path[-1].type, 'Output')

    def test_path_completion(self):
        # Create path containing only the input node
        old_path = [self.graph.input_node]
        # Complete that path
        new_path = self.graph.complete_path(old_path)
        # Test if path start with input node
        self.assertEqual(new_path[0].type, 'Input')
        # Test if path ends with end node
        self.assertEqual(new_path[-1].type, 'Output')

    def test_node_retrieval(self):
        # Test if newly created graph contains the input node
        self.assertEqual(len(self.graph.topology), 1)
        # Retrieve first available transition from the input node
        available_transition = self.graph.input_node.available_transitions[0]
        # Use its name to initialize Node object
        available_transition_name = available_transition[0]
        available_transition_node = Node(available_transition_name)
        self.graph.get_node(available_transition_node, 1)
        # Test if graph depth increased after adding new node
        self.assertEqual(len(self.graph.topology), 2)
        # Test if node was added correctly
        self.assertIs(self.graph.topology[1][available_transition_name], available_transition_node)

    def test_node_expansion(self):
        # Test if input node was not expanded yet
        input_node = self.graph.input_node
        self.assertFalse(input_node.is_expanded)
        self.assertEqual(input_node.neighbours, [])
        # Try to expand it
        has_neighbours = self.graph.has_neighbours(input_node, 0)
        # Test if input node was expanded successfully
        self.assertTrue(input_node.is_expanded)
        # Test if input node has neighbours
        self.assertTrue(has_neighbours)
        self.assertNotEqual(input_node.neighbours, [])
        # Test if neighbour node was added to the topology
        neighbour_node = input_node.neighbours[0].node
        self.assertIs(self.graph.topology[1][neighbour_node.name], neighbour_node)
