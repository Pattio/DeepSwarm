import unittest

from deepswarm.aco import Graph
from deepswarm.nodes import Node


class TestNodes(unittest.TestCase):

    def setUp(self):
        self.graph = Graph()

    def test_graph_init(self):
        # Test if the newly created graph contains the input node
        self.assertEqual(len(self.graph.topology), 1)
        self.assertEqual(self.graph.current_depth, 1)
        input_node = self.graph.input_node
        self.assertIs(self.graph.topology[0][input_node.name], input_node)

    def test_depth_increase(self):
        # Test if the depth is increased correctly
        self.assertEqual(self.graph.current_depth, 1)
        self.graph.increase_depth()
        self.assertEqual(self.graph.current_depth, 2)

    def test_path_generation(self):
        # Create a rule which selects first available node
        def select_rule(neighbours):
            return neighbours[0].node

        # Generate the path
        path = self.graph.generate_path(select_rule)
        # Test if the path is not empty
        self.assertNotEqual(path, [])
        # Test if the path starts with an input node
        self.assertEqual(path[0].type, 'Input')
        # Test if path ends with output node
        self.assertEqual(path[-1].type, 'Output')

    def test_path_completion(self):
        # Create a path containing only the input node
        old_path = [self.graph.input_node]
        # Complete that path
        new_path = self.graph.complete_path(old_path)
        # Test if path starts with an input node
        self.assertEqual(new_path[0].type, 'Input')
        # Test if path ends with output node
        self.assertEqual(new_path[-1].type, 'Output')

    def test_node_retrieval(self):
        # Test if the newly created graph contains the input node
        self.assertEqual(len(self.graph.topology), 1)
        # Retrieve first available transition from the input node
        available_transition = self.graph.input_node.available_transitions[0]
        # Use its name to initialize Node object
        available_transition_name = available_transition[0]
        available_transition_node = Node(available_transition_name)
        self.graph.get_node(available_transition_node, 1)
        # Test if graph's depth increased after adding a new node
        self.assertEqual(len(self.graph.topology), 2)
        # Test if the node was added correctly
        self.assertIs(self.graph.topology[1][available_transition_name], available_transition_node)

    def test_node_expansion(self):
        # Test if the input node was not expanded yet
        input_node = self.graph.input_node
        self.assertFalse(input_node.is_expanded)
        self.assertEqual(input_node.neighbours, [])
        # Try to expand it
        has_neighbours = self.graph.has_neighbours(input_node, 0)
        # Test if the input node was expanded successfully
        self.assertTrue(input_node.is_expanded)
        # Test if the input node has neighbours
        self.assertTrue(has_neighbours)
        self.assertNotEqual(input_node.neighbours, [])
        # Test if neighbour node was added to the topology
        neighbour_node = input_node.neighbours[0].node
        self.assertIs(self.graph.topology[1][neighbour_node.name], neighbour_node)
