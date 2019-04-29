import math
import unittest

from deepswarm import cfg
from deepswarm.aco import ACO, Ant


class TestACO(unittest.TestCase):

    def setUp(self):
        self.aco = ACO(None, None)

    def test_ant_init(self):
        # Test if the ant is initialized properly
        ant = Ant()
        self.assertEqual(ant.loss, math.inf)
        self.assertEqual(ant.accuracy, 0.0)
        self.assertEqual(ant.path, [])
        if cfg['metrics'] == 'loss':
            self.assertEqual(ant.cost, ant.loss)
        else:
            self.assertEqual(ant.cost, ant.accuracy)

    def test_ant_init_with_path(self):
        # Test if the ant is initialized properly when a path is given
        self.aco.graph.increase_depth()
        path = self.aco.graph.generate_path(self.aco.aco_select)
        ant = Ant(path)
        self.assertEqual(ant.loss, math.inf)
        self.assertEqual(ant.accuracy, 0.0)
        self.assertEqual(ant.path, path)

    def test_ant_comparison(self):
        # Test if ants are compared properly
        ant_1 = Ant()
        ant_1.accuracy = 0.8
        ant_2 = Ant()
        ant_2.loss = 0.8
        self.assertTrue(ant_2 < ant_1)

    def test_local_update(self):
        # Test if local update rule works properly
        new_value = self.aco.local_update(11.23, None)
        self.assertEqual(new_value, 10.117)

    def test_global_update(self):
        # Test if global update rule works properly
        new_value = self.aco.global_update(11.23, 13.79)
        self.assertEqual(new_value, 11.486)

    def test_pheromone_update(self):
        # Test pheromone update
        self.aco.graph.increase_depth()
        self.aco.graph.increase_depth()
        path = self.aco.graph.generate_path(self.aco.aco_select)
        ant = Ant(path)
        self.aco.update_pheromone(ant, self.aco.local_update)
        self.aco.update_pheromone(ant, self.aco.global_update)
