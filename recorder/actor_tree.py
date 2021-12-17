#!/usr/bin/python3
import carla
from recorder.actor_factory import ActorFactory, Node


class ActorTree(object):
    def __init__(self, world: carla.World, actor_config_file=None, base_save_dir=None):
        self.world = world
        self.actor_config_file = actor_config_file
        self.actor_factory = ActorFactory(self.world, base_save_dir)
        self.root = Node(None)

    def init(self):
        self.root = self.actor_factory.create_actor_tree(self.actor_config_file)

    def destroy(self):
        self.root.destroy()

    def add_node(self, node):
        self.root.add_child(node)

    def tick_controller(self):
        for v2i_layer_node in self.root.get_children():
            v2i_layer_node.tick_controller()

    def tick_data_saving(self, frame_id):
        for v2i_layer_node in self.root.get_children():
            v2i_layer_node.tick_data_saving(frame_id)
            for sensor_layer_node in v2i_layer_node.get_children():
                sensor_layer_node.tick_data_saving(frame_id)

    def print_tree(self):
        print("------ Actor Tree BEGIN ------")
        for node in self.root.get_children():
            print("- {}".format(node.get_actor().name))
            for child_node in node.get_children():
                if child_node is not None:
                    print("|- {}".format(child_node.get_actor().name))
        print("------ Actor Tree END ------")
