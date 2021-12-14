#!/usr/bin/python3
import carla
from recorder.actor import PseudoActor


class TrafficLight(PseudoActor):
    def __init__(self, uid, name, parent):
        super().__init__(uid, name, parent)
        # TODO