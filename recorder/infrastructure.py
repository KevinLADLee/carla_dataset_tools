#!/usr/bin/python3
import carla
from recorder.actor import PseudoActor


class Infrastructure(PseudoActor):
    def __init__(self,
                 uid,
                 name: str,
                 base_save_dir: str):
        super().__init__(uid=uid, name=name, parent=None)
        self.save_dir = '{}/{}_{}'.format(base_save_dir, self.get_uid(), self.get_type_id())

    def get_type_id(self):
        return 'v2i.infrastructure'

    def get_save_dir(self):
        return self.save_dir

