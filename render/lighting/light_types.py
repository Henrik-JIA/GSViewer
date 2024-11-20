from enum import Enum
import numpy as np

class LightType(Enum):
    NONE = 0
    DIRECTIONAL = 1
    POINT = 2
    SPOT = 3
    AREA = 4
    AMBIENT = 5
    HEMISPHERE = 6
    IBL = 7

class Light:
    def __init__(self):
        self.type = LightType.NONE
        self.color = [1.0, 1.0, 1.0, 1.0]
        self.intensity = 1.0
        self.enabled = True

class DirectionalLight(Light):
    def __init__(self):
        super().__init__()
        self.type = LightType.DIRECTIONAL
        self.direction = [0.0, -1.0, 0.0]

# ... 其他光源类型的定义 ...