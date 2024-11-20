from .light_types import *

class LightManager:
    def __init__(self):
        self.current_light = DirectionalLight()  # 默认使用定向光
        self.lights = [self.current_light]  # 存储所有光源
        
    def set_light_type(self, light_type):
        """设置当前光源类型"""
        if light_type == LightType.DIRECTIONAL:
            if not isinstance(self.current_light, DirectionalLight):
                self.current_light = DirectionalLight()
                self.lights = [self.current_light]
    
    def get_light_data(self):
        """获取用于渲染的光照数据"""
        if not self.current_light.enabled:
            return None
            
        return {
            'type': self.current_light.type,
            'direction': self.current_light.direction,
            'color': self.current_light.color,
            'intensity': self.current_light.intensity
        }