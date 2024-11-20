import numpy as np
from math import cos, sin, radians, pi
from typing import List, Tuple, Optional, Dict, Any
import cv2  # 用于图像处理
from .light_types import LightType

class LightUtils:
    @staticmethod
    def normalize_vector(vector: List[float]) -> List[float]:
        """
        标准化向量
        Args:
            vector: 输入向量 [x, y, z]
        Returns:
            标准化后的向量
        """
        length = np.sqrt(sum(x*x for x in vector))
        if length > 0:
            return [x/length for x in vector]
        return vector

    @staticmethod
    def calculate_attenuation(distance: float, constant: float = 1.0, 
                            linear: float = 0.09, quadratic: float = 0.032) -> float:
        """
        计算点光源和聚光灯的衰减因子
        Args:
            distance: 到光源的距离
            constant: 常数项系数
            linear: 一次项系数
            quadratic: 二次项系数
        Returns:
            衰减因子
        """
        return 1.0 / (constant + linear * distance + quadratic * distance * distance)

    @staticmethod
    def calculate_spot_light_intensity(light_direction: List[float], 
                                     spot_direction: List[float],
                                     inner_cutoff: float,
                                     outer_cutoff: float) -> float:
        """
        计算聚光灯的强度
        Args:
            light_direction: 从光源指向片段的方向向量
            spot_direction: 聚光灯的方向向量
            inner_cutoff: 内切角(度)
            outer_cutoff: 外切角(度)
        Returns:
            聚光灯强度 [0,1]
        """
        # 确保向量已标准化
        light_direction = np.array(LightUtils.normalize_vector(light_direction))
        spot_direction = np.array(LightUtils.normalize_vector(spot_direction))
        
        cos_theta = np.dot(-light_direction, spot_direction)
        cos_inner = cos(radians(inner_cutoff))
        cos_outer = cos(radians(outer_cutoff))
        
        epsilon = cos_inner - cos_outer
        if epsilon == 0:
            return 0.0
            
        intensity = np.clip((cos_theta - cos_outer) / epsilon, 0.0, 1.0)
        return float(intensity)

    @staticmethod
    def calculate_area_light_contribution(surface_point: List[float], 
                                        light_position: List[float],
                                        light_direction: List[float],
                                        light_size: List[float],
                                        samples: int = 16) -> Dict[str, Any]:
        """
        计算面光源的贡献
        Args:
            surface_point: 表面点位置
            light_position: 面光源中心位置
            light_direction: 面光源方向
            light_size: 面光源尺寸 [width, height]
            samples: 采样点数量
        Returns:
            包含光照信息的字典
        """
        # 构建面光源的坐标系
        forward = np.array(LightUtils.normalize_vector(light_direction))
        right = np.cross(forward, [0, 1, 0])
        right = LightUtils.normalize_vector(right.tolist())
        up = np.cross(np.array(right), forward)
        up = LightUtils.normalize_vector(up.tolist())

        total_contribution = 0.0
        width, height = light_size
        
        # 使用分层采样
        samples_sqrt = int(np.sqrt(samples))
        for i in range(samples_sqrt):
            for j in range(samples_sqrt):
                # 计算采样点位置
                u = (i + np.random.random()) / samples_sqrt - 0.5
                v = (j + np.random.random()) / samples_sqrt - 0.5
                
                # 计算世界空间中的采样点位置
                sample_pos = np.array(light_position) + \
                           np.array(right) * (u * width) + \
                           np.array(up) * (v * height)
                
                # 计算从表面点到采样点的向量
                to_light = sample_pos - np.array(surface_point)
                distance = np.linalg.norm(to_light)
                direction = to_light / distance
                
                # 累加贡献
                contribution = max(0, np.dot(direction, forward))
                contribution *= LightUtils.calculate_attenuation(distance)
                total_contribution += contribution

        # 平均所有采样点的贡献
        total_contribution /= samples
        
        return {
            'contribution': total_contribution,
            'average_direction': LightUtils.normalize_vector(
                (np.array(light_position) - np.array(surface_point)).tolist()
            )
        }

    @staticmethod
    def load_hdr_environment_map(file_path: str) -> Optional[np.ndarray]:
        """
        加载HDR环境贴图
        Args:
            file_path: HDR文件路径
        Returns:
            加载的HDR图像数据，加载失败返回None
        """
        try:
            # 使用OpenCV加载HDR图像
            hdr_image = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
            if hdr_image is None:
                raise ValueError(f"Failed to load HDR image: {file_path}")
            # 转换为RGB格式
            hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
            return hdr_image
        except Exception as e:
            print(f"Error loading HDR map: {e}")
            return None

    @staticmethod
    def calculate_hemisphere_light(normal: List[float], 
                                 sky_color: List[float], 
                                 ground_color: List[float]) -> List[float]:
        """
        计算半球光照
        Args:
            normal: 表面法线
            sky_color: 天空颜色 [r,g,b]
            ground_color: 地面颜色 [r,g,b]
        Returns:
            混合后的颜色 [r,g,b]
        """
        normal = LightUtils.normalize_vector(normal)
        up = [0, 1, 0]
        cos_theta = np.dot(normal, up)
        factor = (cos_theta + 1.0) * 0.5
        return [
            sky_color[i] * factor + ground_color[i] * (1.0 - factor)
            for i in range(3)
        ]

    @staticmethod
    def create_light_frustum(light_position: List[float], 
                           light_direction: List[float],
                           fov: float = 45.0,
                           aspect_ratio: float = 1.0,
                           near: float = 0.1,
                           far: float = 100.0) -> Dict[str, List[float]]:
        """
        创建光源视锥体
        Args:
            light_position: 光源位置
            light_direction: 光源方向
            fov: 视场角(度)
            aspect_ratio: 宽高比
            near: 近平面距离
            far: 远平面距离
        Returns:
            包含视锥体顶点的字典
        """
        # 计算视锥体的8个顶点
        tan_half_fov = np.tan(radians(fov * 0.5))
        
        # 近平面高度和宽度
        near_height = 2.0 * near * tan_half_fov
        near_width = near_height * aspect_ratio
        
        # 远平面高度和宽度
        far_height = 2.0 * far * tan_half_fov
        far_width = far_height * aspect_ratio
        
        # 构建坐标系
        forward = np.array(LightUtils.normalize_vector(light_direction))
        right = np.cross(forward, [0, 1, 0])
        right = LightUtils.normalize_vector(right.tolist())
        up = np.cross(np.array(right), forward)
        
        # 计算8个顶点
        vertices = []
        
        # 近平面顶点
        for i in [-1, 1]:
            for j in [-1, 1]:
                vertex = np.array(light_position) + \
                        forward * near + \
                        right * (i * near_width * 0.5) + \
                        up * (j * near_height * 0.5)
                vertices.append(vertex.tolist())
        
        # 远平面顶点
        for i in [-1, 1]:
            for j in [-1, 1]:
                vertex = np.array(light_position) + \
                        forward * far + \
                        right * (i * far_width * 0.5) + \
                        up * (j * far_height * 0.5)
                vertices.append(vertex.tolist())
        
        return {
            'vertices': vertices,
            'near': near,
            'far': far,
            'fov': fov,
            'aspect_ratio': aspect_ratio
        }

    @staticmethod
    def debug_light_visualization(light_type: LightType,
                                position: Optional[List[float]] = None,
                                direction: Optional[List[float]] = None,
                                color: Optional[List[float]] = None,
                                size: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        生成用于调试显示的光源可视化数据
        Args:
            light_type: 光源类型
            position: 光源位置 [x,y,z]
            direction: 光源方向 [x,y,z]
            color: 光源颜色 [r,g,b,a]
            size: 光源尺寸 [width,height]
        Returns:
            可视化数据字典
        """
        if color is None:
            color = [1.0, 1.0, 1.0, 1.0]
            
        if light_type == LightType.DIRECTIONAL:
            return {
                'type': 'arrow',
                'direction': direction,
                'color': color,
                'length': 1.0  # 箭头长度
            }
        elif light_type == LightType.POINT:
            return {
                'type': 'sphere',
                'position': position,
                'radius': 0.1,
                'color': color
            }
        elif light_type == LightType.SPOT:
            return {
                'type': 'cone',
                'position': position,
                'direction': direction,
                'color': color,
                'angle': 45.0  # 圆锥角度
            }
        elif light_type == LightType.AREA:
            return {
                'type': 'rectangle',
                'position': position,
                'direction': direction,
                'size': size,
                'color': color
            }
        return None

    @staticmethod
    def rotate_vector(vector: List[float], 
                     axis: List[float], 
                     angle: float) -> List[float]:
        """
        围绕任意轴旋转向量
        Args:
            vector: 要旋转的向量
            axis: 旋转轴
            angle: 旋转角度(度)
        Returns:
            旋转后的向量
        """
        # 将角度转换为弧度
        angle = radians(angle)
        
        # 标准化旋转轴
        axis = np.array(LightUtils.normalize_vector(axis))
        vector = np.array(vector)
        
        # 使用Rodrigues旋转公式
        cos_theta = cos(angle)
        sin_theta = sin(angle)
        
        rotated = vector * cos_theta + \
                  np.cross(axis, vector) * sin_theta + \
                  axis * np.dot(axis, vector) * (1 - cos_theta)
                  
        return rotated.tolist()

    @staticmethod
    def calculate_light_bounds(lights: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """
        计算场景中所有光源的包围盒
        Args:
            lights: 光源列表
        Returns:
            包含最小和最大边界的字典
        """
        if not lights:
            return {'min': [0, 0, 0], 'max': [0, 0, 0]}
            
        positions = []
        for light in lights:
            if 'position' in light:
                positions.append(light['position'])
            # 对于方向光，可以添加一个远处的点
            elif 'direction' in light:
                far_point = np.array(light['direction']) * 1000.0
                positions.append(far_point.tolist())
                
        if not positions:
            return {'min': [0, 0, 0], 'max': [0, 0, 0]}
            
        positions = np.array(positions)
        return {
            'min': positions.min(axis=0).tolist(),
            'max': positions.max(axis=0).tolist()
        }