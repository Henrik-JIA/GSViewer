from OpenGL import GL as gl
import numpy as np
import ctypes
from typing import Optional
import util
from .base import PrimitiveBase

class AxesHelper(PrimitiveBase):
    """坐标轴辅助器，用于绘制XYZ轴"""
    
    def __init__(self, length: float = 1.0):
        """
        初始化坐标轴辅助工具
        
        Args:
            length: 轴的长度，默认为1.0单位长度
        """
        super().__init__()
        self.length = length
        # 加载着色器
        self.program = util.load_shaders('shaders/axes_vert.glsl', 'shaders/axes_frag.glsl')
        self.init_buffers()
        
    def set_length(self, length: float):
        """设置轴的长度"""
        self.length = length
        
    def init_buffers(self):
        """初始化顶点缓冲区"""
        # 轴的顶点数据 [位置, 颜色]
        self.vertices = np.array([
            # X轴 - 红色
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  # 起点
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0,  # 终点
            # Y轴 - 绿色
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,  # 起点
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0,  # 终点
            # Z轴 - 蓝色
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,  # 起点
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,  # 终点
        ], dtype=np.float32)

        # 创建并绑定VAO和VBO
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, 
                       self.vertices, gl.GL_STATIC_DRAW)

        # 设置顶点属性
        # 位置属性
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 
                                24, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        # 颜色属性
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 
                                24, ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(1)

        # 解绑
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    def update_matrices(self, view_matrix: np.ndarray, projection_matrix: np.ndarray):
        """
        更新视图和投影矩阵
        
        Args:
            view_matrix: 4x4视图矩阵
            projection_matrix: 4x4投影矩阵
        """
        gl.glUseProgram(self.program)
        util.set_uniform_mat4(self.program, view_matrix, "view_matrix")
        util.set_uniform_mat4(self.program, projection_matrix, "projection_matrix")
        self.needs_update = False

    def draw(self, view_matrix: np.ndarray, projection_matrix: np.ndarray, 
            position: Optional[np.ndarray] = None, 
            rotation: Optional[np.ndarray] = None,
            scale: Optional[float] = None,
            line_width: float = 3.5):
        """
        绘制坐标轴
        
        Args:
            view_matrix: 4x4视图矩阵
            projection_matrix: 4x4投影矩阵
            position: 可选，3D位置偏移
            rotation: 可选，3x3旋转矩阵
            scale: 可选，缩放因子
            line_width: 线宽，默认3.5
        """
        if self.needs_update:
            self.update_matrices(view_matrix, projection_matrix)

        gl.glUseProgram(self.program)
        
        # 构建模型矩阵
        model_mat = np.eye(4, dtype=np.float32)
        
        # 应用缩放
        scale_factor = scale if scale is not None else self.length
        model_mat[0:3, 0:3] *= scale_factor
        
        # 应用旋转
        if rotation is not None:
            model_mat[0:3, 0:3] = rotation * scale_factor
            
        # 应用位移
        if position is not None:
            model_mat[0:3, 3] = position
            
        util.set_uniform_mat4(self.program, model_mat, "model_matrix")

        # 设置线宽并绘制
        gl.glLineWidth(line_width)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_LINES, 0, 6)  # 绘制3个轴（每个轴2个顶点）
        gl.glBindVertexArray(0)
        gl.glLineWidth(1.0)  # 恢复默认线宽