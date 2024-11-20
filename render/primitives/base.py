from OpenGL import GL as gl
import numpy as np

class PrimitiveBase:
    """所有基础图元的基类"""
    def __init__(self):
        self.vao = None
        self.vbo = None
        self.program = None
        self.needs_update = True

    def init_buffers(self):
        """初始化缓冲区"""
        raise NotImplementedError()

    def update_matrices(self, view_matrix: np.ndarray, projection_matrix: np.ndarray):
        """更新矩阵"""
        raise NotImplementedError()

    def draw(self, view_matrix: np.ndarray, projection_matrix: np.ndarray, **kwargs):
        """绘制图元"""
        raise NotImplementedError()

    def __del__(self):
        """清理OpenGL资源"""
        if self.vbo is not None:
            gl.glDeleteBuffers(1, [self.vbo])
        if self.vao is not None:
            gl.glDeleteVertexArrays(1, [self.vao])