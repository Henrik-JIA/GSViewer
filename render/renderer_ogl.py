from OpenGL import GL as gl
import util
import util_gau
import numpy as np
import ctypes

try:
    from OpenGL.raw.WGL.EXT.swap_control import wglSwapIntervalEXT
except:
    wglSwapIntervalEXT = None

from .primitives.axes import AxesHelper

_sort_buffer_xyz = None
_sort_buffer_gausid = None  # used to tell whether gaussian is reloaded

def _sort_gaussian_cpu(gaus, view_mat):
    xyz = np.asarray(gaus.xyz)
    view_mat = np.asarray(view_mat)

    # 使用矩阵乘法和广播计算深度
    xyz_view = np.dot(view_mat[:3, :3], xyz.T).T + view_mat[:3, 3]
    depth = xyz_view[:, 2]

    index = np.argsort(depth)
    index = index.astype(np.int32).reshape(-1, 1)
    return index

def _sort_gaussian_cupy(gaus, view_mat):
    import cupy as cp
    global _sort_buffer_gausid, _sort_buffer_xyz
    if _sort_buffer_gausid != id(gaus):
        _sort_buffer_xyz = cp.asarray(gaus.xyz)
        _sort_buffer_gausid = id(gaus)

    xyz = _sort_buffer_xyz
    view_mat = cp.asarray(view_mat)

    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]

    index = cp.argsort(depth)
    index = index.astype(cp.int32).reshape(-1, 1)

    index = cp.asnumpy(index) # convert to numpy
    return index

def _sort_gaussian_torch(gaus, view_mat):
    global _sort_buffer_gausid, _sort_buffer_xyz
    if _sort_buffer_gausid != id(gaus):
        _sort_buffer_xyz = torch.tensor(gaus.xyz).cuda()
        _sort_buffer_gausid = id(gaus)

    xyz = _sort_buffer_xyz
    view_mat = torch.tensor(view_mat).cuda()
    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]
    index = torch.argsort(depth)
    index = index.type(torch.int32).reshape(-1, 1).cpu().numpy()
    return index


# Decide which sort to use
_sort_gaussian = None
try:
    import torch
    if not torch.cuda.is_available():
        raise ImportError
    print("Detect torch cuda installed, will use torch as sorting backend")
    _sort_gaussian = _sort_gaussian_torch
except ImportError:
    try:
        import cupy as cp
        print("Detect cupy installed, will use cupy as sorting backend")
        _sort_gaussian = _sort_gaussian_cupy
    except ImportError:
        _sort_gaussian = _sort_gaussian_cpu


class GaussianRenderBase:
    def __init__(self):
        self.gaussians = None
        self._reduce_updates = True

    @property
    def reduce_updates(self):
        return self._reduce_updates

    @reduce_updates.setter
    def reduce_updates(self, val):
        self._reduce_updates = val
        self.update_vsync()

    def update_vsync(self):
        print("VSync is not supported")

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        raise NotImplementedError()
    
    def sort_and_update(self):
        raise NotImplementedError()

    def set_scale_modifier(self, modifier: float):
        raise NotImplementedError()
    
    def set_render_mod(self, mod: int):
        raise NotImplementedError()
    
    def update_camera_pose(self):
        raise NotImplementedError()

    def update_camera_intrin(self):
        raise NotImplementedError()
    
    def set_enable_cube(self, enable_cube: int):
        raise NotImplementedError()
    
    def set_cube_rotation(self, cube_rotation: list):
        raise NotImplementedError()

    def set_point_cubeMin(self, point_cubeMin: list):
        raise NotImplementedError()
    
    def set_point_cubeMax(self, point_cubeMax: list):
        raise NotImplementedError()
    
    def draw(self):
        raise NotImplementedError()
    
    def set_render_reso(self, w, h):
        raise NotImplementedError()


class OpenGLRenderer(GaussianRenderBase):
    def __init__(self, w, h, camera):
        super().__init__()
        self.camera = camera  # 添加相机属性
        self.view_matrix = camera.get_view_matrix()  # 添加视图矩阵成员
        self.proj_matrix = camera.get_project_matrix()  # 添加投影矩阵成员
        gl.glViewport(0, 0, w, h)

        # 初始化坐标轴辅助器
        self.axes_helper = AxesHelper(length=1.0)
        self.show_axes = True  # 添加标志以控制轴的显示

        self.program = util.load_shaders('shaders/gau_vert.glsl', 'shaders/gau_frag.glsl')
        self.program_boundary_box = util.load_shaders('shaders/boundary_box_vert.glsl','shaders/boundary_box_frag.glsl')
        self.program_axes = util.load_shaders('shaders/axes_vert.glsl', 'shaders/axes_frag.glsl')  # 加载轴的着色器

        # Vertex data for a quad
        self.quad_v = np.array([
            -1,  1,
            1,  1,
            1, -1,
            -1, -1
        ], dtype=np.float32).reshape(4, 2)
        self.quad_f = np.array([
            0, 1, 2,
            0, 2, 3
        ], dtype=np.uint32).reshape(2, 3)
        
        # load quad geometry
        vao, buffer_id = util.set_attributes(self.program, ["position"], [self.quad_v])
        util.set_faces_tovao(vao, self.quad_f)
        self.vao = vao
        self.ebo = util.set_faces_tovao(self.vao, self.quad_f)
        self.gau_bufferid = None
        self.index_bufferid = None

        # initial box 初始包围盒
        self.switch_show_boundary_box = False
        # 创建并绑定顶点数组对象
        self.vao_box = gl.glGenVertexArrays(1)
        self.vbo_box = None
        self.ebo_box_triangles = None
        self.ebo_box_lines = None
        self.box_vertices_count_triangles = 0
        self.box_vertices_count_lines = 0

        # opengl settings
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # 设置初始值color_scale_factors
        util.set_uniform_v3(self.program, [1.0, 1.0, 1.0], "color_scale_factors")
        # 设置初始值DC特征的调整系数dc_factor
        util.set_uniform_1f(self.program, 1.0, "dc_factor")
        # 设置初始值extra_factor
        util.set_uniform_1f(self.program, 1.0, "extra_factor")

        self.update_vsync()

    def __del__(self):
        # 清理资源
        gl.glDeleteBuffers(1, [self.vbo_box])
        gl.glDeleteBuffers(1, [self.ebo_box_triangles])
        gl.glDeleteBuffers(1, [self.ebo_box_lines])
        gl.glDeleteVertexArrays(1, [self.vao_box])

    def update_vsync(self):
        if wglSwapIntervalEXT is not None:
            wglSwapIntervalEXT(1 if self.reduce_updates else 0)
        else:
            print("VSync is not supported")

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        self.gaussians = gaus
        # load gaussian geometry
        gaussian_data = gaus.flat()
        self.gau_bufferid = util.set_storage_buffer_data(
            self.program, 
            "gaussian_data", 
            gaussian_data, 
            bind_idx=0,
            buffer_id=self.gau_bufferid)
        util.set_uniform_1int(self.program, gaus.sh_dim, "sh_dim")

    # 针对高斯元对象调整
    # 更新DC特征的调整系数并应用所有调整
    def adjust_dc_features(self, dc_factor):
        util.set_uniform_1f(self.program, dc_factor, "dc_factor")
    # 更新额外特征的调整系数并应用所有调整
    def adjust_extra_features(self, extra_factor):
        util.set_uniform_1f(self.program, extra_factor, "extra_factor")
    # 更新颜色调整系数并应用所有调整
    def update_color_factor(self, g_rgb_factor):
        util.set_uniform_v3(self.program, g_rgb_factor, "color_scale_factors")
    # 设置旋转修改因子
    def set_rot_modifier(self, modifier):
        quat = util.euler_to_quaternion(modifier[0], modifier[1], modifier[2])
        util.set_uniform_4f(self.program, "rot_modifier", quat.x, quat.y, quat.z, quat.w)
    # 设置光照旋转因子
    def set_light_rotation(self, lightRotation):
        # rotation参数是一个包含三个元素的列表或元组，分别代表绕X、Y、Z轴的旋转角度
        util.set_uniform_v3(self.program, lightRotation, "light_rotation")

    def sort_and_update(self):
        index = _sort_gaussian(self.gaussians, self.view_matrix)
        self.index_bufferid = util.set_storage_buffer_data(
            self.program, 
            "gi", 
            index, 
            bind_idx=1,
            buffer_id=self.index_bufferid
        )
        return
   
    def set_scale_modifier(self, modifier):
        util.set_uniform_1f(self.program, modifier, "gaussian_scale_factor")
    
    def set_screen_scale_factor(self, factor):
        util.set_uniform_1f(self.program, factor, "screen_display_scale_factor")

    def set_render_mod(self, mod: int):
        util.set_uniform_1int(self.program, mod, "render_mod")

    def set_render_reso(self, w, h):
        gl.glViewport(0, 0, w, h)

    def update_camera_pose(self):
        self.view_matrix = self.camera.get_view_matrix()
        util.set_uniform_mat4(self.program, self.view_matrix, "view_matrix")
        util.set_uniform_v3(self.program, self.camera.position, "cam_pos")
        self.axes_helper.needs_update = True  # 每次更新相机姿态时，设置轴需要更新

    def update_camera_intrin(self):
        self.proj_matrix = self.camera.get_project_matrix()
        util.set_uniform_mat4(self.program, self.proj_matrix, "projection_matrix")
        util.set_uniform_v3(self.program, self.camera.get_htanfovxy_focal(), "hfovxy_focal")
        self.axes_helper.needs_update = True # 直接使用 AxesHelper 的更新标志

    # 包围盒
    # Set the center point coordinates
    def set_points_center(self, points_center: list):
        util.set_uniform_v3(self.program, points_center, "points_center")

    # Set whether to use a cube to limit the rendering area aabb
    def set_enable_aabb(self, enable_aabb: int):
        util.set_uniform_1int(self.program, enable_aabb, "enable_aabb")

    # Set whether to use a cube to limit the rendering area obb
    def set_enable_obb(self, enable_obb: int):
        util.set_uniform_1int(self.program, enable_obb, "enable_obb")

    # Set the rotation of the cube
    def set_cube_rotation(self, cube_rotation: list):
        R = util.convert_euler_angles_to_rotation_matrix(cube_rotation)
        util.set_uniform_mat3(self.program, R, "cube_rotation") 

    # Set the minimum coordinates of the cube
    def set_point_cubeMin(self, point_cubeMin: list):
        util.set_uniform_v3(self.program, point_cubeMin, "cubeMin")

    # Set the maximum coordinates of the cube
    def set_point_cubeMax(self, point_cubeMax: list):
        util.set_uniform_v3(self.program, point_cubeMax, "cubeMax")

    def draw_boundary_box(self, points_center: list, point_cubeMin, point_cubeMax, cube_rotation):
        R = util.convert_euler_angles_to_rotation_matrix(cube_rotation)

        # 设置uniforms
        util.set_uniform_mat4(self.program_boundary_box, self.view_matrix, "view_matrix")
        util.set_uniform_mat4(self.program_boundary_box, self.proj_matrix, "projection_matrix")
        util.set_uniform_mat3(self.program_boundary_box, R, "cube_rotation") 
        util.set_uniform_v3(self.program_boundary_box, points_center, "points_center")
        util.set_uniform_v3(self.program_boundary_box, point_cubeMin, "cubeMin")
        util.set_uniform_v3(self.program_boundary_box, point_cubeMax, "cubeMax")

        vertices, indices_triangles, indices_lines = util.create_box_mesh_from_bounds(points_center, point_cubeMin, point_cubeMax)
        rotated_vertices = np.dot(vertices, R.T) 
        vertices = rotated_vertices.flatten()

        # 记录索引数量，用于绘制
        self.box_vertices_count_triangles = len(indices_triangles)
        self.box_vertices_count_lines = len(indices_lines)

        # 检查并释放旧的 VBO 和 EBO
        if self.vbo_box:
            gl.glDeleteBuffers(1, [self.vbo_box])
        if self.ebo_box_triangles:
            gl.glDeleteBuffers(1, [self.ebo_box_triangles])
        if self.ebo_box_lines:
            gl.glDeleteBuffers(1, [self.ebo_box_lines])

        # # 使用 set_attribute 函数设置顶点属性
        # vao, buffer_id = util.set_attribute(
        #             self.program_boundary_box, 
        #             'vertexPosition', 
        #             vertices, 
        #             3,
        #             self.vao_box
        #         )
        # self.vbo_box = buffer_id

        # 使用 set_faces_tovao 函数设置索引缓冲对象，并接收返回的 ebo
        # 设置三角形索引缓冲区
        self.ebo_box_triangles = util.set_faces_tovao(self.vao_box, indices_triangles.flatten())
        # 设置线框索引缓冲区
        self.ebo_box_lines = util.set_faces_tovao(self.vao_box, indices_lines.flatten())

    def clear_boundary_box(self):
        # 清除包围盒的绘制资源
        if self.vbo_box:
            gl.glDeleteBuffers(1, [self.vbo_box])
            self.vbo_box = None
        if self.ebo_box_triangles:
            gl.glDeleteBuffers(1, [self.ebo_box_triangles])
            self.ebo_box_triangles = None
        if self.ebo_box_lines:
            gl.glDeleteBuffers(1, [self.ebo_box_lines])
            self.ebo_box_lines = None
        if self.vao_box:
            gl.glDeleteVertexArrays(1, [self.vao_box])
            self.vao_box = gl.glGenVertexArrays(1)  # 重新生成 VAO
        self.box_vertices_count_triangles = 0
        self.box_vertices_count_lines = 0

    def toggle_draw_boundary_box(self):
        # 切换包围盒显示状态
        self.switch_show_boundary_box = not self.switch_show_boundary_box

    def draw(self):
        # 主渲染高斯
        gl.glUseProgram(self.program)
        gl.glBindVertexArray(self.vao)
        num_gau = len(self.gaussians)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER,self.ebo)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, len(self.quad_f.reshape(-1)), gl.GL_UNSIGNED_INT, None, num_gau)

        # 绘制包围盒
        if self.switch_show_boundary_box:
            if self.vao_box and self.ebo_box_triangles and self.ebo_box_lines:
                # 绑定着色器程序
                gl.glUseProgram(self.program_boundary_box)
                gl.glBindVertexArray(self.vao_box)
                # 使用填充模式绘制面
                gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo_box_triangles)
                gl.glDrawElements(gl.GL_TRIANGLES, self.box_vertices_count_triangles, gl.GL_UNSIGNED_INT, None)
                # 设置线宽
                gl.glLineWidth(3.5)  # 设置线宽
                # 使用线框模式绘制线框
                gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo_box_lines)
                gl.glDrawElements(gl.GL_LINES, self.box_vertices_count_lines, gl.GL_UNSIGNED_INT, None)
                # 解绑 VAO 和 VBO
                gl.glBindVertexArray(0)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
                gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

        # 绘制XYZ轴
        if self.show_axes:
            length=1.0
            line_width=3.5
            self.axes_helper.draw(
                view_matrix=self.view_matrix,
                projection_matrix=self.proj_matrix,
                scale=length,
                line_width=line_width
            )
