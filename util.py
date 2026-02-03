from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import numpy as np
import scipy as sp
import glm
import ctypes

class Camera:
    def __init__(self, h, w):
        self.znear = 0.001
        self.zfar = 500
        self.h = h
        self.w = w
        self.fovy = np.pi / 2
        self.position = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        self.yaw = -np.pi / 2
        self.pitch = 0
        self.roll = 0

        self.is_pose_dirty = True
        self.is_intrin_dirty = True

        self.last_x = 640
        self.last_y = 360
        self.first_mouse = True

        self.is_leftmouse_pressed = False
        self.is_rightmouse_pressed = False

        self.sensitivities = {
            'rot': 0.02,
            'trans': 0.005,
            'zoom': 0.08,
            'roll': 0.03
        }
        self.target_dist = 5.0

        self.rotation = glm.quat(1, 0, 0, 0)  # 初始化为单位四元数
        self.use_free_rotation = True  # 添加自由旋转属性
        self.rotation_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 添加旋转中心属性
        self.use_custom_rotation_center = False  # 添加是否使用自定义旋转中心的标志

        self.use_orthographic = False  # 添加正射视角属性
        self.ortho_scale = 5.0  # 添加正射视角缩放因子
        
        self.invert_mouse = True  # 是否反转鼠标方向（True=拖拽方向与模型移动方向一致）

    def _global_rot_mat(self):
        x = np.array([1, 0, 0])
        z = np.cross(x, self.up)
        z = z / np.linalg.norm(z)
        x = np.cross(self.up, z)
        return np.stack([x, self.up, z], axis=-1)
    
    def set_use_free_rotation(self, use_free_rotation):
        self.use_free_rotation = use_free_rotation
        if use_free_rotation:
            self.rotation = glm.quat(glm.vec3(self.pitch, self.yaw, self.roll))  # 将当前欧拉角转换为四元数
        self.is_pose_dirty = True

    def get_view_matrix(self):
        if self.use_free_rotation:
            rotation_matrix = glm.mat4_cast(self.rotation)
            direction = glm.vec3(rotation_matrix * glm.vec4(0, 0, -1, 0))
            up_direction = glm.vec3(rotation_matrix * glm.vec4(0, 1, 0, 0))
            if self.use_custom_rotation_center:
                self.position = self.rotation_center - direction * self.target_dist
            else:
                self.position = self.target - direction * self.target_dist
            view_matrix = np.array(glm.lookAt(self.position, self.target, up_direction))
        else:
            position = glm.vec3(self.position)
            target = glm.vec3(self.target)
            up = glm.vec3(self.up)
            view_matrix = np.array(glm.lookAt(position, target, up))
        return view_matrix

    def get_project_matrix(self):
        if self.use_orthographic:
            aspect_ratio = self.w / self.h
            project_mat = glm.ortho(
                -self.ortho_scale * aspect_ratio, self.ortho_scale * aspect_ratio,
                -self.ortho_scale, self.ortho_scale,
                self.znear, self.zfar
            )
        else:
            project_mat = glm.perspective(
                self.fovy,
                self.w / self.h,
                self.znear,
                self.zfar
            )
        return np.array(project_mat, dtype=np.float32)

    def get_rotation_matrix(self):
        rotation_matrix = glm.mat4_cast(self.rotation)
        return np.array(rotation_matrix)[:3, :3]

    def process_mouse(self, xpos, ypos):
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False

        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos
        self.last_x = xpos
        self.last_y = ypos
        
        # 方向因子：invert_mouse=True 时反转方向
        dir_factor = -1 if self.invert_mouse else 1

        if self.is_leftmouse_pressed:
            if self.use_free_rotation:
                yaw_quat = glm.angleAxis(glm.radians(dir_factor * xoffset * self.sensitivities['rot'] * 20), glm.vec3(0, 1, 0))
                pitch_quat = glm.angleAxis(glm.radians(-dir_factor * yoffset * self.sensitivities['rot'] * 20), glm.vec3(1, 0, 0))
                roll_quat = glm.angleAxis(glm.radians(self.roll), glm.vec3(0, 0, 1))  # 处理 roll 角度
                self.rotation = glm.normalize(yaw_quat * self.rotation * pitch_quat * roll_quat)
                if self.use_custom_rotation_center:
                    self.position = self.rotation_center - glm.vec3(glm.mat4_cast(self.rotation) * glm.vec4(0, 0, self.target_dist, 0))
            else:
                self.yaw += dir_factor * xoffset * self.sensitivities['rot']
                self.pitch -= dir_factor * yoffset * self.sensitivities['rot']
                self.pitch = np.clip(self.pitch, -np.pi / 2, np.pi / 2)
                front = np.array([np.cos(self.yaw) * np.cos(self.pitch), 
                                np.sin(self.pitch), np.sin(self.yaw) * 
                                np.cos(self.pitch)])
                front = self._global_rot_mat() @ front.reshape(3, 1)
                front = front[:, 0]
                if self.use_custom_rotation_center:
                    self.position = self.rotation_center - front * np.linalg.norm(self.position - self.rotation_center)
                else:
                    self.position[:] = - front * np.linalg.norm(self.position - self.target) + self.target
            self.is_pose_dirty = True

        if self.is_rightmouse_pressed:
            if self.use_free_rotation:
                # 使用旋转矩阵计算方向向量
                rotation_matrix = glm.mat4_cast(self.rotation)
                front = glm.vec3(rotation_matrix * glm.vec4(0, 0, -1, 0))
                right = glm.vec3(rotation_matrix * glm.vec4(1, 0, 0, 0))
                cam_up = glm.vec3(rotation_matrix * glm.vec4(0, 1, 0, 0))

                self.position += dir_factor * right * xoffset * self.sensitivities['trans']
                self.target += dir_factor * right * xoffset * self.sensitivities['trans']
                self.position += dir_factor * cam_up * yoffset * self.sensitivities['trans']
                self.target += dir_factor * cam_up * yoffset * self.sensitivities['trans']

                self.is_pose_dirty = True
            else:
                front = self.target - self.position
                front = front / np.linalg.norm(front)
                right = np.cross(self.up, front)
                if self.use_custom_rotation_center:
                    self.position -= dir_factor * right * xoffset * self.sensitivities['trans']
                    self.rotation_center -= dir_factor * right * xoffset * self.sensitivities['trans']
                    cam_up = np.cross(right, front)
                    self.position -= dir_factor * cam_up * yoffset * self.sensitivities['trans']
                    self.rotation_center -= dir_factor * cam_up * yoffset * self.sensitivities['trans']
                else:
                    self.position -= dir_factor * right * xoffset * self.sensitivities['trans']
                    self.target -= dir_factor * right * xoffset * self.sensitivities['trans']
                    cam_up = np.cross(right, front)
                    self.position -= dir_factor * cam_up * yoffset * self.sensitivities['trans']
                    self.target -= dir_factor * cam_up * yoffset * self.sensitivities['trans']
                self.is_pose_dirty = True

    def process_wheel(self, dx, dy):
        if self.use_orthographic:
            self.ortho_scale -= dy * self.sensitivities['zoom']
            self.ortho_scale = max(1.0, self.ortho_scale)  # 确保缩放因子不小于1.0
            self.is_intrin_dirty = True
        else:
            front = self.target - self.position
            front = front / np.linalg.norm(front)
            if self.use_custom_rotation_center:
                self.position += front * dy * self.sensitivities['zoom']
                self.rotation_center += front * dy * self.sensitivities['zoom']
            else:
                self.position += front * dy * self.sensitivities['zoom']
                self.target += front * dy * self.sensitivities['zoom']
            self.is_pose_dirty = True

    def get_htanfovxy_focal(self):
        htany = np.tan(self.fovy / 2)
        htanx = htany / self.h * self.w
        focal = self.h / (2 * htany)
        return [htanx, htany, focal]

    def get_focal(self):
        return self.h / (2 * np.tan(self.fovy / 2))

    def process_roll_key(self, d):
        if self.use_free_rotation:
            roll_quat = glm.angleAxis(glm.radians(d * self.sensitivities['roll'] * 20), glm.vec3(0, 0, 1))
            self.rotation = glm.normalize(self.rotation * roll_quat)
        else:
            front = self.target - self.position
            right = np.cross(front, self.up)
            new_up = self.up + right * (d * self.sensitivities['roll'] / np.linalg.norm(right))
            self.up = new_up / np.linalg.norm(new_up)
        self.is_pose_dirty = True

    def flip_ground(self):
        if self.use_free_rotation:
            flip_quat = glm.angleAxis(glm.radians(180), glm.vec3(0, 0, 1))
            self.rotation = glm.normalize(flip_quat * self.rotation)
        else:
            self.up = -self.up
        self.is_pose_dirty = True

    def update_target_distance(self):
        _dir = self.target - self.position
        _dir = _dir / np.linalg.norm(_dir)
        self.target = self.position + _dir * self.target_dist
        self.is_pose_dirty = True

    def update_resolution(self, height, width):
        self.h = max(height, 1)
        self.w = max(width, 1)
        self.is_intrin_dirty = True

def load_shaders(vs, fs):
    with open(vs, 'r', encoding='utf-8') as v_shader_file:
        vertex_shader = v_shader_file.read()
    with open(fs, 'r', encoding='utf-8') as f_shader_file:
        fragment_shader = f_shader_file.read()

    active_shader = shaders.compileProgram(
        shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
    )
    return active_shader


def compile_shaders(vertex_shader, fragment_shader):
    active_shader = shaders.compileProgram(
        shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
    )
    return active_shader

# program：指的是已经编译和链接的着色器程序的ID。这个程序包含了OpenGL应用的顶点着色器和片段着色器。
# keys：一个字符串列表，每个字符串代表一个顶点属性的名称。这些名称应与顶点着色器中定义的属性相匹配。
# values：一个包含实际顶点数据的NumPy数组列表。每个数组对应于keys列表中的一个属性。
# vao：顶点数组对象(Vertex Array Object)的ID。如果为None，函数将创建一个新的VAO。
# buffer_ids：一个包含顶点缓冲对象(Vertex Buffer Object, VBO)的ID列表。如果为None，函数将为每个属性创建一个新的VBO。
def set_attributes(program, keys, values, lengths=None, vao=None, buffer_ids=None):
    glUseProgram(program)
    if vao is None:
        vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    if buffer_ids is None:
        buffer_ids = [None] * len(keys)
    if lengths is None:
        lengths = [None] * len(keys)  # 确保lengths列表存在，即使是全None

    for i, (key, value, b) in enumerate(zip(keys, values, buffer_ids)):
        if b is None:
            b = glGenBuffers(1)
            buffer_ids[i] = b
        glBindBuffer(GL_ARRAY_BUFFER, b)
        glBufferData(GL_ARRAY_BUFFER, value.nbytes, value.reshape(-1), GL_STATIC_DRAW)
        length = lengths[i] if lengths[i] is not None else value.shape[-1]  # 使用lengths列表中的值，如果为None，则使用value.shape[-1]
        pos = glGetAttribLocation(program, key)
        glVertexAttribPointer(pos, length, GL_FLOAT, False, 0, None)
        glEnableVertexAttribArray(pos)
    
    glBindBuffer(GL_ARRAY_BUFFER,0) #解绑GL_ARRAY_BUFFER
    glBindVertexArray(0)
    return vao, buffer_ids

def set_attribute(program, key, value, lengths=None, vao=None, buffer_id=None):
    glUseProgram(program)
    if vao is None:
        vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    if buffer_id is None:
        buffer_id = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id)
    glBufferData(GL_ARRAY_BUFFER, value.nbytes, value.reshape(-1), GL_STATIC_DRAW)
    length = lengths if lengths is not None else value.shape[-1]
    pos = glGetAttribLocation(program, key)
    glVertexAttribPointer(pos, length, GL_FLOAT, False, 0, None)
    glEnableVertexAttribArray(pos)
    glBindBuffer(GL_ARRAY_BUFFER,0)
    return vao, buffer_id

def set_attribute_instanced(program, key, value, instance_stride=1, vao=None, buffer_id=None):
    glUseProgram(program)
    if vao is None:
        vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    if buffer_id is None:
        buffer_id = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id)
    glBufferData(GL_ARRAY_BUFFER, value.nbytes, value.reshape(-1), GL_STATIC_DRAW)
    length = value.shape[-1]
    pos = glGetAttribLocation(program, key)
    glVertexAttribPointer(pos, length, GL_FLOAT, False, 0, None)
    glEnableVertexAttribArray(pos)
    glVertexAttribDivisor(pos, instance_stride)
    glBindBuffer(GL_ARRAY_BUFFER,0)
    return vao, buffer_id

def set_storage_buffer_data(program, key, value: np.ndarray, bind_idx, vao=None, buffer_id=None):
    glUseProgram(program)
    # if vao is None:  # TODO: if this is really unnecessary?
    #     vao = glGenVertexArrays(1)
    if vao is not None:
        glBindVertexArray(vao)
    
    if buffer_id is None:
        buffer_id = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id)
    glBufferData(GL_SHADER_STORAGE_BUFFER, value.nbytes, value.reshape(-1), GL_STATIC_DRAW)
    # pos = glGetProgramResourceIndex(program, GL_SHADER_STORAGE_BLOCK, key)  # TODO: ???
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind_idx, buffer_id)
    # glShaderStorageBlockBinding(program, pos, pos)  # TODO: ???
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    return buffer_id

def set_faces_tovao(vao, faces: np.ndarray):
    # faces
    glBindVertexArray(vao)
    element_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)
    return element_buffer

def set_gl_bindings(vertices, faces):
    # vertices
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    # vertex_buffer = glGenVertexArrays(1)
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 4, GL_FLOAT, False, 0, None)
    glEnableVertexAttribArray(0)

    # faces
    element_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)
    # glVertexAttribPointer(1, 3, GL_FLOAT, False, 36, ctypes.c_void_p(12))
    # glEnableVertexAttribArray(1)
    # glVertexAttribPointer(2, 3, GL_FLOAT, False, 36, ctypes.c_void_p(12))
    # glEnableVertexAttribArray(2)

def set_uniform_mat3(shader, content, name):
    glUseProgram(shader)
    if isinstance(content, glm.mat3):
        content = np.array(content).astype(np.float32)
    else:
        content = content.T
    glUniformMatrix3fv(
        glGetUniformLocation(shader, name),
        1,
        GL_FALSE,
        content.astype(np.float32)
    )

def set_uniform_mat4(shader, content, name):
    glUseProgram(shader)
    if isinstance(content, glm.mat4):
        content = np.array(content).astype(np.float32)
    else:
        content = content.T
    glUniformMatrix4fv(
        glGetUniformLocation(shader, name), 
        1,
        GL_FALSE,
        content.astype(np.float32)
    )

def set_uniform_1f(shader, content, name):
    glUseProgram(shader)
    glUniform1f(
        glGetUniformLocation(shader, name), 
        content,
    )

def set_uniform_1int(shader, content, name):
    glUseProgram(shader)
    glUniform1i(
        glGetUniformLocation(shader, name), 
        content
    )

def set_uniform_v3f(shader, contents, name):
    glUseProgram(shader)
    glUniform3fv(
        glGetUniformLocation(shader, name),
        len(contents),
        contents
    )

def set_uniform_v3(shader, contents, name):
    glUseProgram(shader)
    glUniform3f(
        glGetUniformLocation(shader, name),
        contents[0], contents[1], contents[2]
    )

def set_uniform_4f(shader, name, x, y, z, w):
    glUseProgram(shader)
    location = glGetUniformLocation(shader, name)
    glUniform4f(location, x, y, z, w)

def set_uniform_v1f(shader, contents, name):
    glUseProgram(shader)
    glUniform1fv(
        glGetUniformLocation(shader, name),
        len(contents),
        contents
    )
    
def set_uniform_v2(shader, contents, name):
    glUseProgram(shader)
    glUniform2f(
        glGetUniformLocation(shader, name),
        contents[0], contents[1]
    )

def set_texture2d(img, texid=None):
    h, w, c = img.shape
    assert img.dtype == np.uint8
    if texid is None:
        texid = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texid)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,   
        GL_RGB, GL_UNSIGNED_BYTE, img
    )
    glActiveTexture(GL_TEXTURE0)  # can be removed
    # glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    return texid

def update_texture2d(img, texid, offset):
    x1, y1 = offset
    h, w = img.shape[:2]
    glBindTexture(GL_TEXTURE_2D, texid)
    glTexSubImage2D(
        GL_TEXTURE_2D, 0, x1, y1, w, h,
        GL_RGB, GL_UNSIGNED_BYTE, img
    )

def convert_euler_angles_to_rotation_matrix(angles):
    # Convert angles from degrees to radians
    angles = np.radians(angles)
    # Compute sine and cosine for each angle
    sx, sy, sz = np.sin(angles)
    cx, cy, cz = np.cos(angles)
    # Rotation matrix around the X-axis
    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])
    # Rotation matrix around the Y-axis
    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])
    # Rotation matrix around the Z-axis
    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ])
    # Combined rotation matrix
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def euler_to_quaternion(roll, pitch, yaw):
    # 将角度从度转换为弧度
    roll = glm.radians(roll)
    pitch = glm.radians(pitch)
    yaw = glm.radians(yaw)

    qx = glm.sin(roll/2) * glm.cos(pitch/2) * glm.cos(yaw/2) - glm.cos(roll/2) * glm.sin(pitch/2) * glm.sin(yaw/2)
    qy = glm.cos(roll/2) * glm.sin(pitch/2) * glm.cos(yaw/2) + glm.sin(roll/2) * glm.cos(pitch/2) * glm.sin(yaw/2)
    qz = glm.cos(roll/2) * glm.cos(pitch/2) * glm.sin(yaw/2) - glm.sin(roll/2) * glm.sin(pitch/2) * glm.cos(yaw/2)
    qw = glm.cos(roll/2) * glm.cos(pitch/2) * glm.cos(yaw/2) + glm.sin(roll/2) * glm.sin(pitch/2) * glm.sin(yaw/2)

    return glm.quat(qw, qx, qy, qz)

def convert_rotation_matrix_to_euler_angles(R):
    """
    从旋转矩阵转换到欧拉角（以度为单位）
    """
    rotation = sp.spatial.transform.Rotation.from_matrix(R)
    # 转换到欧拉角，这里使用'xyz'作为旋转顺序
    euler_angles = rotation.as_euler('xyz', degrees=True)
    return euler_angles

def create_box_line_from_bounds(points_center, cube_min, cube_max):
    # 计算顶点
    vertices = np.array([
        [cube_min[0], cube_max[1], cube_min[2]],  # 0 near_top_left
        [cube_max[0], cube_max[1], cube_min[2]],  # 1 near_top_right
        [cube_max[0], cube_min[1], cube_min[2]],  # 2 near_bottom_right
        [cube_min[0], cube_min[1], cube_min[2]],  # 3 near_bottom_left
        [cube_min[0], cube_max[1], cube_max[2]],  # 4 far_top_left
        [cube_max[0], cube_max[1], cube_max[2]],  # 5 far_top_right
        [cube_max[0], cube_min[1], cube_max[2]],  # 6 far_bottom_right
        [cube_min[0], cube_min[1], cube_max[2]],  # 7 far_bottom_left
    ], dtype=np.float32)
    
    # 将顶点移动到以 points_center 为中心
    center_offset = (np.array(cube_min) + np.array(cube_max)) / 2
    vertices = vertices - center_offset + points_center

    # 计算索引
    indices = np.array([
        0, 1, 1, 2, 2, 3, 3, 0,  # Near plane edges
        4, 5, 5, 6, 6, 7, 7, 4,  # Far plane edges
        0, 4, 1, 5, 2, 6, 3, 7,  # Sides connecting near and far planes
    ], dtype=np.uint32).reshape(-1, 2)
    
    return vertices, indices

def create_box_mesh_from_bounds(points_center, cube_min, cube_max):
    # 计算顶点
    vertices = np.array([
        [cube_min[0], cube_max[1], cube_min[2]],  # 0 near_top_left
        [cube_max[0], cube_max[1], cube_min[2]],  # 1 near_top_right
        [cube_max[0], cube_min[1], cube_min[2]],  # 2 near_bottom_right
        [cube_min[0], cube_min[1], cube_min[2]],  # 3 near_bottom_left
        [cube_min[0], cube_max[1], cube_max[2]],  # 4 far_top_left
        [cube_max[0], cube_max[1], cube_max[2]],  # 5 far_top_right
        [cube_max[0], cube_min[1], cube_max[2]],  # 6 far_bottom_right
        [cube_min[0], cube_min[1], cube_max[2]],  # 7 far_bottom_left
    ], dtype=np.float32)
    
    # 将顶点移动到以 points_center 为中心
    center_offset = (np.array(cube_min) + np.array(cube_max)) / 2
    vertices = vertices - center_offset + points_center

    # 计算三角形索引
    indices_triangles = np.array([
        # 顶部面
        0, 1, 4,
        1, 5, 4,
        # 底部面
        3, 2, 7,
        2, 6, 7,
        # 前面
        0, 3, 1,
        1, 3, 2,
        # 后面
        4, 5, 7,
        5, 6, 7,
        # 左面
        0, 4, 3,
        3, 4, 7,
        # 右面
        1, 2, 5,
        2, 6, 5
    ], dtype=np.uint32)

    # 计算线框索引
    indices_lines = np.array([
        0, 1, 1, 2, 2, 3, 3, 0,  # Near plane edges
        4, 5, 5, 6, 6, 7, 7, 4,  # Far plane edges
        0, 4, 1, 5, 2, 6, 3, 7,  # Sides connecting near and far planes
    ], dtype=np.uint32)

    return vertices, indices_triangles, indices_lines