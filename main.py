import glfw
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
import imgui
import numpy as np
import util
import util_gau
import tkinter as tk
import os
import sys
import argparse
import ctypes
from render.renderer_ogl import OpenGLRenderer, GaussianRenderBase

from gui.scene_environment_control import scene_environment_control_ui
from gui.camera_control import camera_control_ui
from gui.gs_elements_control import gs_elements_control_ui
from gui.render_boundary_control import render_boundary_control_ui 
from gui.help_content import help_window_ui

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 将包含main.py的目录添加到Python路径中
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# 将当前工作目录更改为脚本的目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

g_camera = util.Camera(720, 1280)
BACKEND_OGL=0
BACKEND_CUDA=1
g_renderer_list = [
    None, # ogl
]
g_renderer_idx = BACKEND_OGL
g_renderer: GaussianRenderBase = g_renderer_list[g_renderer_idx]
g_scale_modifier = 1.0
g_screen_scale_factor = 1.0
g_auto_sort = False
g_show_gs_elements_control = True
g_show_help_control = False
g_show_camera_control = False
g_render_mode_tables = ["Gaussian Ball", "Flat Ball", "Billboard", "Depth", "Normal", "Billboard Normal", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"]
g_render_mode = 9

g_background_color = [1.0, 1.0, 1.0, 1.0]
# g_background_color = [0.0, 0.0, 0.0, 1.0]  # 初始化背景颜色为黑色，不透明
g_show_scene_control = False  # 控制是否显示场景环境控制UI
g_directional_light_direction = [0.0, -1.0, 0.0]  # 初始定向光源方向：从上方向下
g_directional_light_color = [1.0, 1.0, 1.0, 1.0]  # 初始定向光源颜色：白色
g_directional_light_intensity = 1.0  # 初始定向光源强度

# 初始化渲染包围盒边界相关变量
g_show_render_boundary_control = False
use_axis_for_rotation = False 
g_enable_render_boundary_aabb = 0
g_enable_render_boundary_obb = 0
export_path = ""
export_status = "Please select export file path"
g_cube_rotation = [0.0, 0.0, 0.0]
g_cube_min = [-4.0, -4.0, -4.0]
g_cube_max = [4.0, 4.0, 4.0]
tmp_cube_rotation = [0.0, 0.0, 0.0]
tmp_cube_min = [-4.0, -4.0, -4.0]
tmp_cube_max = [4.0, 4.0, 4.0]

dc_scale_factor = 1.0  # 直流特征的缩放因子
extra_scale_factor = 1.0  # 额外特征的缩放因子
g_rgb_factor = [1.0, 1.0, 1.0]# 在rgb全局变量区域添加
g_rot_modifier = [0.0, 0.0, 0.0] #设置一个旋转修正因子，默认是1。
g_light_rotation = [0.0, 0.0, 0.0]  # 光照旋转角度，初始化为0

show_axes = True  # 初始化show_axes变量

def impl_glfw_init():
    window_name = "GSViewer"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    global window
    window = glfw.create_window(
        g_camera.w, g_camera.h, window_name, None, None
    )
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    # glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL);
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window

def cursor_pos_callback(window, xpos, ypos):
    if imgui.get_io().want_capture_mouse:
        g_camera.is_leftmouse_pressed = False
        g_camera.is_rightmouse_pressed = False
    g_camera.process_mouse(xpos, ypos)

def mouse_button_callback(window, button, action, mod):
    if imgui.get_io().want_capture_mouse:
        return
    pressed = action == glfw.PRESS
    g_camera.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed)
    g_camera.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed)

def wheel_callback(window, dx, dy):
    g_camera.process_wheel(dx, dy)

def key_callback(window, key, scancode, action, mods):
    if action == glfw.REPEAT or action == glfw.PRESS:
        if key == glfw.KEY_Q:
            g_camera.process_roll_key(1)
        elif key == glfw.KEY_E:
            g_camera.process_roll_key(-1)

def update_camera_pose_lazy():
    if g_camera.is_pose_dirty:
        g_renderer.update_camera_pose()
        g_camera.is_pose_dirty = False

def update_camera_intrin_lazy():
    if g_camera.is_intrin_dirty:
        g_renderer.update_camera_intrin()
        g_camera.is_intrin_dirty = False

def update_activated_renderer_state(gaus: util_gau.GaussianData):
    g_renderer.update_gaussian_data(gaus)
    g_renderer.sort_and_update()
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_screen_scale_factor(g_screen_scale_factor)
    g_renderer.set_rot_modifier(g_rot_modifier)
    g_renderer.set_render_mod(g_render_mode - 3)
    g_renderer.update_camera_pose()
    g_renderer.update_camera_intrin()
    g_renderer.set_render_reso(g_camera.w, g_camera.h)

def window_resize_callback(window, width, height):
    gl.glViewport(0, 0, width, height)
    g_camera.update_resolution(height, width)
    g_renderer.set_render_reso(width, height)

def main(args):
    global g_camera, g_renderer, g_renderer_list, g_renderer_idx, g_scale_modifier, g_screen_scale_factor, g_auto_sort, \
        g_show_gs_elements_control, g_show_help_control, g_show_camera_control, \
        g_render_mode, g_render_mode_tables, \
        dc_scale_factor, extra_scale_factor, g_rgb_factor, g_rot_modifier, g_light_rotation, \
        g_show_render_boundary_control, g_enable_render_boundary_aabb, g_enable_render_boundary_obb, use_axis_for_rotation, g_cube_min, g_cube_max, g_cube_rotation, tmp_cube_min, tmp_cube_max, tmp_cube_rotation, \
        show_axes, export_path, export_status, \
        g_background_color, g_show_scene_control, \
        g_directional_light_direction, g_directional_light_color, g_directional_light_intensity

    imgui.create_context()
    # 如果命令行参数中包含--hidpi，则启用HiDPI缩放
    if args.hidpi:
        imgui.get_io().font_global_scale = 1.5
    # # 自动判断是否进行1.5倍处理
    # user32 = ctypes.windll.user32
    # user32.SetProcessDPIAware()
    # screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    # if screen_width > 1920 or screen_height > 1080:  # 假设高于1080p的分辨率需要缩放
    #     imgui.get_io().font_global_scale = 1.5

    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    root = tk.Tk()  # used for file dialog
    root.withdraw()
    
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, wheel_callback)
    glfw.set_key_callback(window, key_callback)
    
    glfw.set_window_size_callback(window, window_resize_callback)

    # init renderer
    g_renderer_list[BACKEND_OGL] = OpenGLRenderer(g_camera.w, g_camera.h, g_camera)
    try:
        from render.renderer_cuda import CUDARenderer
        g_renderer_list += [CUDARenderer(g_camera.w, g_camera.h)]
    except ImportError:
        g_renderer_idx = BACKEND_OGL
    else:
        g_renderer_idx = BACKEND_CUDA

    g_renderer = g_renderer_list[g_renderer_idx]

    # gaussian data
    gaussians = util_gau.naive_gaussian()
    update_activated_renderer_state(gaussians)
    
    # settings
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        
        gl.glClearColor(g_background_color[0], g_background_color[1], 
                       g_background_color[2], g_background_color[3])
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        update_camera_pose_lazy()
        update_camera_intrin_lazy()
        
        g_renderer.draw()

        # imgui ui
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Window", True):
                clicked, g_show_gs_elements_control = imgui.menu_item(
                    "Show GS Elements Control", None, g_show_gs_elements_control
                )
                clicked, g_show_scene_control = imgui.menu_item(
                    "Show Scene Environment Control", None, g_show_scene_control
                )
                clicked, g_show_render_boundary_control = imgui.menu_item(
                    "Show Render Boundary", None, g_show_render_boundary_control
                )
                clicked, g_show_camera_control = imgui.menu_item(
                    "Show Camera Control", None, g_show_camera_control
                )
                clicked, g_show_help_control = imgui.menu_item(
                    "Show Help", None, g_show_help_control
                )
                imgui.end_menu()
            imgui.end_main_menu_bar()
        
        # 场景环境控制UI
        if g_show_scene_control:
            g_background_color, g_directional_light_direction, g_directional_light_color, g_directional_light_intensity = scene_environment_control_ui(
                g_show_scene_control, 
                g_background_color,
                g_directional_light_direction,
                g_directional_light_color,
                g_directional_light_intensity
            )
        
        # 显示GS元素控制UI
        if g_show_gs_elements_control:
            g_renderer, gaussians, g_camera, dc_scale_factor, extra_scale_factor, g_rgb_factor, g_rot_modifier, g_light_rotation, g_scale_modifier, g_screen_scale_factor, g_auto_sort, g_renderer_idx, g_renderer_list, g_render_mode, changed, show_axes = gs_elements_control_ui(
                window, g_renderer, gaussians, g_camera, dc_scale_factor, extra_scale_factor, g_rgb_factor, g_rot_modifier, g_light_rotation, g_scale_modifier, g_screen_scale_factor, g_auto_sort, g_renderer_idx, g_renderer_list, g_render_mode, g_render_mode_tables, show_axes
            )

        # 显示渲染包围盒控制UI
        if g_show_render_boundary_control:
            g_enable_render_boundary_aabb, g_enable_render_boundary_obb, g_cube_min, g_cube_max, g_cube_rotation, tmp_cube_min, tmp_cube_max, tmp_cube_rotation, use_axis_for_rotation, export_path, export_status = render_boundary_control_ui(
                g_renderer, gaussians, g_enable_render_boundary_aabb, g_enable_render_boundary_obb, g_cube_min, g_cube_max, g_cube_rotation, tmp_cube_min, tmp_cube_max, tmp_cube_rotation, use_axis_for_rotation, export_path, export_status
            )

        # 相机控制（包括相机旋转、平移、缩放）
        camera_control_ui(g_camera, g_show_camera_control)

        # 帮助内容
        help_window_ui(g_show_help_control)
        
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    # 保存原始的 sys.stderr
    original_stderr = sys.stderr

    try:
        parser = argparse.ArgumentParser(description="3DGS viewer with optional HiDPI support.")
        parser.add_argument("--hidpi", action="store_true", help="Enable HiDPI scaling for the interface.")
        args = parser.parse_args()

        main(args)
    except Exception as e:
        # 恢复 sys.stderr 以确保错误信息可以被正确打印
        sys.stderr = original_stderr
        print(f"An error occurred: {e}")
