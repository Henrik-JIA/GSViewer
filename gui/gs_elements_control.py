import imgui
from tkinter import filedialog
import util_gau
import glfw
import OpenGL.GL as gl
import imageio
import numpy as np
from PIL import Image
from datetime import datetime


def export_high_resolution(window, g_renderer, g_camera, scale_factor):
    """
    离屏高分辨率渲染导出
    
    Args:
        window: GLFW窗口
        g_renderer: 渲染器实例
        g_camera: 相机实例
        scale_factor: 放大倍数（2=2倍分辨率，4=4倍分辨率等）
    """
    # 获取当前窗口尺寸
    orig_width, orig_height = glfw.get_framebuffer_size(window)
    
    # 计算高分辨率尺寸
    hd_width = orig_width * scale_factor
    hd_height = orig_height * scale_factor
    
    print(f"[HD Export] Rendering at {hd_width}x{hd_height} ({scale_factor}x)...")
    
    # 弹出文件保存对话框
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"GSViewer_HD_{hd_width}x{hd_height}_{timestamp}.png"
    file_path = filedialog.asksaveasfilename(
        title=f"Export High Resolution Image ({hd_width}x{hd_height})",
        initialfile=default_filename,
        defaultextension=".png",
        filetypes=[
            ('PNG files', '*.png'),
            ('TIFF files', '*.tiff'),
            ('JPEG files', '*.jpg'),
            ('All files', '*.*')
        ]
    )
    
    if not file_path:
        print("[HD Export] Cancelled")
        return
    
    try:
        # ========== 创建离屏 FBO ==========
        # 创建帧缓冲对象
        fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
        
        # 创建颜色纹理
        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, hd_width, hd_height, 0, 
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, 
                                  gl.GL_TEXTURE_2D, texture, 0)
        
        # 创建深度渲染缓冲（可选，但有助于正确渲染）
        rbo = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, rbo)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH24_STENCIL8, hd_width, hd_height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_STENCIL_ATTACHMENT, 
                                     gl.GL_RENDERBUFFER, rbo)
        
        # 检查帧缓冲完整性
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            print("[HD Export] Error: Framebuffer is not complete!")
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            gl.glDeleteFramebuffers(1, [fbo])
            gl.glDeleteTextures(1, [texture])
            gl.glDeleteRenderbuffers(1, [rbo])
            return
        
        # ========== 备份并修改相机/渲染器状态 ==========
        # 备份原始相机分辨率
        orig_cam_w, orig_cam_h = g_camera.w, g_camera.h
        
        # 更新相机分辨率
        g_camera.update_resolution(hd_height, hd_width)
        g_camera.is_intrin_dirty = True
        
        # 更新渲染器
        g_renderer.set_render_reso(hd_width, hd_height)
        g_renderer.update_camera_intrin()
        g_renderer.update_camera_pose()
        
        # 设置视口
        gl.glViewport(0, 0, hd_width, hd_height)
        
        # ========== 渲染到 FBO ==========
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)  # 白色背景
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # 渲染场景
        g_renderer.draw()
        
        # 确保渲染完成
        gl.glFinish()
        
        # ========== 读取像素数据 ==========
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        bufferdata = gl.glReadPixels(0, 0, hd_width, hd_height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        img_array = np.frombuffer(bufferdata, np.uint8).reshape(hd_height, hd_width, 3)
        img_array = img_array[::-1]  # 翻转（OpenGL坐标系）
        
        # ========== 恢复原始状态 ==========
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        
        # 恢复相机分辨率
        g_camera.update_resolution(orig_cam_h, orig_cam_w)
        g_camera.is_intrin_dirty = True
        
        # 恢复渲染器
        g_renderer.set_render_reso(orig_width, orig_height)
        g_renderer.update_camera_intrin()
        g_renderer.update_camera_pose()
        
        # 恢复视口
        gl.glViewport(0, 0, orig_width, orig_height)
        
        # 清理 FBO 资源
        gl.glDeleteFramebuffers(1, [fbo])
        gl.glDeleteTextures(1, [texture])
        gl.glDeleteRenderbuffers(1, [rbo])
        
        # ========== 保存图像 ==========
        img = Image.fromarray(img_array, mode='RGB')
        
        # 根据文件扩展名选择保存格式，设置300 DPI
        if file_path.lower().endswith('.tiff') or file_path.lower().endswith('.tif'):
            img.save(file_path, dpi=(300, 300), compression='tiff_lzw')
        elif file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
            img.save(file_path, dpi=(300, 300), quality=95)
        else:
            img.save(file_path, dpi=(300, 300))
        
        # 计算打印尺寸
        print_width_cm = (hd_width / 300) * 2.54
        print_height_cm = (hd_height / 300) * 2.54
        
        print(f"[HD Export] ✓ Success!")
        print(f"[HD Export] File: {file_path}")
        print(f"[HD Export] Resolution: {hd_width} x {hd_height} pixels")
        print(f"[HD Export] Print size @ 300 DPI: {print_width_cm:.1f} x {print_height_cm:.1f} cm")
        
    except Exception as e:
        print(f"[HD Export] Error: {e}")
        # 确保恢复原始状态
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        g_camera.update_resolution(orig_cam_h, orig_cam_w)
        g_camera.is_intrin_dirty = True
        g_renderer.set_render_reso(orig_width, orig_height)
        g_renderer.update_camera_intrin()
        gl.glViewport(0, 0, orig_width, orig_height)

def gs_elements_control_ui(window, g_renderer, gaussians, g_camera, dc_scale_factor, extra_scale_factor, g_rgb_factor, g_rot_modifier, g_light_rotation, g_scale_modifier, g_screen_scale_factor, g_auto_sort, g_renderer_idx, g_renderer_list, g_render_mode, g_render_mode_tables, show_axes):
    changed = False

    if imgui.begin("Control", True):
        imgui.push_item_width(180)  # 设置滑动条宽度
        # rendering backend 执行图形渲染的底层技术选择
        changed_backend, g_renderer_idx = imgui.combo(
            "backend", g_renderer_idx, ["ogl", "cuda"][:len(g_renderer_list)]
            )
        if changed_backend:
            g_renderer = g_renderer_list[g_renderer_idx]
            update_activated_renderer_state(gaussians)

        imgui.text(f"fps = {imgui.get_io().framerate:.1f}")

        changed_reduce_updates, g_renderer.reduce_updates = imgui.checkbox(
                "Reduce updates", g_renderer.reduce_updates,
            )
        
        # 添加控制draw_axes的勾选框
        changed_show_axes, show_axes = imgui.checkbox("Show Axes", show_axes)
        if changed_show_axes:
            g_renderer.show_axes = show_axes
        
        # 添加反转鼠标方向的勾选框
        changed_invert_mouse, g_camera.invert_mouse = imgui.checkbox(
            "Invert Mouse", g_camera.invert_mouse)
        
        # 添加正射/透视投影切换
        changed_ortho, g_camera.use_orthographic = imgui.checkbox(
            "Orthographic View", g_camera.use_orthographic)
        if changed_ortho:
            g_camera.is_intrin_dirty = True
        
        # 正射视角缩放控制（仅在正射模式下显示）
        if g_camera.use_orthographic:
            changed_ortho_scale, g_camera.ortho_scale = imgui.slider_float(
                "Ortho Scale", g_camera.ortho_scale, 0.1, 50.0, "Scale = %.2f")
            imgui.same_line()
            if imgui.button("Reset##ortho_reset"):
                g_camera.ortho_scale = 5.0
                changed_ortho_scale = True
            if changed_ortho_scale:
                g_camera.is_intrin_dirty = True

        imgui.text(f"Gaus number = {len(gaussians)}")
        if imgui.button(label='Open ply'):
            file_path = filedialog.askopenfilename(title="open ply",
                initialdir="C:\\Users\\MSI_NB\\Downloads\\viewers",
                filetypes=[('ply file', '.ply')]
                )
            if file_path:
                try:
                    gaussians = util_gau.load_ply(file_path)  # 移除缩放因子参数
                    gaussians.scale_data(5.0)  # 应用缩放
                    g_renderer.update_gaussian_data(gaussians)
                    g_renderer.set_points_center(gaussians.points_center)
                    g_renderer.sort_and_update()
                except RuntimeError as e:
                    pass

        # 添加控制features_dc的滑动条
        changed_dc_scale, new_dc_scale_factor = imgui.slider_float(
            "DC", dc_scale_factor, 0.1, 2.0, "DC Scale Factor = %.2f")
        imgui.same_line()
        if imgui.button(label="Reset DC Scale"):
            new_dc_scale_factor = 1.
            changed_dc_scale = True
        if changed_dc_scale:
            g_renderer.adjust_dc_features(new_dc_scale_factor)
            dc_scale_factor = new_dc_scale_factor

        # 添加控制features_extra的滑动条
        changed_extra_scale, new_extra_scale_factor = imgui.slider_float(
            "Extra", extra_scale_factor, 0.1, 2.0, "Extra Scale Factor = %.2f")
        imgui.same_line()
        if imgui.button(label="Reset Extra Scale"):
            new_extra_scale_factor = 1.
            changed_extra_scale = True
        if changed_extra_scale:
            g_renderer.adjust_extra_features(new_extra_scale_factor)
            extra_scale_factor = new_extra_scale_factor

        # 在Control面板中添加滑动条，对rgb进行变化
        changed_red, g_rgb_factor[0] = imgui.slider_float(
            "R", g_rgb_factor[0], 0.00, 2.0, "Red = %.4f")
        imgui.same_line()
        if imgui.button(label="Reset Red"):
            g_rgb_factor[0] = 1.
            changed_red = True
        changed_green, g_rgb_factor[1] = imgui.slider_float(
            "G", g_rgb_factor[1], 0.00, 2.0, "Green = %.4f")
        imgui.same_line()
        if imgui.button(label="Reset Green"):
            g_rgb_factor[1] = 1.
            changed_green = True
        changed_blue, g_rgb_factor[2] = imgui.slider_float(
            "B", g_rgb_factor[2], 0.00, 2.0, "Blue = %.4f")
        imgui.same_line()
        if imgui.button(label="Reset Blue"):
            g_rgb_factor[2] = 1.
            changed_blue = True
        if changed_red or changed_green or changed_blue:
            # 当任何一个颜色滑动条的值改变时，更新渲染器中的颜色
            g_renderer.update_color_factor(g_rgb_factor)

        # Gaussian Scale Modifier
        changed_scale, g_scale_modifier = imgui.slider_float(
            "Gaussian Scale", g_scale_modifier, 0.0, 10, "Gaussian Scale = %.2f"
            )
        imgui.same_line()
        if imgui.button(label="Reset Gaussian Scale"):
            g_scale_modifier = 1.
            changed_scale = True

        if changed_scale:
            g_renderer.set_scale_modifier(g_scale_modifier)

        # Screen Display Scale Factor
        changed_screen_scale, g_screen_scale_factor = imgui.slider_float(
            "Screen Display Scale", g_screen_scale_factor, 0.0, 10, "Screen Display Scale = %.2f"
            )
        imgui.same_line()
        if imgui.button(label="Reset Screen Display Scale"):
            g_screen_scale_factor = 1.
            changed_screen_scale = True

        if changed_screen_scale:
            g_renderer.set_screen_scale_factor(g_screen_scale_factor)

        # 在ImGui的控制面板中添加旋转控制的滑动条
        changed_x, g_rot_modifier[0] = imgui.slider_float(
            "Rot X°", g_rot_modifier[0], -180.0, 180.0, "Rot X° = %.1f"
            )
        imgui.same_line()
        if imgui.button("Reset X"):
            g_rot_modifier[0] = 0.0
            changed_x = True
        changed_y, g_rot_modifier[1] = imgui.slider_float(
            "Rot Y°", g_rot_modifier[1], -180.0, 180.0, "Rot Y° = %.1f"
            )
        imgui.same_line()
        if imgui.button("Reset Y"):
            g_rot_modifier[1] = 0.0
            changed_y = True
        changed_z, g_rot_modifier[2] = imgui.slider_float(
            "Rot Z°", g_rot_modifier[2], -180.0, 180.0, "Rot Z° = %.1f"
            )
        imgui.same_line()
        if imgui.button("Reset Z"):
            g_rot_modifier[2] = 0.0
            changed_z = True
        # 当旋转滑动条的值改变时，或者任何一个轴的重置按钮被点击时
        if changed_x or changed_y or changed_z:
            g_renderer.set_rot_modifier(g_rot_modifier)

        # 添加控制光照旋转的滑动条
        changed_x, g_light_rotation[0] = imgui.slider_float(
            "Light Rot X°", g_light_rotation[0], -180.0, 180.0, "Light Rot X° = %.1f"
            )
        imgui.same_line()
        if imgui.button("Reset Light X"):
            g_light_rotation[0] = 0.0
            changed_x = True
        changed_y, g_light_rotation[1] = imgui.slider_float(
            "Light Rot Y°", g_light_rotation[1], -180.0, 180.0, "Light Rot Y° = %.1f"
            )
        imgui.same_line()
        if imgui.button("Reset Light Y"):
            g_light_rotation[1] = 0.0
            changed_y = True
        changed_z, g_light_rotation[2] = imgui.slider_float(
            "Light Rot Z°", g_light_rotation[2], -180.0, 180.0, "Light Rot Z° = %.1f"
            )
        imgui.same_line()
        if imgui.button("Reset Light Z"):
            g_light_rotation[2] = 0.0
            changed_z = True
        if changed_x or changed_y or changed_z:
            # 当任何一个轴的旋转角度改变时，更新渲染器中的光照旋转
            g_renderer.set_light_rotation(g_light_rotation)

        # render mode
        changed_render_mode, g_render_mode = imgui.combo(
            "Shading", g_render_mode, g_render_mode_tables
            )
        if changed_render_mode:
            g_renderer.set_render_mod(g_render_mode - 6)

        # sort button
        if imgui.button(label='sort Gaussians'):
            g_renderer.sort_and_update()
        imgui.same_line()
        changed_auto_sort, g_auto_sort = imgui.checkbox(
                "Auto Sort", g_auto_sort,
            )
        if g_auto_sort:
            g_renderer.sort_and_update()

        # 快速保存（屏幕分辨率）
        if imgui.button(label='Quick Save'):
            width, height = glfw.get_framebuffer_size(window)
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
            gl.glReadBuffer(gl.GL_FRONT)
            bufferdata = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
            imageio.imwrite("Save.png", img[::-1])
            print(f"[Quick Save] Saved to Save.png ({width}x{height})")
        
        imgui.same_line()
        
        # 高分辨率导出（离屏渲染）
        if imgui.button(label='HD Export'):
            imgui.open_popup("hd_export_popup")
        
        # 高分辨率导出弹窗
        if imgui.begin_popup("hd_export_popup"):
            imgui.text("Select export resolution:")
            imgui.separator()
            
            # 预设分辨率选项
            export_options = [
                ("2x (Print ~20cm)", 2),
                ("3x (Print ~30cm)", 3),
                ("4x (Print ~40cm, 4K)", 4),
                ("6x (Print ~60cm, 6K)", 6),
                ("8x (Print ~80cm, 8K)", 8),
            ]
            
            for label, scale in export_options:
                if imgui.selectable(label)[0]:
                    # 执行高分辨率导出
                    export_high_resolution(window, g_renderer, g_camera, scale)
            
            imgui.end_popup()

        imgui.pop_item_width()  # 恢复滑动条默认宽度
        imgui.end()

    return g_renderer, gaussians, g_camera, dc_scale_factor, extra_scale_factor, g_rgb_factor, g_rot_modifier, g_light_rotation, g_scale_modifier, g_screen_scale_factor, g_auto_sort, g_renderer_idx, g_renderer_list, g_render_mode, changed, show_axes