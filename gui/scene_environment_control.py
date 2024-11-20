import imgui
import numpy as np

def scene_environment_control_ui(show_scene_control, background_color, directional_light_direction, directional_light_color, directional_light_intensity):
    if show_scene_control:
        imgui.begin("Scene Environment Control", True)
        
        imgui.push_item_width(200)

        # 背景颜色控制
        changed_bg, background_color = imgui.color_edit4("Background Color", *background_color)
        background_color = list(background_color)  # 确保是列表类型
        
        imgui.separator()
        
        imgui.text("Directional Light Settings")
        
        # 光源方向控制
        changed_dir, values = imgui.drag_float3("Light Direction", *directional_light_direction, 0.01, -1.0, 1.0)
        directional_light_direction = list(values)  # 确保是列表类型
        if changed_dir:
            length = np.sqrt(sum(x*x for x in values))
            if length > 0:
                directional_light_direction = [x/length for x in values]
        
        # 光源颜色控制
        changed_color, directional_light_color = imgui.color_edit4("Light Color", *directional_light_color)
        directional_light_color = list(directional_light_color)  # 确保是列表类型
        
        # 光源强度控制
        changed_intensity, directional_light_intensity = imgui.drag_float("Light Intensity", directional_light_intensity, 0.01, 0.0, 10.0)
        
        # 恢复默认宽度
        imgui.pop_item_width()
        imgui.end()
    
    return background_color, directional_light_direction, directional_light_color, directional_light_intensity

# def scene_environment_control_ui(show_scene_control, background_color, light_direction=[0.0, -1.0, 0.0], light_color=[1.0, 1.0, 1.0, 1.0], light_intensity=1.0):
#     if show_scene_control:
#         imgui.begin("Scene Environment Control", True)
        
#         # 设置颜色编辑器的宽度
#         imgui.push_item_width(200)  # 例如，设置宽度为200像素

#         # 创建颜色编辑器，支持RGBA
#         changed, background_color = imgui.color_edit4("Background Color", *background_color)
        
#         imgui.separator()

#         # 光源控制
#         imgui.text("Directional Light")
        
#         # 光源方向控制
#         changed_dir, light_direction = imgui.drag_float3("Direction", *light_direction, 0.01, -1.0, 1.0)
#         if changed_dir:
#             # 标准化方向向量
#             length = np.sqrt(sum(x*x for x in light_direction))
#             if length > 0:
#                 light_direction = [x/length for x in light_direction]
        
#         # 光源颜色控制
#         changed_color, light_color = imgui.color_edit4("Light Color", *light_color)
        
#         # 光源强度控制
#         changed_intensity, light_intensity = imgui.drag_float("Intensity", light_intensity, 0.01, 0.0, 10.0)

#         # 恢复默认宽度
#         imgui.pop_item_width()

#         imgui.end()
    
#     return background_color