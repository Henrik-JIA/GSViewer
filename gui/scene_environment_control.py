import imgui

def scene_environment_control_ui(show_scene_control, background_color):
    if show_scene_control:
        imgui.begin("Scene Environment Control", True)
        
        # 设置颜色编辑器的宽度
        imgui.push_item_width(200)  # 例如，设置宽度为200像素

        # 创建颜色编辑器，支持RGBA
        changed, background_color = imgui.color_edit4("Background Color", *background_color)
        
        # 恢复默认宽度
        imgui.pop_item_width()

        imgui.end()
    
    return background_color