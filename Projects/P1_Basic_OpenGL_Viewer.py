#H4
from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

# Projection settings
g_is_perspective = True

# Setting initial angle and distance
g_cam_azimuth = np.radians(45.)
g_cam_elevation = np.radians(45.)
g_cam_distance = 10.

# lookAt function arguments
g_eye = glm.vec3(g_cam_distance * np.cos(g_cam_elevation) * np.sin(g_cam_azimuth), g_cam_distance * np.sin(g_cam_elevation), g_cam_distance *
                 np.cos(g_cam_elevation) * np.cos(g_cam_azimuth))
g_target = glm.vec3(0.0, 0.0, 0.0) #changed to g_target (g_at)
g_up_vector = glm.vec3(0, 1, 0)

# Default mouse position
g_prev_xpos = 0.
g_prev_ypos = 0.


#---------------------(New)
g_target = glm.vec3(0.0, 0.0, 0.0)
g_is_orbit = False
g_is_pan = False
g_is_zoom = False

g_vertex_shader_src = '''
# version 330 core

layout (location = 0) in vec3 vin_pos;
layout (location = 1) in vec3 vin_color;

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src = '''
# version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color; 
}
'''


def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------

    # vertex shader
    # create an empty shader object
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    # provide shader source code
    glShaderSource(vertex_shader, vertex_shader_source)
    # compile the shader object
    glCompileShader(vertex_shader)

    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())

    # fragment shader
    # create an empty shader object
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    # provide shader source code
    glShaderSource(fragment_shader, fragment_shader_source)
    # compile the shader object
    glCompileShader(fragment_shader)

    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    # create an empty program object
    shader_program = glCreateProgram()
    # attach the shader objects to the program object
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program

# #--------------------------------------------------------------------(edited)
def mouse_button_callback(window, button, action, mods):
    global g_is_orbit, g_is_pan, g_is_zoom
    if button == GLFW_MOUSE_BUTTON_LEFT:
        if action == GLFW_PRESS:
            if mods == GLFW_MOD_ALT:
                g_is_orbit = True
            elif mods == (GLFW_MOD_ALT | GLFW_MOD_SHIFT):
                g_is_pan = True
            elif mods == (GLFW_MOD_ALT | GLFW_MOD_CONTROL):
                g_is_zoom = True
        elif action == GLFW_RELEASE:
            g_is_orbit = g_is_pan = g_is_zoom = False

def cursor_callback(window, xpos, ypos):
    global g_cam_azimuth, g_cam_elevation, g_cam_distance
    global g_eye, g_target, g_up_vector, g_prev_xpos, g_prev_ypos #changed to g_target(g_at)

    delta_x = xpos - g_prev_xpos
    delta_y = ypos - g_prev_ypos
    
    if g_is_orbit:
        # Orbit: Change azimuth and elevation angles
        sensitivity = 0.01
        g_cam_azimuth += delta_x * sensitivity
        g_cam_elevation = np.clip(g_cam_elevation + delta_y * sensitivity, -np.pi/2 + 0.01, np.pi/2 - 0.01)
        
        # Update eye position using spherical coordinates
        g_eye = g_target + glm.vec3(
            g_cam_distance * np.cos(g_cam_elevation) * np.sin(g_cam_azimuth),
            g_cam_distance * np.sin(g_cam_elevation),
            g_cam_distance * np.cos(g_cam_elevation) * np.cos(g_cam_azimuth)
        )
    
    elif g_is_pan:
        # Pan: Move both camera and target in camera's local frame
        sensitivity = 0.01
        cam_w = glm.normalize(g_eye - g_target)
        cam_u = glm.normalize(glm.cross(g_up_vector, cam_w))
        cam_v = glm.normalize(glm.cross(cam_w, cam_u))
        
        pan = cam_u * delta_x * sensitivity + cam_v * delta_y * sensitivity
        g_target += pan
        g_eye += pan
    
    elif g_is_zoom:
        # Zoom: Move camera along view direction
        sensitivity = 0.01
        zoom_factor = -delta_y * sensitivity  # Negative because pulling up should zoom in
        view_dir = glm.normalize(g_target - g_eye)
        g_cam_distance = max(0.1, g_cam_distance * (1.0 - zoom_factor))
        g_eye = g_target - view_dir * g_cam_distance

    g_prev_xpos = xpos
    g_prev_ypos = ypos

# # #--------------------------------------------------------------------(Edited)

def key_callback(window, key, scancode, action, mods):
    global g_cam_azimuth, g_is_perspective
    if key == GLFW_KEY_ESCAPE and action == GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)


def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
                         # position        # color
                         -15.0, 0.0, 0.0,  1.0, 0.0, 0.0,  # x-axis start
                         15.0, 0.0, 0.0,  1.0, 0.0, 0.0,  # x-axis end
                         0.0, 0.0, 0.0,  0.0, 1.0, 0.0,  # y-axis start
                         0.0, 0.0, 0.0,  0.0, 1.0, 0.0,  # y-axis end
                         0.0, 0.0, -15.0,  0.0, 0.0, 1.0,  # z-axis start
                         0.0, 0.0, 15.0,  0.0, 0.0, 1.0,  # z-axis end
                         )

    # create and activate VAO (vertex array object)
    # create a vertex array object ID and store it to VAO variable
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    # create a buffer object ID and store it to VBO variable
    VBO = glGenBuffers(1)
    # activate VBO as a vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    # allocate GPU memory for and copy vertex data to the currently bound vertex buffer
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes,
                 vertices.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

#--------------------------------------------------------------------
def prepare_vao_xz():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
                         # position        # color
                         -15.0, 0.0, 0.0,  1.0, 1.0, 1.0,  # x-axis start
                         15.0, 0.0, 0.0,  1.0, 1.0, 1.0,  # x-axis end
                         0.0, 0.0, -15.0,  1.0, 1.0, 1.0,  # z-axis start
                         0.0, 0.0, 15.0,  1.0, 1.0, 1.0,  # z-axis end
                         )

    # create and activate VAO (vertex array object)
    # create a vertex array object ID and store it to VAO variable
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    # create a buffer object ID and store it to VBO variable
    VBO = glGenBuffers(1)
    # activate VBO as a vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    # allocate GPU memory for and copy vertex data to the currently bound vertex buffer
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes,
                 vertices.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 *
                          glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO
#--------------------------------------------------------------------

def main():
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)  # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, '2022055741_Project1', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    
    #----------------------------------(Edited)
    # register event callbacks
    glfwSetMouseButtonCallback(window, mouse_button_callback)
    glfwSetCursorPosCallback(window, cursor_callback)


    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')

    # prepare vaos
    vao_frame = prepare_vao_frame()

    vao_xz = prepare_vao_xz()

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        # projection matrix
        global g_is_perspective
        if g_is_perspective:
            P = glm.perspective(45, 1, 1, 100)

        # view matrix
        # rotate camera position with g_cam_azimuth / move camera up & down with g_cam_height
        V = glm.lookAt(g_eye, g_target, g_up_vector)

        # current frame: P*V*I (now this is the world frame)
        I = glm.mat4()
        MVP = P*V*I
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 6)

        # draw cube
        # draw_cube(vao_cube, MVP, MVP_loc)

        # draw x grid
        for i in range(-15, 16):
            T = glm.translate(glm.vec3(0, 0, i))
            M = T

            MVP = P*V*M
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

            glBindVertexArray(vao_xz)
            glDrawArrays(GL_LINES, 0, 2)

        # draw z grid
        for i in range(-15, 16):
            T = glm.translate(glm.vec3(i, 0, 0))
            M = T

            MVP = P*V*M
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

            glBindVertexArray(vao_xz)
            glDrawArrays(GL_LINES, 2, 4)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()


if __name__ == "__main__":
    main()
