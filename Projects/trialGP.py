import glfw
from OpenGL.GL import *
import numpy as np
from pyglm import mat4, vec3, perspective, lookAt, radians, value_ptr
import ctypes

# ========== SHADERS ==========
VERTEX_SHADER_SRC = """
#version 330 core
layout(location = 0) in vec3 aPos;

uniform mat4 uProjection;
uniform mat4 uView;
uniform mat4 uModel;

void main() {
    gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0);
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
out vec4 FragColor;

void main() {
    FragColor = vec4(0.6, 0.6, 0.6, 1.0);
}
"""

# ========== GRID LINE GENERATOR ==========
def generate_grid_lines(size=10, spacing=1.0):
    lines = []
    start = -size
    end = size

    # Lines parallel to Z axis
    for x in range(start, end + 1):
        lines.append([x * spacing, 0.0, start * spacing])
        lines.append([x * spacing, 0.0, end * spacing])

    # Lines parallel to X axis
    for z in range(start, end + 1):
        lines.append([start * spacing, 0.0, z * spacing])
        lines.append([end * spacing, 0.0, z * spacing])

    return np.array(lines, dtype=np.float32)

# ========== SHADER COMPILATION ==========
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compile failed:\n{error}")
    return shader

def create_shader_program(vertex_src, fragment_src):
    vertex_shader = compile_shader(vertex_src, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_src, GL_FRAGMENT_SHADER)

    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Shader link failed:\n{error}")

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program

# ========== MAIN FUNCTION ==========
def main():
    if not glfw.init():
        raise Exception("GLFW initialization failed")

    # OpenGL 3.3 Core Profile
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 600, "XZ Grid Reference Plane", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed")

    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)

    # === Grid ===
    grid_vertices = generate_grid_lines(size=10, spacing=1.0)
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, grid_vertices.nbytes, grid_vertices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * grid_vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # === Shader ===
    shader = create_shader_program(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC)
    glUseProgram(shader)

    # === Camera & Matrices (PyGLM) ===
    projection = perspective(radians(45.0), 800 / 600, 0.1, 100.0)
    view = lookAt(vec3(10.0, 10.0, 10.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0))
    model = mat4(1.0)

    # Pass matrices to shader
    glUniformMatrix4fv(glGetUniformLocation(shader, "uProjection"), 1, GL_FALSE, value_ptr(projection))
    glUniformMatrix4fv(glGetUniformLocation(shader, "uView"), 1, GL_FALSE, value_ptr(view))
    glUniformMatrix4fv(glGetUniformLocation(shader, "uModel"), 1, GL_FALSE, value_ptr(model))

    # === Render Loop ===
    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glBindVertexArray(VAO)
        glDrawArrays(GL_LINES, 0, len(grid_vertices))

        glfw.swap_buffers(window)

    # === Cleanup ===
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glfw.terminate()

if __name__ == "__main__":
    main()