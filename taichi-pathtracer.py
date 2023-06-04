import numpy as np
import taichi as ti
import taichi.math as tm


ti.init(arch=ti.gpu)

@ti.func
def quadratic_eqn_roots(a, b, c) -> tm.vec2:

    discriminant = b ** 2 - 4 * a * c
    root = tm.vec2([0, 0])
    if a == 0:
        root = tm.vec2([0, 0])
    elif discriminant < 0:
        root = tm.vec2([0, 0])
    elif discriminant == 0:
        root = tm.vec2([-b / (2 * a), -b / (2 * a)])
    else:
        sqrt_discriminant = ti.sqrt(discriminant)
        root = tm.vec2([(-b - sqrt_discriminant) / (2 * a), (-b + sqrt_discriminant) / (2 * a)])
    return root

@ti.func
def lorentz_boost(beta):

    beta_vec = ti.Vector([beta[0], beta[1], beta[2]])

    beta_squared = beta_vec.dot(beta_vec)

    gamma = 1 / ti.sqrt(1 - beta_squared)

    lambda_00 = ti.Matrix([[gamma]])
    lambda_0j = -gamma * beta
    # lambda_i0 = lambda_0j.transpose()
    lambda_ij = ti.Matrix.identity(ti.f32, 3) + (gamma - 1) * beta_vec.outer_product(beta_vec) / beta_squared

    return ti.Matrix([[lambda_00[0, 0], lambda_0j[0], lambda_0j[1], lambda_0j[2]],
                      [lambda_0j[0], lambda_ij[0, 0], lambda_ij[0, 1], lambda_ij[0, 2]],
                      [lambda_0j[1], lambda_ij[1, 0], lambda_ij[1, 1], lambda_ij[1, 2]],
                      [lambda_0j[2], lambda_ij[2, 0], lambda_ij[2, 1], lambda_ij[2, 2]]])


ORIGIN = tm.vec4([0, 0, 0, 0])
DEFAULT_OBJ_COLOR = tm.vec3([255, 255, 255])
DEFAULT_BG_COLOR = tm.vec3([0.0, 0.0, 0.0])

@ti.dataclass
class Material:
    albedo: tm.vec3
    emission: tm.vec3
    roughness: float
    metallic: float
    transmission: float
    ior: float

@ti.dataclass
class Ray:
    """A ray in Minkowski space."""
    start: tm.vec4
    direction: tm.vec4

    @ti.func
    def boost(self, boost_matrix):
        return Ray(boost_matrix @ self.start, boost_matrix @ self.direction)

    @ti.func
    def translate(self, offset):
        return Ray(self.start + offset, self.direction)
    
    @ti.func
    def at(self, t):
        return self.start + t * self.direction

@ti.dataclass
class Sphere:
    """A (3d) sphere centered at the origin."""
    center: tm.vec3
    radius: float
    material: Material
    color: tm.vec3 = DEFAULT_OBJ_COLOR

    @ti.func
    def get_intersection_and_color(self, ray: Ray):
        intersection = self._get_intersection(ray)
        color = tm.vec3([0.0, 0.0, 0.0])
        if intersection[0] != -1.0:
            color = DEFAULT_OBJ_COLOR

        return intersection, color

    @ti.func
    def _get_intersection(self, ray: Ray) -> tm.vec3:
        x0 = tm.vec3(ray.start[1:4])-self.center
        d = tm.vec3(ray.direction[1:4])

        a = d.dot(d)
        b = 2 * x0.dot(d)
        c = x0.dot(x0) - self.radius ** 2

        solns = quadratic_eqn_roots(a, b, c)
        t = tm.vec3([-1.0, -1.0, -1.0])
        if solns[0] > 0:
            t = x0 + solns[0]*d
        elif solns[1] > 0:
            t = x0 + solns[1]*d
        return t
    
    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):
        x0 = tm.vec3(ray.start[1:4])-self.center
        d = tm.vec3(ray.direction[1:4])

        a = d.dot(d)
        b = 2 * x0.dot(d)
        c = x0.dot(x0) - self.radius ** 2

        solns = quadratic_eqn_roots(a, b, c)
        root = tm.vec3([-1.0, -1.0, -1.0])
        is_hit = False
        front_face = False
        hit_point =  ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        if solns[0]<t_min or solns[0]>t_max:
            if solns[1]>=t_min and solns[1]<=t_max:
                root = solns[1]
                is_hit = True
        else:
            root = solns[0]
            is_hit = True
        if is_hit:
            hit_point = ray.at(root)
            hit_point_normal = (hit_point - self.center) / self.radius
            # Check which side does the ray hit, we set the hit point normals always point outward from the surface
            if ray.direction.dot(hit_point_normal) < 0:
                front_face = True
            else:
                hit_point_normal = -hit_point_normal
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color
    
@ti.func
def rand3():
    return ti.Vector([ti.random(), ti.random(), ti.random()])

@ti.func
def random_in_unit_sphere():
    p = 2.0 * rand3() - ti.Vector([1, 1, 1])
    while p.norm() >= 1.0:
        p = 2.0 * rand3() - ti.Vector([1, 1, 1])
    return p

@ti.func
def random_unit_vector():
    return random_in_unit_sphere().normalized()

@ti.func
def to_light_source(hit_point, light_source):
    return light_source - hit_point

@ti.func
def reflect(v, normal):
    return v - 2 * v.dot(normal) * normal

@ti.func
def refract(uv, n, etai_over_etat):
    cos_theta = min(n.dot(-uv), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.dot(r_out_perp))) * n
    return r_out_perp + r_out_parallel

@ti.func
def reflectance(cosine, ref_idx):
    # Use Schlick's approximation for reflectance.
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * pow((1 - cosine), 5)

@ti.data_oriented
class Hittable_list:
    def __init__(self):
        self.objects = []
    def add(self, obj):
        self.objects.append(obj)
    def clear(self):
        self.objects = []

    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):
        closest_t = t_max
        is_hit = False
        front_face = False
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        color = ti.Vector([0.0, 0.0, 0.0])
        material = 1
        for index in ti.static(range(len(self.objects))):
            is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp =  self.objects[index].obj.hit(ray, t_min, closest_t)
            if is_hit_tmp:
                closest_t = root_tmp
                is_hit = is_hit_tmp
                hit_point = hit_point_tmp
                hit_point_normal = hit_point_normal_tmp
                front_face = front_face_tmp
                material = material_tmp
                color = color_tmp
        return is_hit, hit_point, hit_point_normal, front_face, material, color

    @ti.func
    def hit_shadow(self, ray, t_min=0.001, t_max=10e8):
        is_hit_source = False
        is_hit_source_temp = False
        hitted_dielectric_num = 0
        is_hitted_non_dielectric = False
        # Compute the t_max to light source
        is_hit_tmp, root_light_source, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp = \
        self.objects[0].hit(ray, t_min)
        for index in ti.static(range(len(self.objects))):
            is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp =  self.objects[index].hit(ray, t_min, root_light_source)
            if is_hit_tmp:
                if material_tmp != 3 and material_tmp != 0:
                    is_hitted_non_dielectric = True
                if material_tmp == 3:
                    hitted_dielectric_num += 1
                if material_tmp == 0:
                    is_hit_source_temp = True
        if is_hit_source_temp and (not is_hitted_non_dielectric) and hitted_dielectric_num == 0:
            is_hit_source = True
        return is_hit_source, hitted_dielectric_num, is_hitted_non_dielectric

@ti.dataclass
class MovingObject:
    obj: Sphere
    beta: tm.vec3
    offset: tm.vec4
#----------------------------------------
# 设置场景
ti.pi = 3.14159
height = 150
width = 500
focal_length = 200

# Rendering parameters
samples_per_pixel = 4
max_depth = 10
sample_on_unit_sphere_surface = True

image_matrix = ti.Vector.field(3, float, (width, height))

beta = [0, 0, 0.8]
offset = [0, -200, 0, 200]

scene = Hittable_list()
scene.add(MovingObject(Sphere(tm.vec3(100,100,0), 50, Material(tm.vec3(1, 1, 1)*0.4, tm.vec3(1), 1, 0, 0, 1.530)), beta, offset))

WORLD_LIST = [
    MovingObject(Sphere(tm.vec3(100,100,0), 50, Material(tm.vec3(1, 1, 1)*0.4, tm.vec3(1), 1, 0, 0, 1.530)), beta, offset),
]

objects_num = len(WORLD_LIST)
objects = MovingObject.field(shape=objects_num)
for i in range(objects_num): objects[i] = WORLD_LIST[i]

#----------------------------------------
@ti.func
def ray_color(ray):
    color_buffer = ti.Vector([0.0, 0.0, 0.0])
    brightness = ti.Vector([1.0, 1.0, 1.0])
    scattered_origin = ray.start
    scattered_direction = ray.direction
    p_RR = 0.8
    for n in range(max_depth):
        if ti.random() > p_RR:
            break
        is_hit, hit_point, hit_point_normal, front_face, material, color = scene.hit(Ray(scattered_origin, scattered_direction))
        if is_hit:
            if material == 0:
                color_buffer = color * brightness
                break
            else:
                # Diffuse
                if material == 1:
                    target = hit_point + hit_point_normal
                    if sample_on_unit_sphere_surface:
                        target += random_unit_vector()
                    else:
                        target += random_in_unit_sphere()
                    scattered_direction = target - hit_point
                    scattered_origin = hit_point
                    brightness *= color
                # Metal and Fuzz Metal
                elif material == 2 or material == 4:
                    fuzz = 0.0
                    if material == 4:
                        fuzz = 0.4
                    scattered_direction = reflect(scattered_direction.normalized(),
                                                  hit_point_normal)
                    if sample_on_unit_sphere_surface:
                        scattered_direction += fuzz * random_unit_vector()
                    else:
                        scattered_direction += fuzz * random_in_unit_sphere()
                    scattered_origin = hit_point
                    if scattered_direction.dot(hit_point_normal) < 0:
                        break
                    else:
                        brightness *= color
                # Dielectric
                elif material == 3:
                    refraction_ratio = 1.5
                    if front_face:
                        refraction_ratio = 1 / refraction_ratio
                    cos_theta = min(-scattered_direction.normalized().dot(hit_point_normal), 1.0)
                    sin_theta = ti.sqrt(1 - cos_theta * cos_theta)
                    # total internal reflection
                    if refraction_ratio * sin_theta > 1.0 or reflectance(cos_theta, refraction_ratio) > ti.random():
                        scattered_direction = reflect(scattered_direction.normalized(), hit_point_normal)
                    else:
                        scattered_direction = refract(scattered_direction.normalized(), hit_point_normal, refraction_ratio)
                    scattered_origin = hit_point
                    brightness *= color
                brightness /= p_RR
    return color_buffer


@ti.func
def trace_ray(time: ti.f32, bg_color: tm.vec3, x: ti.i32, y: ti.i32) -> tm.vec3:
    # 计算洛伦兹变换矩阵
    boost_matrix = lorentz_boost(objects[0].beta)
    # 计算图像平面上的坐标
    origin_to_image_time = ti.sqrt(x ** 2 + y ** 2 + focal_length ** 2)

    image_coords = tm.vec4([-origin_to_image_time, x, y, focal_length])
    # 将相机坐标系中的射线转换到物体坐标系中
    camera_frame_ray = Ray(ORIGIN, image_coords).translate(tm.vec4([time, 0, 0, 0]))

    object_frame_ray = camera_frame_ray.translate(-objects[0].offset).boost(boost_matrix)

    # 计算射线与物体的交点和颜色
    color = tm.vec3([0.0, 0.0, 0.0])
    # for n in range(samples_per_pixel):
        #color += ray_color(object_frame_ray)
    #color /= samples_per_pixel
    intersection, color = objects[0].obj.get_intersection_and_color(object_frame_ray)
    if color[0] == -1:
        color = bg_color
    return color

@ti.kernel
def render(time: ti.f32, bg_color: tm.vec3):

    for i, j in ti.ndrange(width, height):
        # 计算像素点在图像平面上的坐标
        x = i - width / 2 + ti.random()
        y = -(j - height / 2 + ti.random())
        # 计算像素点的颜色
        color = trace_ray(time, bg_color, x, y)
        # 将像素点的颜色存储到图像矩阵中
        image_matrix[i, j] = color
'''''''''
@ti.kernel
def render():
    for i, j in canvas:
        u = (i + ti.random()) / image_width
        v = (j + ti.random()) / image_height
        color = ti.Vector([0.0, 0.0, 0.0])
        for n in range(samples_per_pixel):
            ray = camera.get_ray(u, v)
            color += ray_color(ray)
        color /= samples_per_pixel
        canvas[i, j] += color
'''''''''
def example():

    for time in range(0, 1600, 100):
        render(time, bg_color=DEFAULT_BG_COLOR)
        ti.tools.imwrite(image_matrix.to_numpy(), './out/example_' + str(time) + '.png')

if __name__ == '__main__':
    example()
