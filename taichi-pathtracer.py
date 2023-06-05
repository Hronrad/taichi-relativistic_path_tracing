import numpy as np
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

@ti.func
def vec3_add_time(vec3, time):
    vec4 = tm.vec4(time, vec3[0], vec3[1], vec3[2])
    return vec4

@ti.func
def vec4_to_vec3(vec4):
    vec3 = tm.vec3(vec4[1:4])
    return vec3
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
    beta_vec = beta

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

'''''''''
@ti.func
def SPD(color, f):
    cr = color[0]
    cg = color[1]
    cb = color[2]
    c = 0.0
    if f < FRC:
        c = 0.2 * cr
    else:
        if(f >= FRC) and (f < FR):
            c = 0.8 * cr * (f - FRC) / (FR - FRC) + 0.2 * cr
        else:
            if (f >= FR) and (f < FG):
                c = cg + (cr - cg) * ((FR / f) * FG - FR) / (FG - FR)
            else:
                if (f >= FG) and (f < FB):
                    c = cb + (cg - cb) * ((FG / f) * FB - FG) / (FB - FG)
                else:
                    if (f >= FB) and (f < FBC):
                        c = 0.8 * cb * (FBC - f) / (FBC - FB) + 0.2 * cb
                    else:
                        c = 0.2 * cb
    return c
'''''''''

wavelength_min = 380
wavelength_max = 720
hue_min = -180
hue_max = 370

@ti.func
def rgb_to_hsv(r, g, b):
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    h = 0.0
    s = 0.0
    if minc != maxc:
        s = (maxc-minc) / maxc
        rc = (maxc-r) / (maxc-minc)
        gc = (maxc-g) / (maxc-minc)
        bc = (maxc-b) / (maxc-minc)
        if r == maxc:
            h = bc-gc
        elif g == maxc:
            h = 2.0+rc-bc
        else:
            h = 4.0+gc-rc
        h = (h/6.0) % 1.0
    return h, s, v

@ti.func
def hsv_to_rgb(h, s, v):
    r1 = 0.0
    r2 = 0.0
    r3 = 0.0
    if s == 0.0:
        r1, r2, r3 = v, v, v
    
    i = int(h*6.0) # XXX assume int() truncates!
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    if i == 0:
        r1, r2, r3 = v, t, p
    if i == 1:
        r1, r2, r3 = q, v, p
    if i == 2:
        r1, r2, r3 = p, v, t
    if i == 3:
        r1, r2, r3 = p, q, v
    if i == 4:
        r1, r2, r3 = t, p, v
    if i == 5:
        r1, r2, r3 = v, p, q
    return r1, r2, r3


@ti.func
def wave_to_color(wavelength, saturation=1, value=1):
    hue_min = -180
    hue_max = 370
    # 计算波长范围对应的色相
    hue = 0.0
    hue = (650 - wavelength) * 1.3714 
    hue = ((hue - hue_min) / (hue_max - hue_min) - 0.2) % 1.0 
    return hsv_to_rgb(hue ,saturation, value)

@ti.func
def color_to_wave(color):
    hue, saturation, value = rgb_to_hsv(color[0], color[1], color[2])
    if 0.80 < hue < 0.875:
        hue = 0.80
    elif 0.875 <= hue < 0.95:
        hue = 0.95
    res = 0.0
    if hue > 0.9:
        res = (hue - 1 + 0.2)*(hue_max - hue_min) + hue_min
    else:
        res = (hue + 0.2)*(hue_max - hue_min) + hue_min
    res = 650 - res/1.3714
    return res, saturation, value


@ti.func
def Doppler(color, factor):
    wavelength, saturation, value = color_to_wave(color)
    wavelength = wavelength / factor
    if wavelength < 380:
        wavelength = 380
    elif wavelength > 720:
        wavelength = 720
    return wave_to_color(wavelength, saturation, value)


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
        return tm.vec3(vec4_to_vec3(self.start) + t * vec4_to_vec3(self.direction))

@ti.dataclass
class Sphere:
    """A (3d) sphere centered at the origin."""
    center: tm.vec3
    radius: float
    material: int
    color: tm.vec3
  
    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):

        x0 = vec4_to_vec3(ray.start)-self.center
        d = vec4_to_vec3(ray.direction)

        a = d.dot(d)
        b = 2 * x0.dot(d)
        c = x0.dot(x0) - self.radius ** 2

        solns = quadratic_eqn_roots(a, b, c)

        root = tm.vec3([-1.0, -1.0, -1.0])
        is_hit = False
        front_face = False
        hit_point =  tm.vec3([0.0, 0.0, 0.0])
        hit_point_normal = tm.vec3([0.0, 0.0, 0.0])
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
            if tm.vec3(ray.direction[1:4]).dot(hit_point_normal) < 0:
                front_face = True
            else:
                hit_point_normal = -hit_point_normal

        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color
    
@ti.func
def rand3():
    return tm.vec3([ti.random(), ti.random(), ti.random()])

@ti.func
def random_in_unit_sphere():
    p = 2.0 * rand3() - tm.vec3([1, 1, 1])
    while p.norm() >= 1.0:
        p = 2.0 * rand3() - tm.vec3([1, 1, 1])
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
    def hit(self, ray, t_min=0.001, t_max=10e7):

        is_hit = False
        front_face = False
        hit_point = tm.vec3([0.0, 0.0, 0.0])
        hit_point_normal = tm.vec3([0.0, 0.0, 0.0])
        color = tm.vec3([0.0, 0.0, 0.0])
        material = 1
        for index in ti.static(range(len(self.objects))):
            is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp =  self.objects[index].obj.hit(ray, t_min, t_max)
            if is_hit_tmp:
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
#-----------------------------------------------------------------------------------
# 设置场景
ti.pi = 3.14159
eps = 1e-4
height = 800
width = 800
focal_length = 1400.0

ORIGIN = tm.vec4([0, 0, 0, 0])
DEFAULT_OBJ_COLOR = tm.vec3([1, 1, 1])
DEFAULT_BG_COLOR = tm.vec3([0.0, 0.0, 0.0])

# Rendering parameters
samples_per_pixel = 1200
max_depth = 10

image_matrix = ti.Vector.field(3, ti.f32, (width, height))

# beta = v/c
beta = tm.vec3([0, 0, -0.4])
offset = tm.vec4([0, 0, 0, 10])

scene = Hittable_list()
# scene.add(MovingObject(Sphere(center=tm.vec3(20,20,50), radius = 20, material = 0, color=DEFAULT_OBJ_COLOR), beta, offset))
# Light source
scene.add(MovingObject(Sphere(center=tm.vec3([0, 5.4, -1]), radius=30.0, material=0, color=ti.Vector([22.0, 22.0, 22.0])), beta, offset))
# Ground
scene.add(MovingObject(Sphere(center=ti.Vector([0, -100.5, -1]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])), beta, offset))
# ceiling
scene.add(MovingObject(Sphere(center=ti.Vector([0, 152.5, -1]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])), beta, offset))
# back wall
scene.add(MovingObject(Sphere(center=ti.Vector([0, 1, 101]), radius=100.0, material=1, color=ti.Vector([0.5, 0.5, 0.8])), beta, offset))
# front wall
#scene.add(MovingObject(Sphere(center=ti.Vector([0, 1, -208]), radius=100.0, material=1, color=ti.Vector([0.6, 0.0, 0.0])), beta, offset))
# right wall
scene.add(MovingObject(Sphere(center=ti.Vector([-101.5, 0, -1]), radius=100.0, material=1, color=ti.Vector([0.6, 0.0, 0.0])), beta, offset))
# left wall
scene.add(MovingObject(Sphere(center=ti.Vector([101.5, 0, -1]), radius=100.0, material=1, color=ti.Vector([0.0, 0.6, 0.0])), beta, offset))
# Diffuse ball
scene.add(MovingObject(Sphere(center=ti.Vector([0, -0.2, -1.5]), radius=0.3, material=1, color=ti.Vector([0.8, 0.3, 0.3])), beta, offset))
# Metal ball
scene.add(MovingObject(Sphere(center=ti.Vector([-0.8, 0.2, -1]), radius=0.7, material=2, color=ti.Vector([0.6, 0.8, 0.8])), beta, offset))
# Glass ball
scene.add(MovingObject(Sphere(center=ti.Vector([0.7, 0, -0.5]), radius=0.5, material=3, color=ti.Vector([1.0, 1.0, 1.0])), beta, offset))
# Metal ball-2
scene.add(MovingObject(Sphere(center=ti.Vector([0.6, -0.3, -2.0]), radius=0.2, material=4, color=ti.Vector([0.8, 0.6, 0.2])), beta, offset))


# WORLD_LIST = [
#     MovingObject(Sphere(tm.vec3(100,100,0), 50, Material(tm.vec3(1, 1, 1)*0.4, tm.vec3(1), 1, 0, 0, 1.530)), beta, offset),
# ]

# objects_num = len(WORLD_LIST)
# objects = MovingObject.field(shape=objects_num)
# for i in range(objects_num): objects[i] = WORLD_LIST[i]

#-----------------------------------------------------------------------------------
@ti.func
def ray_color(ray, time):
    color_buffer = tm.vec3([0.0, 0.0, 0.0])
    brightness = tm.vec3([1.0, 1.0, 1.0])
    scattered_origin = ray.start
    scattered_direction = ray.direction
    scattered_origin_3 = vec4_to_vec3(scattered_origin)
    scattered_direction_3 = vec4_to_vec3(scattered_direction)

    p_RR = 0.8
    count = 0
    for n in range(max_depth):
        if ti.random() > p_RR:
            break
        scattered_origin_3 = vec4_to_vec3(scattered_origin)
        scattered_direction_3 = vec4_to_vec3(scattered_direction)
        scattered_ray_3 = scattered_direction_3 - scattered_origin_3
        is_hit, hit_point, hit_point_normal, front_face, material, color = scene.hit(Ray(scattered_origin, scattered_direction))
        if is_hit:
            count += 1
            #print('hhit', hit_point, hit_point_normal, front_face, material, color)
            cos_theta = beta.dot(scattered_ray_3.normalized())
            D = ti.sqrt(1 - beta.norm_sqr()) / (1 + cos_theta)
            #print('D', D)
            if material == 0:
                color_buffer = color * brightness
                break
            else:
                # Diffuse
                if material == 1:
                    target = hit_point + hit_point_normal
                    target += random_unit_vector()
                    scattered_direction = vec3_add_time(target - hit_point, time)
                    scattered_origin = vec3_add_time(hit_point, time)
                    if count == 1:
                        color = Doppler(color, D)
                    brightness *= color
                # Metal and Fuzz Metal
                elif material == 2 or material == 4:
                    fuzz = 0.0
                    if material == 4:
                        fuzz = 0.4
                    scattered_direction = vec3_add_time(reflect(vec4_to_vec3(scattered_direction).normalized(),
                                                  hit_point_normal), time)
                    scattered_direction += vec3_add_time(fuzz * random_unit_vector(), time)
                    scattered_origin = vec3_add_time(hit_point, time)
                    if vec4_to_vec3(scattered_direction).dot(hit_point_normal) < 0:
                        break
                    else:
                        if count == 1:
                            color = Doppler(color, D)
                        brightness *= color
                # Dielectric
                elif material == 3:
                    refraction_ratio = 1.5
                    if front_face:
                        refraction_ratio = 1 / refraction_ratio
                    cos_theta = min(-vec4_to_vec3(scattered_direction).normalized().dot(hit_point_normal), 1.0)
                    sin_theta = ti.sqrt(1 - cos_theta * cos_theta)
                    # total internal reflection
                    if refraction_ratio * sin_theta > 1.0 or reflectance(cos_theta, refraction_ratio) > ti.random():
                        scattered_direction = vec3_add_time(reflect(vec4_to_vec3(scattered_direction).normalized(), hit_point_normal), time)
                    else:
                        scattered_direction = vec3_add_time(refract(vec4_to_vec3(scattered_direction).normalized(), hit_point_normal, refraction_ratio), time)
                    scattered_origin = vec3_add_time(hit_point, time)
                    if count == 1:
                        color = Doppler(color, D)                        
                    brightness *= color
                brightness /= p_RR
    return color_buffer


@ti.func
def trace_ray(time: ti.f32, x: ti.f32, y: ti.f32) -> tm.vec3:
    #print('tracing ray at', time, x, y, '...')
    # 计算洛伦兹变换矩阵
    boost_matrix = lorentz_boost(beta)

    origin_to_image_time = ti.sqrt(x ** 2 + y ** 2 + focal_length ** 2)

    image_coords = tm.vec4([-origin_to_image_time, x, y, focal_length])

    camera_frame_ray = Ray(ORIGIN, image_coords).translate(tm.vec4([time, 0, 0, 0]))

    object_frame_ray = camera_frame_ray.translate(-offset).boost(boost_matrix)

    # 计算射线与物体的交点和颜色
    color = tm.vec3([0.0, 0.0, 0.0])
    for n in range(samples_per_pixel):
        color += ray_color(object_frame_ray, time)
    color /= samples_per_pixel
    return color

@ti.kernel
def render(time: ti.f32):
    print('rendering at', time, '...')
    for i, j in ti.ndrange(width, height):
        # 计算像素点在图像平面上的坐标
        x = i - width / 2 + ti.random()
        y = -(j - height / 2 + ti.random())
        # 计算像素点的颜色
        color = trace_ray(time, x, y)
        # 将像素点的颜色存储到图像矩阵中
        image_matrix[width -i -1, height -j -1] = color


def example():
    print('Rendering...')
    for time in range(0, 25, 1):
        render(time)
        ti.tools.imwrite(image_matrix.to_numpy(), './out/example_' + str(time) + '.png')


if __name__ == '__main__':
    example()
