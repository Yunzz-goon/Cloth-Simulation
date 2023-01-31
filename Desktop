#
# NOTE - THIS CODE IS SLIGHTLY BROKEN - Read Assignment Page
# First thing you need to do is fix it. We will discuss in lectures
#
from vpython import *
from random import random
import getopt, sys


class ball(object):
    def __init__(self, x=0.0, y=0.0, z=0.0, r=1.0):

        # note we make radius of sphere slightly shorter than it should be - why?
        self.x = x
        self.y = y
        self.z = z
        self.radius = r
        self.visible = sphere(
            pos=vector(x, y, z), radius=r * 0.95, color=vector(0, 1, 0)
        )


def create_cloth(N, ballsize, nodes):
    #   create nodes in cloth
    for nx in range(N):
        x = nx * separation - (N - 1) * separation * 0.5 + offset
        for ny in range(N):
            y = ny * separation - (N - 1) * separation * 0.5 + offset
            # if you set radius to something you will see the nodes
            node = sphere(
                pos=vector(x, ballsize + 1.0, y), radius=0, color=vector(1, 1, 0)
            )
            node.force = vector(0.0, 0.0, 0.0)
            node.velocity = vector(0.0, 0.0, 0.0)
            node.oldforce = vector(0.0, 0.0, 0.0)
            node.dist = vector(0.0, 0.0, 0.0)
            node.exceed = False
            nodes.append(node)
    #   colour squares between nodes
    for nx in range(N - 1):
        for ny in range(N - 1):
            c = vector(random(), random(), random())
            pt1 = vertex(pos=nodes[nx * N + ny].pos, color=c)
            pt2 = vertex(pos=nodes[(nx + 1) * N + ny].pos, color=c)
            pt3 = vertex(pos=nodes[(nx + 1) * N + ny + 1].pos, color=c)
            pt4 = vertex(pos=nodes[nx * N + ny + 1].pos, color=c)
            nodes[nx * N + ny].fill = quad(v0=pt1, v1=pt2, v2=pt3, v3=pt4)


def compute_force(delta, gravity, separation, fcon):
    r12 = vector(0.0, 0.0, 0.0)
    PE = 0.0
    #   loop over nodes in x and y direction
    for nx in range(N):
        for ny in range(N):
            #   add gravitational force
            nodes[nx * N + ny].force = vector(0.0, -gravity, 0.0)
            #   for node (nx,ny) loop over surrounding nodes and eval force/PE
            for dx in range(max(nx - delta, 0), min(nx + delta + 1, N)):
                for dy in range(max(ny - delta, 0), min(ny + delta + 1, N)):
                    len = sqrt(float((nx - dx) ** 2 + (ny - dy) ** 2)) * separation
                    #   don't self interact
                    if nx != dx or ny != dy:
                        r12 = nodes[dx * N + dy].pos - nodes[nx * N + ny].pos
                        PE += fcon * (r12.mag - len) * (r12.mag - len)
                        nodes[nx * N + ny].force += fcon * r12.norm() * (r12.mag - len)
    return PE


def usage():
    print(" -h or --help : This info")
    print(" -v or --verbose")
    print(" -n or --nodes Nodes_per_dimension (int) ")
    print(" -s or --separation Grid_separation (float)")
    print(" -m or --mass Mass_of_node (float)")
    print(" -f or --fcon Force_constant (float)")
    print(" -i or --interact Node_interaction_level (int)")
    print(" -g or --gravity Gravity (float)")
    print(" -b or --ballsize Radius_of_ball (float)")
    print(" -o or --offset offset_of_falling_cloth (float)")
    print(" -t or --timestep timestep (float)")
    print(" -u or --update Timesteps_per_display_update (int)")
    return


def read_arg(argv):
    #   fancy input processing
    global verbose, dt, N, mass, fcon, separation, ballsize, gravity, offset, interact, update
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hvn:s:m:f:i:g:t:o:u:g:b:d:",
            [
                "help",
                "verbose",
                "nodes=",
                "separation=",
                "mass=",
                "fcon=",
                "interact=",
                "gravity=",
                "ballsize=",
                "offset=",
                "timestep=",
                "update=",
            ],
        )
    except getopt.GetoptError:
        print("using default parameters")
        opts = {}
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-v", "--verbose"):
            verbose = 1
        elif o in ("-n", "--nodes"):
            N = int(a)
        elif o in ("-s", "--separation"):
            separation = float(a)
        elif o in ("-m", "--mass"):
            mass = float(a)
        elif o in ("-f", "--fcon"):
            fcon = float(a)
        elif o in ("-i", "--interact"):
            interact = int(a)
        elif o in ("-g", "--gravity"):
            gravity = float(a)
        elif o in ("-b", "--ballsize"):
            ballsize = float(a)
        elif o in ("-o", "--offset"):
            offset = float(a)
        elif o in ("-t", "--timestep"):
            dt = float(a)
        elif o in ("-u", "--update"):
            update = int(a)
        else:
            assert False, "unhandled option"

    print("The cloth ")
    print("  Nodes per dimension ", N)
    print("  Grid Separation     ", separation)
    print("  Mass of node        ", mass)
    print("  Force constant      ", fcon)
    print("  Node interaction    ", interact)
    print("The Environment")
    print("  Gravity             ", gravity)
    print("  Ballsize            ", ballsize)
    print("  Offset              ", offset)
    print("The Simulation")
    print("  Timestep            ", dt)
    print("  Updates per display ", update)
    print("  Verbose             ", verbose)
    return


# some default input parameters
N = int(20)
separation = float(1.0)
mass = float(1.0)
fcon = float(10.0)
interact = int(2)
gravity = float(0.981)
ballsize = float(5.0)
offset = float(0.0)
dt = float(0.02)
update = int(2)
verbose = int(1)
read_arg(sys.argv[1:])

scene.autoscale = 0
myball = ball(0, 0, 0, ballsize)

nodes = []
create_cloth(N, ballsize, nodes)
PE = compute_force(interact, gravity, separation, fcon)

iter = 0
maxit = 400
while iter < maxit:
    iter += 1
    if verbose:
        print("iteration and potential energy ", iter, PE)

    #   Update coordinates using same MD velocity verlet
    for node in nodes:
        node.pos += dt * (node.velocity + dt * node.force * 0.5)
        node.oldforce = node.force

    #   apply constraints (move nodes to surface of ball)
    for node in nodes:
        node.dist = node.pos - vector(myball.x, myball.y, myball.z)  # node与球心的距离向量，方向指出圆心
        if node.dist.mag < myball.radius:
            fvector = node.dist / node.dist.mag * myball.radius
            node.pos = vector(myball.x, myball.y, myball.z) + fvector
            node.exceed = True
            # move to newest surface(node与圆心连线的反向延长线与surface的交点)

    if iter % update == 0:
        #   update the view if necessary
        for nx in range(N - 1):
            for ny in range(N - 1):
                nodes[nx * N + ny].fill.v0.pos = nodes[nx * N + ny].pos
                nodes[nx * N + ny].fill.v1.pos = nodes[(nx + 1) * N + ny].pos
                nodes[nx * N + ny].fill.v2.pos = nodes[(nx + 1) * N + ny + 1].pos
                nodes[nx * N + ny].fill.v3.pos = nodes[nx * N + ny + 1].pos

        sleep(0.05)
    PE = compute_force(interact, gravity, separation, fcon)

    #   Update velocity using same MD velocity verlet
    damp = 0.995
    for node in nodes:
        node.velocity += dt * (node.force + node.oldforce) * 0.5 * damp
        if node.dist.mag < myball.radius:
            # notice that the direction of dist is outward to the sphere
            node.velocity = -node.velocity
            fvector = node.dist / node.dist.mag * dot(node.velocity, node.dist) / node.dist.mag
            node.velocity = node.velocity - fvector

