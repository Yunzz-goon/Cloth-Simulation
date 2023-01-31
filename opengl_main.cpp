
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "./cloth_code.h"
#include "./cloth_param.h"

// ---GUI dependencies for Apple OS X
#if __APPLE__ & __MACH__
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
// ---GUI dependencies for Linux
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif

#include <math.h>

// static GLfloat btrans[3]; // current background translation
// static GLfloat brot[3];   // current background rotation

// static GLfloat trans[3]; // current translation
// static GLfloat rot[3];   // current model rotation
static GLfloat crot[3]; // current camera rotation

// static GLuint startList;
static GLuint body;
static GLuint mainball;

// static GLfloat plasticbody_color[] = {1.0, 0.39, 0.39, 1.0};
// static GLfloat mat_ambient[] = {0.7, 0.7, 0.7, 1.0};
static GLfloat mat_diffuse[] = {0.1, 0.5, 0.8, 1.0};
static GLfloat mat_specular[] = {0.80, 0.8, 0.80, 1.0};
// static GLfloat light_amb[] = {0.525, 0.525, 0.525, 1.0};
static GLfloat light_position[] = {25.0, 45.0, 19.0, 1.0};
static GLfloat lmodel_ambient[] = {0.3, 0.3, 0.3, 1.0};

static char progress[512];
static int mousex, mousey;

static struct timeval *tp1, *tp2;

static GLUquadricObj *qobj;

#define crossProduct(a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1, d2)           \
  (d0) = ((b1) - (a1)) * ((c2) - (a2)) - ((c1) - (a1)) * ((b2) - (a2));        \
  (d1) = ((b2) - (a2)) * ((c0) - (a0)) - ((c2) - (a2)) * ((b0) - (a0));        \
  (d2) = ((b0) - (a0)) * ((c1) - (a1)) - ((c0) - (a0)) * ((b1) - (a1));

static void keyboard(unsigned char key, int x, int y);
static void special_key(int key, int x, int y);
static void motion(int x, int y);
static void mouse(int button, int state, int x, int y);
static void display(void);
static void reshape(int width, int height);
static void bitmap_string(void *font, float x, float y, char *str);
static void draw_scene();
static void init();
static void reset_simulation();

static void idleTime(void) // when idle...
{
  float elapsed;
  double pe, ke, te;

  if (paused) {
    snprintf(progress, sizeof(progress),
             "Nodes %d Separation %5.2f  Ball %4.2f Offset %4.2f", n, sep,
             rball, offset);
    glutPostRedisplay(); // -----GUI
    usleep(10000);
  } else {
    /***************************************************/
    /* HERE YOU DO 1 DYNAMICS TIMESTEP WITH GUI        */
    /* YOU NEED TO WRITE THIS ROUTINE IN MYCODE.C      */

    loopcode(n, mass, fcon, delta, grav, sep, rball, xball, yball, zball, dt, x,
             y, z, fx, fy, fz, vx, vy, vz, oldfx, oldfy, oldfz, &pe, &ke, &te);

    /***************E N D*******************************/
    if (loop % update == 0) {
      gettimeofday(tp2, NULL);
      elapsed = (float)(tp2->tv_sec - tp1->tv_sec) +
                (float)(tp2->tv_usec - tp1->tv_usec) * 1.0e-6;
      snprintf(
          progress, sizeof(progress),
          "Iteration %10d PE %10.5f KE %10.5f TE %10.5f Elapsed Time %12.6f",
          loop, pe, ke, te, elapsed);
      glutPostRedisplay(); // -----GUI
      printf("Iteration %10d PE %10.5f  KE %10.5f  TE %10.5f \n ", loop, pe, ke,
             te);
    }
    loop++;
    /* For presentations, reset simulation every so often */
    // if (reset_time > 1 && elapsed > reset_time) {
    if (loop >= maxiter) {
      // reset_simulation();
      paused = 1;
    }
  }
}

int main(int argc, char **argv) {

  int i;
  // assess input flags
  if (argc % 2 == 0)
    argv[1][0] = 'x';

  for (i = 1; i < argc; i += 2) {
    if (argv[i][0] == '-') {
      switch (argv[i][1]) {
      case 'n':
        n = atoi(argv[i + 1]);
        break;
      case 's':
        sep = atof(argv[i + 1]);
        break;
      case 'm':
        mass = atof(argv[i + 1]);
        break;
      case 'f':
        fcon = atof(argv[i + 1]);
        break;
      case 'd':
        delta = atoi(argv[i + 1]);
        break;
      case 'g':
        grav = atof(argv[i + 1]);
        break;
      case 'b':
        rball = atof(argv[i + 1]);
        break;
      case 'o':
        offset = atof(argv[i + 1]);
        break;
      case 't':
        dt = atof(argv[i + 1]);
        break;
      case 'u':
        update = atoi(argv[i + 1]);
        break;
      case 'r':
        rendermode = atoi(argv[i + 1]);
        break;
      case 'i':
        maxiter = atoi(argv[i + 1]);
        break;
      default:
        printf(" %s\n"
               "Nodes_per_dimension:             -n int \n"
               "Grid_separation:                 -s float \n"
               "Mass_of_node:                    -m float \n"
               "Force_constant:                  -f float \n"
               "Node_interaction_level:          -d int \n"
               "Gravity:                         -g float \n"
               "Radius_of_ball:                  -b float \n"
               "offset_of_falling_cloth:         -o float \n"
               "timestep:                        -t float \n"
               "num iterations:                  -i int \n"
               "Timesteps_per_display_update:    -u int \n"
               "Perform X timesteps without GUI: -x int\n"
               "Rendermode (1 for face shade, 2 for vertex shade, 3 for no "
               "shade):\n"
               "                                 -r (1,2,3)\n",
               argv[0]);
        return -1;
      }
    } else {
      printf(
          " %s\n"
          "Nodes_per_dimension:             -n int \n"
          "Grid_separation:                 -s float \n"
          "Mass_of_node:                    -m float \n"
          "Force_constant:                  -f float \n"
          "Node_interaction_level:          -d int \n"
          "Gravity:                         -g float \n"
          "Radius_of_ball:                  -b float \n"
          "offset_of_falling_cloth:         -o float \n"
          "timestep:                        -t float \n"
          "num iterations:                  -i int \n"
          "Timesteps_per_display_update:    -u int \n"
          "Perform X timesteps without GUI: -x int\n"
          "Rendermode (1 for face shade, 2 for vertex shade, 3 for no shade):\n"
          "                                 -r (1,2,3)\n",
          argv[0]);
      return -1;
    }
  }

  // print out values to be used in the program
  printf("____________________________________________________\n"
         "_____ COMP3320 Assignment 2 - Cloth Simulation _____\n"
         "____________________________________________________\n"
         "Number of nodes per dimension:  %d\n"
         "Grid separation:                %f\n"
         "Mass of node:                   %f\n"
         "Force constant                  %f\n"
         "Node Interaction Level (delta): %d\n"
         "Gravity:                        %f\n"
         "Radius of Ball:                 %f\n"
         "Offset of falling cloth:        %f\n"
         "Timestep:                       %f\n"
         "Timesteps per display update:   %i\n",
         n, sep, mass, fcon, delta, grav, rball, offset, dt, update);

  tp1 = (struct timeval *)malloc(sizeof(struct timeval));
  tp2 = (struct timeval *)malloc(sizeof(struct timeval));

  /***************************************************/
  /* HERE WE INITIALISE THE DYANMICS PART OF THE CODE*/
  /* YOU NEED TO WRITE THIS ROUTINE IN MYCODE.C      */

  initMatrix(n, mass, fcon, delta, grav, sep, rball, offset, dt, &x, &y, &z,
             &cpx, &cpy, &cpz, &fx, &fy, &fz, &vx, &vy, &vz, &oldfx, &oldfy,
             &oldfz);

  /***************E N D*******************************/

  printf("\nDrag mouse to rotate\n"
         "KEYS:\n"
         "spacebar to pause/resume\n"
         "'R' to reset simulation\n"
         "up/down arrow keys to change number of nodes\n"
         "left/right arrow keys to change node spacing\n"
         "'b' or 'B' to decrease or increase ball size\n"
         "'[' or ']' to offset cloth\n"
         "'-' to zoom out\n"
         "'=' to zoom in\n"
         "'1' for per face shading\n"
         "'2' for per vertex shading\n"
         "'3' for no shading\n"
         "'Esc' to quit\n");

  // ----- GUI starts here
  glutInitWindowSize(1024, 800);
  glutInitWindowPosition(128, 64);
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
  glutCreateWindow("Cloth Simulation");

  init(); // set up all lists and other stuff

  glutKeyboardFunc(keyboard);
  glutSpecialFunc(special_key);
  glutMotionFunc(motion);
  glutMouseFunc(mouse);
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutIdleFunc(idleTime);

  gettimeofday(tp1, NULL);
  glutMainLoop();
  return 1;
}

void noshade(void) {
  int nx, ny;

  glDisable(GL_LIGHTING);
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  glNormal3f(1, 1, 1);

  for (nx = 0; nx < n - 1; nx++) {
    for (ny = 0; ny < n - 1; ny++) {
      glBegin(GL_TRIANGLES);
      glVertex3f(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny]);
      glVertex3f(x[(nx + 1) * n + ny], y[(nx + 1) * n + ny],
                 z[(nx + 1) * n + ny]);
      glVertex3f(x[nx * n + ny + 1], y[nx * n + ny + 1], z[nx * n + ny + 1]);
      glVertex3f(x[(nx + 1) * n + ny], y[(nx + 1) * n + ny],
                 z[(nx + 1) * n + ny]);
      glVertex3f(x[nx * n + ny + 1], y[nx * n + ny + 1], z[nx * n + ny + 1]);
      glVertex3f(x[(nx + 1) * n + ny + 1], y[(nx + 1) * n + ny + 1],
                 z[(nx + 1) * n + ny + 1]);
      glEnd();
    }
  }
}

void faceshade(void) {
  int nx, ny;
  float cx, cy, cz;

  glEnable(GL_LIGHTING);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  for (nx = 0; nx < n - 1; nx++) {
    for (ny = 0; ny < n - 1; ny++) {
      glBegin(GL_TRIANGLES);
      crossProduct(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny],
                   x[(nx + 1) * n + ny], y[(nx + 1) * n + ny],
                   z[(nx + 1) * n + ny], x[nx * n + ny + 1], y[nx * n + ny + 1],
                   z[nx * n + ny + 1], cx, cy, cz);
      glNormal3f(cx, cy, cz);
      glVertex3f(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny]);
      glVertex3f(x[(nx + 1) * n + ny], y[(nx + 1) * n + ny],
                 z[(nx + 1) * n + ny]);
      glVertex3f(x[nx * n + ny + 1], y[nx * n + ny + 1], z[nx * n + ny + 1]);
      crossProduct(x[(nx + 1) * n + ny], y[(nx + 1) * n + ny],
                   z[(nx + 1) * n + ny], x[nx * n + ny + 1], y[nx * n + ny + 1],
                   z[nx * n + ny + 1], x[(nx + 1) * n + ny + 1],
                   y[(nx + 1) * n + ny + 1], z[(nx + 1) * n + ny + 1], cx, cy,
                   cz);
      glNormal3f(-cx, -cy, -cz);
      glVertex3f(x[(nx + 1) * n + ny], y[(nx + 1) * n + ny],
                 z[(nx + 1) * n + ny]);
      glVertex3f(x[nx * n + ny + 1], y[nx * n + ny + 1], z[nx * n + ny + 1]);
      glVertex3f(x[(nx + 1) * n + ny + 1], y[(nx + 1) * n + ny + 1],
                 z[(nx + 1) * n + ny + 1]);
      glEnd();
    }
  }
}

void vertexshade(void) {
  int nx, ny;
  float cx, cy, cz;

  glEnable(GL_LIGHTING);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  for (nx = 0; nx < n; nx++) {
    for (ny = 0; ny < n; ny++) {
      cpx[n * nx + ny] = 0;
      cpz[n * nx + ny] = 0;
      cpy[n * nx + ny] = 0;
    }
  }
  nx = 0;
  ny = 0;
  crossProduct(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny],
               x[nx * n + ny + 1], y[nx * n + ny + 1], z[nx * n + ny + 1],
               x[(nx + 1) * n + ny], y[(nx + 1) * n + ny], z[(nx + 1) * n + ny],
               cx, cy, cz);
  cpx[n * nx + ny] -= cx;
  cpz[n * nx + ny] -= cy;
  cpy[n * nx + ny] -= cz;
  ny = n - 1;
  crossProduct(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny],
               x[(nx + 1) * n + ny], y[(nx + 1) * n + ny], z[(nx + 1) * n + ny],
               x[nx * n + ny - 1], y[nx * n + ny - 1], z[nx * n + ny - 1], cx,
               cy, cz);
  cpx[n * nx + ny] -= cx;
  cpz[n * nx + ny] -= cy;
  cpy[n * nx + ny] -= cz;
  nx = n - 1;
  crossProduct(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny],
               x[nx * n + ny - 1], y[nx * n + ny - 1], z[nx * n + ny - 1],
               x[(nx - 1) * n + ny], y[(nx - 1) * n + ny], z[(nx - 1) * n + ny],
               cx, cy, cz);
  cpx[n * nx + ny] -= cx;
  cpz[n * nx + ny] -= cy;
  cpy[n * nx + ny] -= cz;
  ny = 0;
  crossProduct(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny],
               x[(nx - 1) * n + ny], y[(nx - 1) * n + ny], z[(nx - 1) * n + ny],
               x[nx * n + ny + 1], y[nx * n + ny + 1], z[nx * n + ny + 1], cx,
               cy, cz);
  cpx[n * nx + ny] -= cx;
  cpz[n * nx + ny] -= cy;
  cpy[n * nx + ny] -= cz;

  for (ny = 1; ny < n - 1; ny++) {
    nx = 0;
    crossProduct(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny],
                 x[nx * n + ny + 1], y[nx * n + ny + 1], z[nx * n + ny + 1],
                 x[(nx + 1) * n + ny], y[(nx + 1) * n + ny],
                 z[(nx + 1) * n + ny], cx, cy, cz);
    cpx[n * nx + ny] -= cx;
    cpz[n * nx + ny] -= cy;
    cpy[n * nx + ny] -= cz;
    crossProduct(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny],
                 x[(nx + 1) * n + ny], y[(nx + 1) * n + ny],
                 z[(nx + 1) * n + ny], x[nx * n + ny - 1], y[nx * n + ny - 1],
                 z[nx * n + ny - 1], cx, cy, cz);
    cpx[n * nx + ny] -= cx;
    cpz[n * nx + ny] -= cy;
    cpy[n * nx + ny] -= cz;
    nx = n - 1;
    crossProduct(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny],
                 x[(nx - 1) * n + ny], y[(nx - 1) * n + ny],
                 z[(nx - 1) * n + ny], x[nx * n + ny + 1], y[nx * n + ny + 1],
                 z[nx * n + ny + 1], cx, cy, cz);
    cpx[n * nx + ny] -= cx;
    cpz[n * nx + ny] -= cy;
    cpy[n * nx + ny] -= cz;
    crossProduct(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny],
                 x[nx * n + ny - 1], y[nx * n + ny - 1], z[nx * n + ny - 1],
                 x[(nx - 1) * n + ny], y[(nx - 1) * n + ny],
                 z[(nx - 1) * n + ny], cx, cy, cz);
    cpx[n * nx + ny] -= cx;
    cpz[n * nx + ny] -= cy;
    cpy[n * nx + ny] -= cz;
  }

  for (nx = 1; nx < n - 1; nx++) {
    ny = 0;
    crossProduct(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny],
                 x[(nx - 1) * n + ny], y[(nx - 1) * n + ny],
                 z[(nx - 1) * n + ny], x[nx * n + ny + 1], y[nx * n + ny + 1],
                 z[nx * n + ny + 1], cx, cy, cz);
    cpx[n * nx + ny] -= cx;
    cpz[n * nx + ny] -= cy;
    cpy[n * nx + ny] -= cz;
    crossProduct(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny],
                 x[nx * n + ny + 1], y[nx * n + ny + 1], z[nx * n + ny + 1],
                 x[(nx + 1) * n + ny], y[(nx + 1) * n + ny],
                 z[(nx + 1) * n + ny], cx, cy, cz);
    cpx[n * nx + ny] -= cx;
    cpz[n * nx + ny] -= cy;
    cpy[n * nx + ny] -= cz;
    ny = n - 1;
    crossProduct(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny],
                 x[(nx + 1) * n + ny], y[(nx + 1) * n + ny],
                 z[(nx + 1) * n + ny], x[nx * n + ny - 1], y[nx * n + ny - 1],
                 z[nx * n + ny - 1], cx, cy, cz);
    cpx[n * nx + ny] -= cx;
    cpz[n * nx + ny] -= cy;
    cpy[n * nx + ny] -= cz;
    crossProduct(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny],
                 x[nx * n + ny - 1], y[nx * n + ny - 1], z[nx * n + ny - 1],
                 x[(nx - 1) * n + ny], y[(nx - 1) * n + ny],
                 z[(nx - 1) * n + ny], cx, cy, cz);
    cpx[n * nx + ny] -= cx;
    cpz[n * nx + ny] -= cy;
    cpy[n * nx + ny] -= cz;
  }

  for (nx = 1; nx < n - 1; nx++) {
    for (ny = 1; ny < n - 1; ny++) {
      crossProduct(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny],
                   x[(nx - 1) * n + ny], y[(nx - 1) * n + ny],
                   z[(nx - 1) * n + ny], x[nx * n + ny + 1], y[nx * n + ny + 1],
                   z[nx * n + ny + 1], cx, cy, cz);
      cpx[n * nx + ny] -= cx;
      cpz[n * nx + ny] -= cy;
      cpy[n * nx + ny] -= cz;
      crossProduct(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny],
                   x[nx * n + ny + 1], y[nx * n + ny + 1], z[nx * n + ny + 1],
                   x[(nx + 1) * n + ny], y[(nx + 1) * n + ny],
                   z[(nx + 1) * n + ny], cx, cy, cz);
      cpx[n * nx + ny] -= cx;
      cpz[n * nx + ny] -= cy;
      cpy[n * nx + ny] -= cz;
      crossProduct(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny],
                   x[(nx + 1) * n + ny], y[(nx + 1) * n + ny],
                   z[(nx + 1) * n + ny], x[nx * n + ny - 1], y[nx * n + ny - 1],
                   z[nx * n + ny - 1], cx, cy, cz);
      cpx[n * nx + ny] -= cx;
      cpz[n * nx + ny] -= cy;
      cpy[n * nx + ny] -= cz;
      crossProduct(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny],
                   x[nx * n + ny - 1], y[nx * n + ny - 1], z[nx * n + ny - 1],
                   x[(nx - 1) * n + ny], y[(nx - 1) * n + ny],
                   z[(nx - 1) * n + ny], cx, cy, cz);
      cpx[n * nx + ny] -= cx;
      cpz[n * nx + ny] -= cy;
      cpy[n * nx + ny] -= cz;
    }
  }
  for (nx = 0; nx < n - 1; nx++) {
    for (ny = 0; ny < n - 1; ny++) {
      glBegin(GL_TRIANGLES);
      glNormal3f(cpx[nx * n + ny], cpy[nx * n + ny], cpz[nx * n + ny]);
      glVertex3f(x[nx * n + ny], y[nx * n + ny], z[nx * n + ny]);
      glNormal3f(cpx[(nx + 1) * n + ny], cpy[(nx + 1) * n + ny],
                 cpz[(nx + 1) * n + ny]);
      glVertex3f(x[(nx + 1) * n + ny], y[(nx + 1) * n + ny],
                 z[(nx + 1) * n + ny]);
      glNormal3f(cpx[nx * n + ny + 1], cpy[nx * n + ny + 1],
                 cpz[nx * n + ny + 1]);
      glVertex3f(x[nx * n + ny + 1], y[nx * n + ny + 1], z[nx * n + ny + 1]);
      glNormal3f(cpx[(nx + 1) * n + ny], cpy[(nx + 1) * n + ny],
                 cpz[(nx + 1) * n + ny]);
      glVertex3f(x[(nx + 1) * n + ny], y[(nx + 1) * n + ny],
                 z[(nx + 1) * n + ny]);
      glNormal3f(cpx[nx * n + ny + 1], cpy[nx * n + ny + 1],
                 cpz[nx * n + ny + 1]);
      glVertex3f(x[nx * n + ny + 1], y[nx * n + ny + 1], z[nx * n + ny + 1]);
      glNormal3f(cpx[(nx + 1) * n + ny + 1], cpy[(nx + 1) * n + ny + 1],
                 cpz[(nx + 1) * n + ny + 1]);
      glVertex3f(x[(nx + 1) * n + ny + 1], y[(nx + 1) * n + ny + 1],
                 z[(nx + 1) * n + ny + 1]);
      glEnd();
    }
  }
}

static void init(void) {
  int temp;
  float ballrendersize;

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_NORMALIZE);
  glEnable(GL_POINT_SMOOTH);

  body = glGenLists(1);
  mainball = glGenLists(1);
  qobj = gluNewQuadric();
  gluQuadricNormals(qobj, GLU_SMOOTH);
  temp = 3;
  ballrendersize = rball - 0.06;
  if (ballrendersize < 0.01)
    ballrendersize = rball;
  glNewList(body, GL_COMPILE); // sphere
  gluSphere(qobj, 0.1, temp, temp);
  glEndList();
  glNewList(mainball, GL_COMPILE); // sphere
  gluSphere(qobj, ballrendersize * 0.98, 8, 8);
  glEndList();

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glLightfv(GL_LIGHT0, GL_SPECULAR, mat_specular);
  glLightfv(GL_LIGHT0, GL_POSITION, light_position);
  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_AUTO_NORMAL);
}

static void reset_simulation() {
  gettimeofday(tp1, NULL);
  gettimeofday(tp2, NULL);
  loop = 0;
  initMatrix(n, mass, fcon, delta, grav, sep, rball, offset, dt, &x, &y, &z,
             &cpx, &cpy, &cpz, &fx, &fy, &fz, &vx, &vy, &vz, &oldfx, &oldfy,
             &oldfz);
  init(); // Because the ball size might have changed
  paused = 1;
}

void mouse(int UNUSED(button), int UNUSED(state), int x,
           int y) { // when button is pressed
  mousex = x; // set the x and y coords for use in calculating movement in next
              // iteration
  mousey = y;
}

void motion(int x, int y) {
  crot[2] += (x - mousex) * 360 / 500; // camera rotate based on mouse movement
  crot[1] -= (y - mousey) * 360 / 500;
  mousex = x; // set the x and y coords for use in calculating movement in next
              // iteration
  mousey = y;
}

void special_key(int key, int UNUSED(x), int UNUSED(y)) {
  switch (key) {
  case GLUT_KEY_DOWN:
    if (n > 10)
      n--;
    reset_simulation();
    break;
  case GLUT_KEY_UP:
    if (n < 200)
      n++;
    reset_simulation();
    break;
  case GLUT_KEY_LEFT:
    if (sep > 0.01)
      sep -= 0.01;
    reset_simulation();
    break;
  case GLUT_KEY_RIGHT:
    if (sep < 2)
      sep += 0.1;
    reset_simulation();
    break;
  case GLUT_KEY_PAGE_UP:
    printf("Increase stiffness\n");
    reset_simulation();
    break;
  case GLUT_KEY_PAGE_DOWN:
    printf("Decrease stiffness\n");
    reset_simulation();
    break;
  default:
    break;
  }
  glutPostRedisplay();
}

void keyboard(unsigned char key, int UNUSED(x), int UNUSED(y)) // key bindings
{

  switch (key) {
  case '1':
    rendermode = 1;
    break;
  case '2':
    rendermode = 2;
    break;
  case '3':
    rendermode = 3;
    break;
  case 32: // space
    paused = !paused;
    break;
  case 'b':
    if (rball > 1)
      rball -= 0.25;
    reset_simulation();
    break;
  case 'B':
    if (rball < 5)
      rball += 0.25;
    reset_simulation();
    break;
  case '[':
    if (offset < rball + 1)
      offset += 0.25;
    reset_simulation();
    break;
  case ']':
    if (offset > 0)
      offset -= 0.25;
    reset_simulation();
    break;
  case 'r':
  case 'R':
    reset_simulation();
    paused = 0;
    break;
  case '-':
    glScalef(0.9, 0.9, 0.9);
    break;
  case '=':
    glScalef(1.2, 1.2, 1.2);
    break;
  case 27: // Escape = Exit
    exit(0);
    break;
  }
  glutPostRedisplay();
}

void reshape(int width,
             int height) // fix up orientations and initial translations
{
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (float)width / height, 2, 1000.0);
  glMatrixMode(GL_MODELVIEW);
}

static void bitmap_string(void *font, float x, float y, char *str) {
  size_t i;

  /* Use the WindowPos extension originally from Mesa so we
     always get window coords without having to set up a
     special ortho projection */
  //  glWindowPos2f(x, y);
  glRasterPos2f(x, y);
  for (i = 0; i < strlen(str); i++)
    glutBitmapCharacter(font, str[i]);
}

void display() // main display routine
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  draw_scene();
  glColor3f(1, 1, 1);
  bitmap_string(GLUT_BITMAP_HELVETICA_18, 64, 32, progress);
  glFlush();
  glutSwapBuffers();
}

void draw_scene() // draw scene
{
  glPushMatrix();

  glLoadIdentity();

  glTranslatef(0.0f, 0.0f, -20.0f);
  glRotatef(-90, 1.0f, 0.0f, 0.0f);
  glRotatef(0, 0.0f, 1.0f, 0.0f);
  glRotatef(90, 0.0f, 0.0f, 1.0f);

  glRotatef(crot[1], 0.0f, 1.0f, 0.0f);
  glRotatef(crot[2], 0.0f, 0.0f, 1.0f);
  glRotatef(crot[0], 1.0f, 0.0f, 0.0f);

  mat_diffuse[0] = 1.0;
  mat_diffuse[1] = 1.0;
  mat_diffuse[2] = 0.6;

  glPushMatrix();
  glColor3f(0, 0, 1);
  glCallList(mainball);
  glColor3f(0.2, 0.7, 0.2);
  switch (rendermode) {
  case 1:
    faceshade();
    break;
  case 2:
    vertexshade();
    break;
  default:
    noshade();
  }
  glPopMatrix();

  glPopMatrix();
}
