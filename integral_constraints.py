import numpy as np
from matplotlib import pyplot as plt

def obstacle(x,y,W1=1,r=(1,1),c=(0,0)):
    '''
    Define an area that will represent an obstacle
    
    Parameters:
        x (float): x position in space
        y (float): y position in space
        W1 (float): weight of cost
        r (tuple): radius in x and y direction
        c (tuple): center of the ellipse
    '''

    ellipse = ((x - c[0])**2/r[0] + (y - c[1])**2/r[1])**20 + 1

    return W1 / ellipse

def obstacle_dx(x,y,W1=1,r=(1,1,),c=(0,0)):
    '''
    x derivative of the obstacle

    Parameters:
        x (float): x position in space
        y (float): y position in space
        W (float): weight of cost
        r (tuple): radius in x and y direction
        c (tuple): center of the ellipse
    '''

    circle = (x - c[0])**2/r[0] + (y - c[1])**2/r[1]
    numer = -40* W1 * (x-c[0])*(circle)**19
    denom = r[0]*((circle)**20 + 1)**2

    return numer / denom

def obstacle_dy(x,y,W1=1,r=(1,1,),c=(0,0)):
    '''
    y derivative of the obstacle

    Parameters:
        x (float): x position in space
        y (float): y position in space
        W1 (float): weight of cost
        r (tuple): radius in x and y direction
        c (tuple): center of the ellipse
    '''

    circle = (x - c[0])**2/r[0] + (y - c[1])**2/r[1]
    numer = -40 * W1 * (y-c[1])*(circle)**19
    denom = r[1]*((circle)**20 + 1)**2

    return numer / denom

def C(x, y, W1=1):
    '''
    Uses the obstacle function to combine all of our ellipses that we used to make our track
    '''
    return obstacle(x, y, W1, r=(15,10)) - obstacle(x, y, W1,r=(10,5)) + obstacle(x, y, W1) + obstacle(x, y, W1, c=(1,0)) + obstacle(x, y, W1, c=(-1,0)) + obstacle(x, y, W1, c=(.5,0)) + obstacle(x, y, W1, c=(-.5,0))

def C_dx(x, y, W1=1):
    '''
    derivative with respect to x of the C function
    '''
    return obstacle_dx(x, y, W1, r=(15,10)) - obstacle_dx(x, y, W1,r=(10,5)) + obstacle_dx(x, y, W1) + obstacle_dx(x, y, W1, c=(1,0)) + obstacle_dx(x, y, W1, c=(-1,0)) + obstacle_dx(x, y, W1, c=(.5,0)) + obstacle_dx(x, y, W1, c=(-.5,0))

def C_dy(x, y, W1=1):
    ''' 
    derivative with respect to y of the C function 
    '''
    return obstacle_dy(x, y, W1, r=(15,10)) - obstacle_dy(x, y, W1,r=(10,5)) + obstacle_dy(x, y, W1) + obstacle_dy(x, y, W1, c=(1,0)) + obstacle_dy(x, y, W1, c=(-1,0)) + obstacle_dy(x, y, W1, c=(.5,0)) + obstacle_dy(x, y, W1, c=(-.5,0))

def K(delta, vx, vy, lmb = 20, cushion=.1, L = .05, M = 3.):
    '''Integral constraint due to centripetal acceleration'''
    return (cushion / (delta - np.arctan(L*M/(vx**2 + vy**2)))) ** lmb

def K_dvx(delta, vx, vy, lmb=20, cushion=.1, L=.05, M=3.):
    ''' derivative of K with respect to vx'''
    v2 = vx**2 + vy**2
    lm_v2 = L*M/(v2)
    numerator = -2*cushion*lmb*L*M*vx*(cushion/(delta-np.arctan(lm_v2)))**(lmb-1)
    denominator = (v2**2)*(lm_v2**2 + 1)*(delta - np.arctan(lm_v2))**2
    return numerator / denominator

def K_dvy(delta, vx, vy, lmb=20, cushion=.1, L=.05, M=3.):
    ''' derivative of K with respect to vy'''
    v2 = vx**2 + vy**2
    lm_v2 = L*M/(v2)
    numerator = -2*cushion*lmb*L*M*vy*(cushion/(delta-np.arctan(lm_v2)))**(lmb-1)
    denominator = (v2**2)*(lm_v2**2 + 1)*(delta - np.arctan(lm_v2))**2
    return numerator / denominator

def K_ddelta(delta, vx, vy, lmb=20, cushion=.1, L=.5, M=3.):
    ''' derivative of K with respect to delta '''
    return -lmb*(cushion / (delta - np.arctan(L*M/(vx**2 + vy**2))))**(lmb+1) / cushion

def plot_track():
    X,Y = np.meshgrid(np.linspace(-5,5,600),np.linspace(-5,5,600))
    Z = C(X,Y, W1=3)

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X,Y,Z,edgecolor=None,linewidth=0)
    ax.set_zlim(0,3)
    ax.view_init(elev=84,azim=90)

    ax2 = fig.add_subplot(122)
    ax2.contour(X, Y, Z)
    ax2.set_xbound([-4,4])
    ax2.set_ybound([-7,7])

    plt.axis('off')
    plt.show()

plot_track()