# -*- coding: utf-8 -*-

"""Main module."""
import numpy as np

def mask_maker(image,stds=3):
    '''Returns a boolean mask for numbers inside of the mean +- stds * std'''
    bool_mask=~(image>np.mean(image)+stds*np.std(image))|(image<np.mean(image)-stds*np.std(image))
    return bool_mask

def masker(image, mask=None, stds=None, copy=True):
    '''Returns a masked numpy array'''
    im=image
    if copy==True:
        im=image.copy()

    if (stds!=0) & (stds!=None):
        m=mask_maker(im,stds=stds)
        a=np.ma.masked_where(m,image)
        return a
    if (stds==0):
        return image
    
    if (mask!=None).any():
        a=np.ma.masked_where(mask,im)
        return a
    
    return image

def positions_nonzero(masked_image,z_values=False):
    '''Return positions where the mask is not'''
    y,x=np.ma.nonzero(masked_image)
    if z_values==True:
        z=masked_image[y.astype(np.int),x.astype(np.int)]
        mim_coordinates=np.concatenate((x[:,np.newaxis],y[:,np.newaxis],z[:,np.newaxis]),axis=1)
        return mim_coordinates
    return y,x

def fit_linear(XY, R2=False):
    from sklearn.linear_model import LinearRegression
    X,y=XY[:,0].reshape(-1, 1),XY[:,1].reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    params=reg.coef_,reg.intercept_
    if R2==True:
        params.append(reg.score)
    return [params[0][0][0],params[1][0]]

def slope_proj(m):
    '''
    Gets 2 perpendicular vectors from a slope
    Returned as a Matrix [[x1,y1],[x2,y2]]
    '''
    v1=np.array([1,m])/(1**2+m**2)**.5
    mp=-1/m
    v2=np.array([1,mp])/(1**2+mp**2)**.5
    M=np.array((v1,v2))
    return M
def slope_proj_inv(M):
    m=M[0,1]/M[0,0]
    return m

def position_means(masked_image):
    '''Returns the mean y,x positions'''
    b,a=positions_nonzero(masked_image)
    a_m=np.nanmean(a)
    b_m=np.nanmean(b)
    return b_m,a_m

def coord_to_im(xyz,shape='square'):
    '''Return image from xyz coordinates as an nxm matrix'''
    z=xyz[:,2]
    if shape=='square':
        l=int(xyz.shape[0]**.5)
        image=z.reshape(l,l).T
        return image

    y,x=shape
    image=z.reshape(y,x)
        
    return image

def im_to_coord(image):
    '''Return image xyz coordinates as an X*Yx3 matrix'''
    n,m=image.shape
    yy,xx=np.meshgrid(np.arange(n),np.arange(m))
    x=xx.flatten()
    y=yy.flatten()
    z=image[y.astype(np.int),x.astype(np.int)]
    im_coordinates=np.concatenate((x[:,np.newaxis],y[:,np.newaxis],z[:,np.newaxis]),axis=1)
    return im_coordinates
def make_odd(N):
    X=int(N)
    if X%2:
        return N
    Y=X+1
    return Y
def make_even(N):
    X=int(N)
    if X%2:
        Y=X+1
        return Y
    return N

def shift_image(im,shift,delete=True,fill=0):
    from scipy.ndimage import fourier_shift
    """Shifts image based on scipy.ndimage shift vector.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    shift : sequence
        The size of the box used for filtering.
        If a float, `shift` is the same for all axes. If a sequence, `shift`
        has to contain one value for each axis.

    Returns
    -------
    fourier_shift : ndarray
        The shifted input.

    """
    
    rank=im.ndim
    if hasattr(shift, '__iter__'):
        n_shift = list(shift)
    else:
        n_shift = [shift] * rank  
    fft_im=fourier_shift(np.fft.fftn(im),n_shift)
    shifted_im=np.fft.ifftn(fft_im)
#     return shifted_im,n_shift
    if delete ==True:
        ylim=int(n_shift[0])
        xlim=int(n_shift[1])
        if ylim>=0:
            shifted_im[0:ylim,:]=fill
        if ylim<0:
            shifted_im[ylim:,:]=fill
        if xlim>=0:
            shifted_im[:,0:xlim]=fill
        if xlim<0:
            shifted_im[:,xlim:]=fill
        
    return np.array(shifted_im)

def to_8bit(image):
    im=image.copy()
    im=(im-np.min(im)).real
    im=(im/np.max(im))*255

    return im.astype(int)
def to_binary(image):
    im=image.copy()
    mean=np.mean(im)
    im[im<=mean]=0
    im[im>mean]=1
    return im.astype(int)
def thresh(image,threshtype='local', make_binary=True, simple=False):
    im=image.copy()
    if simple==True:
        im[im<0]=0
        return im

    if threshtype=='global':
        from skimage.filters import threshold_otsu
        try:
            thresh = threshold_otsu(im) 
        except ValueError:
            thresh=0
        binary= im>thresh
        
        if make_binary==False:
            return binary
        return binary.astype(int)
            
    if threshtype=='local':
        from skimage.filters import threshold_sauvola
        thresh = threshold_sauvola(im)

    binary = im< thresh
    if make_binary==False:
        return binary
    return binary.astype(int)
def gaussian_filter(image,sig=3):
    from skimage.filters import gaussian
    im=image.copy()
    g = gaussian(im, sigma=sig, preserve_range=True)
    return g
def binarize(image,sig=3,threshtype='local'):
    im=image.copy()
    t=thresh(im,threshtype=threshtype)
    filt=gaussian_filter(t,sig)
    binary=thresh(filt,threshtype='global').astype(int)
    return binary.astype(int)

def rigid_registration(to_be_registered,reference,pixel_resolution,get_shift=False):
    from skimage.feature import register_translation
    to_be_registered_bin=binarize(to_be_registered)
    reference_bin=binarize(reference)
    upsample_factor=1/pixel_resolution
    shift, error, diffphase = register_translation(reference_bin,to_be_registered_bin, upsample_factor)
    shifted_image=shift_image(to_be_registered,shift)
    if get_shift==True:
        return shifted_image.real, shift
    return shifted_image.real



def spot_linescan(image,pts=None,filt_stdevs=3, mask_stdevs=1, nearest_nb=3,mode='Linear', pad=True,show_im=False, plotline=False,overlay=False):
    '''
    Fit line to data and project data onto that line
    '''
    masked_im=masker(cleaner(image,num_stdev=filt_stdevs, NN=nearest_nb),stds=mask_stdevs)
    x=positions_nonzero(masked_im,z_values=True)
    x_orig=x[:,0].copy()
    if mode in ['Linear','PCA']:
        
        if mode=='Linear':
            XY=x[:,:2].copy()
            line=fit_linear(XY)
            directions=slope_proj(line[0])
        if mode=='PCA':
            pca=PCA(n_components=2)
            a=pca.fit(x[:,:2])
            directions=a.components_
        
        x[:,:2]=np.dot(x[:,:2],directions.T)
        ids=x[:,0].astype(np.int)-np.min(x[:,0].astype(np.int)) ## Has to be positive
        data=x[:,2]
        bins=np.bincount(ids,weights=data)
        y=bins[bins!=0]
        Y=y/np.unique(ids,return_counts=True)[1] ## Normalized to get mean rather than sum
        
        x_lnrange=np.arange(x_orig.min(),x_orig.max())
        pad_size=x_lnrange.size*.1
        xleft=np.min(x_lnrange)-pad_size
        xright=np.max(x_lnrange)+pad_size
        if xleft<0:
            xleft=0
        if xright>image.shape[1]:
            xright=image.shape[1]
        x_lnrange=np.arange(xleft,xright)
        if pad==False:
            x_lnrange=x_lnrange=np.arange(x_orig.min(),x_orig.max())
#         return x_lnrange
    if show_im==True:
       
        if plotline==False:
            
            plt.rcParams['figure.figsize'] = (16,9)
            if overlay==True:
                plt.imshow(image,alpha=.9)
            plt.imshow(masked_im)
            n,m=image.shape
            xrange=x_lnrange
            if mode=='Linear':
                yrange=line[0]*xrange+line[1]
            if mode=='PCA':
                y_m,x_m=position_means(masked_im)
                yrange=slope_proj_inv(directions)*(xrange-x_m)+y_m
                
            if mode=='Points':
                '''pts in form [[x1,y1],[x2,y2]]   '''
                pts=np.asarray(pts)
                n,m=image.shape
                xmin,xmax=np.min(pts[:,0]),np.max(pts[:,0])
                xrange=np.arange(xmin,xmax)
                if pts[0,1]<pts[1,1]:
                    yrange=np.linspace(pts[0,1],pts[1,1],xrange.size)
                else:
                    yrange=np.linspace(pts[1,1],pts[0,1],xrange.size)[::-1]
            plt.plot(xrange,yrange,color='r')
        if plotline==True:
            plt.rcParams['figure.figsize'] = (16,9)
            fig = plt.figure()
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
            ax1.imshow(masked_im)
            if overlay==True:
                ax1.imshow(image,alpha=.9)
            n,m=image.shape
            xrange=x_lnrange
            if mode=='Linear':
                yrange=line[0]*xrange+line[1]
            if mode=='PCA':
                y_m,x_m=position_means(masked_im)
                yrange=slope_proj_inv(directions)*(xrange-x_m)+y_m
                
            if mode=='Points':
                '''pts in form [[x1,y1],[x2,y2]]   '''
                plt.rcParams['figure.figsize'] = (16,9)
                pts=np.asarray(pts)
                n,m=image.shape
                xmin,xmax=np.min(pts[:,0]),np.max(pts[:,0])
                xrange=np.arange(xmin,xmax)
                if pts[0,1]<pts[1,1]:
                    yrange=np.linspace(pts[0,1],pts[1,1],xrange.size)
                else:
                    yrange=np.linspace(pts[1,1],pts[0,1],xrange.size)[::-1]
                Y=image[yrange.astype(np.int),xrange.astype(np.int)]
                Y=Y/np.max(Y)

#             yrange=line[0]*xrange+line[1]
            plt.rcParams['figure.figsize'] = (16,9)
            ax1.plot(xrange,yrange,color='r')
            if pad==True:
                pad_size=int(Y.size*.1)
                ax2.plot(np.pad(Y,pad_size,'edge'))
            if pad==False:
                ax2.plot(Y)
        return

    if plotline==True:
        plt.rcParams['figure.figsize'] = (16,9)
        if pad==True:
            pad_size=int(Y.size*.1)
            plt.plot(np.pad(Y,pad_size,'edge'))
        if pad==False:
            plt.plot(Y)
        return
    if pad==False:
        return Y
    pad_size=int(Y.size*.1)
    return np.pad(Y,pad_size,'edge')
    
def slope_from_points(pts):
    M=np.array(pts)
    if (M[0,0]-M[1,0])==0:
        return np.Inf
    m=(M[0,1]-M[1,1])/(M[0,0]-M[1,0])

    return m
def perp_slope_from_points(pts):
    m=slope_from_points(pts)
    if m==0:
        return np.Inf
    return -1/m

def get_line_params(pts):
    pts=np.array(pts)
    m=slope_from_points(pts)
    if m==np.Inf:
        return m, np.nan
    b=m*pts[0,0]-pts[0,1]
    return m,b
    
def get_line_width_coords(pts, width):
    '''Gets coordinates (x,y) coordinates for 4 positions based on width given
       pts is input as [(x1,y1),(x2,y2)]
       returns start, end 
       start = [bottom_start,top_start]
       bottom_start=(x,y)
       top_start=(x,y)
       similar for end
    '''
    pts=np.array(pts)
    m,b=get_line_params(pts)
    x1,x2=pts[0,0],pts[1,0]
    y1,y2=pts[0,1],pts[1,1]
    if x2<x1:
        xt=x2.copy()
        x2=x1.copy()
        x1=xt.copy()
        yt=y2.copy()
        y2=y1.copy()
        y1=yt.copy()
#     return x1,x2, y1,y2
    
    theta=np.arctan(m)
#     return np.cos(theta), np.sin(theta)
    if m<0:
            
        botx_start=np.rint(x1+np.sin(theta)*width/2)
        topx_start=np.rint(x1-np.sin(theta)*width/2)
        botx_end=np.rint(x2+np.sin(theta)*width/2)
        topx_end=np.rint(x2-np.sin(theta)*width/2)

        
        boty_start=np.rint(y1-np.cos(theta)*width/2)
        topy_start=np.rint(y1+np.cos(theta)*width/2)
        boty_end=np.rint(y2-np.cos(theta)*width/2)
        topy_end=np.rint(y2+np.cos(theta)*width/2)
        
        
    if m>0:

        botx_start=np.rint(x1+np.sin(theta)*width/2)
        topx_start=np.rint(x1-np.sin(theta)*width/2)
        botx_end=np.rint(x2+np.sin(theta)*width/2)
        topx_end=np.rint(x2-np.sin(theta)*width/2)
            
        
        boty_start=np.rint(y1-np.cos(theta)*width/2)
        topy_start=np.rint(y1+np.cos(theta)*width/2)
        boty_end=np.rint(y2-np.cos(theta)*width/2)
        topy_end=np.rint(y2+np.cos(theta)*width/2)
        
    if m==0:
        botx_start=np.rint(x1)
        topx_start=np.rint(x1)
        botx_end=np.rint(x2)
        topx_end=np.rint(x2)

        boty_start=np.rint(y1-width/2)
        topy_start=np.rint(y1+width/2)
        boty_end=np.rint(y2-width/2)
        topy_end=np.rint(y2+width/2)
              
    if m==np.inf:
        if y1>y2:
            boty_start=np.rint(y2)
            topy_start=np.rint(y2)
            boty_end=np.rint(y1)
            topy_end=np.rint(y1)
        if y1<y2:
            boty_start=np.rint(y1)
            topy_start=np.rint(y1)
            boty_end=np.rint(y2)
            topy_end=np.rint(y2)
            
        botx_start=np.rint(x1+width/2)
        topx_start=np.rint(x1-width/2)
        botx_end=np.rint(x2+width/2)
        topx_end=np.rint(x2-width/2)
#     x=np.array([botx_start,topx_start,topx_end,botx_end])
#     y=np.array([boty_start,topy_start,topy_end,boty_end])
#     return x,y
#     return m
    c1=(botx_start.astype(int),boty_start.astype(int))
    c2=(topx_start.astype(int),topy_start.astype(int))
    start=[c1,c2]
    start.sort(key=lambda x:x[1])
    c3=(botx_end.astype(int),boty_end.astype(int))
    c4=(topx_end.astype(int),topy_end.astype(int))
    end=[c3,c4]
    end.sort(key=lambda x:x[1])
    outlist=[start,end]
    outlist.sort(key=lambda x:x[0][0])
    return [outlist[0][0],outlist[0][1],outlist[1][1],outlist[1][0]]

def connectpoints(x,y,p1,p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    plt.plot([x1,x2],[y1,y2],'r-',linewidth=4)
def draw_box(im,corners,binary=True):
    '''Draw a box on the image with the given corners
    corners like [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]'''
    if binary==False:
        plt.imshow(im)
    else:
        plt.imshow(binarize(im))
    pts=np.asarray(corners)
    x=pts[:,0]
    y=pts[:,1]
    x=np.append(x,pts[0,0])
    y=np.append(y,pts[0,1])
    for i in np.arange(0,len(x)-1):
        connectpoints(x,y,i,i+1)
    return plt.gcf()
def linescan(im,pts,width,thresh_type=None, show=False, show_bin=False):
    from skimage.transform import rotate
    """Gives a summed linescan between pts along the perpendicular direction with the number of points given by width

    Parameters
    ----------
    im : np.array
      Shape N X M.

    pts : [[x1,y1],[x2,y2]]
      Coordinates between points you want to draw the line
      
    width : int 
      width of summed response

    Returns
    -------
    1-D numpy array of intensity values between pts
    and summed along the perpendicular direction number of points give by width

    """
    pts=np.asarray(pts)
    m,b=get_line_params(pts)
    theta=np.arctan(m)
    x1,x2=pts[0,0],pts[1,0]
    y1,y2=pts[0,1],pts[1,1]
    if x1<x2:
        centx=x1.astype(int)
        centy=y1.astype(int)
    if x1>x2:
        centx=x2.astype(int)
        centy=y2.astype(int)

    
    d=np.rint(((x1-x2)**2+(y1-y2)**2)**.5).astype(int)
    
    image=im.copy()
    if thresh_type=='binarize':
        image=binarize(image)
    if thresh_type=='loc_thresh':
        image=thresh(image)
    if thresh_type=='glob_thresh':
        image=thresh(image, thresh_type='global')
#     print(m)
    if (m!=0) & (m!=np.inf):
        image=rotate(image,theta*180/np.pi,center=(centx,centy),mode='constant',cval=np.nan, order=0,preserve_range=True)
        ymin=np.rint(centy-width/2).astype(int)
        ymax=np.rint(centy+width/2).astype(int)
        xmin=centx
        xmax=int(xmin+d)
        
        Y=image[ymin:ymax+1,xmin:xmax+1]
        if x1>x2:
            Y=np.nansum(Y,axis=0)[::-1]
            if show==True:
                pts=np.asarray(get_line_width_coords([[x1,y1],[x2,y2]],width))
                draw_box(im,pts, binary=show_bin)
        if x2>x1:
            Y=np.nansum(Y,axis=0)
            if show==True:
                pts=np.asarray(get_line_width_coords([[x1,y1],[x2,y2]],width))
                draw_box(im,pts, binary=show_bin)
        return Y

#         return ymin,ymax,xmin,xmax
#         return image
    elif m==0:
        ymin,ymax=(np.rint(y1-width/2)).astype(int),(np.rint(y1+width/2)).astype(int)
        if x1<x2:
            xmin,xmax=x1.astype(int),x2.astype(int)
        if x2<x1:
            xmin,xmax=x2.astype(int),x1.astype(int)
        Y=image[ymin:ymax+1,xmin:xmax+1]
        Y=np.nansum(Y,axis=0)
        if show==True:
            pts=np.asarray(get_line_width_coords([[x1,y1],[x2,y2]],width))
            draw_box(im,pts, binary=show_bin)
        return Y

    elif m==np.inf:
        xmin=np.rint(x1-width/2).astype(int)
        xmax=np.rint(x1+width/2).astype(int)
        if y1<y2:
            ymin,ymax=y2.astype(int),y1.astype(int)
        if y1<y2:
            ymin,ymax=y1.astype(int),y2.astype(int)
        Y=image[ymin:ymax+1,xmin:xmax+1]
        Y=np.nansum(Y,axis=1)
        if show==True:
            pts=np.asarray(get_line_width_coords([[x1,y1],[x2,y2]],width))
            draw_box(im,pts, binary=show_bin)
        return Y
    
            
def image_correlation(image1, image2):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    im1=im_to_coord(image1)
    im2=im_to_coord(image2)
    z1=im1[:,2]
    z2=im2[:,2]
    mu_z1 = z1.mean()
    mu_z2 = z2.mean()
    n = z1.shape[0]
    s_z1 = z1.std(0, ddof=n - 1)
    s_z2 = z2.std(0, ddof=n - 1)
    cov = np.dot(z1,
                 z2.T) - n * np.dot(mu_z1,
                                  mu_z2)
    return cov / np.dot(s_z1, s_z2)