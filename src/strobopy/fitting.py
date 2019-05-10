# -*- coding: utf-8 -*-

"""Fitting module."""
import numpy as np
import matplotlib.pyplot as plt
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
import numpy as np
def outliers(image,quart=5, num_stdev=2, get_cutoff=False, dist='Normal',index=False, diff=False):
    """Check if value is an outlier based on standard deviations
    quart and num_stdev determine the qualifiers for the outlier (mean +- 1.5*(inner quartile) and mean +- num_stdev *std)
    get_cutoff=True gets the value of the cutoff as determined by num_stdev and quart
    Use dist='Normal' for normal distributions and dist=any other string if not normal dist
    returns boolean array the same size as image where each element is True if an outlier

    """
    if dist=='Normal':
        """Check if value is an outlier based on normal distributions"""
        m=np.mean(image)
        std=np.std(image)
        cutoff=num_stdev*std
        lower, upper = m-cutoff, m+cutoff
        outs =np.logical_or(image<=lower,image>=upper)
        if index==False:
            if get_cutoff==True:
                return outs, (lower, upper)
            return outs
        if index==True:
            if get_cutoff==True:
                return outs, np.argwhere(outs),(lower, upper)
            return outs, np.argwhere(outs)

    else:
        """Check if value is an outlier based on percentiles"""
        uquart=100-quart
        q25, q75 = np.percentile(image, quart), np.percentile(image, uquart)
        iqr = q75 - q25
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off
        outs =np.logical_or(image<=lower,image>=upper)
        if index==False:
            if get_cutoff==True:
                return outs, (lower, upper)
            return outs
        if index==True:
            if get_cutoff==True:
                return outs, np.argwhere(outs),(lower, upper)
            return outs, np.argwhere(outs)

def local_mean(matrix, y_axis, x_axis, NN=1):
    '''Gets the local mean around a specific index in the matrix
        with NN 1=3x3 -center point = 8 points NN; 2=5x5 -center pt = 24 points!
        Now works with any size by folding around at edges
        Returns the local mean for the specified x and y
        '''
    if (NN+2)**2>=matrix.size:
        raise ValueError("NN is larger than input matrix")
    local_data=[]
    numb=[]
    nsize=matrix.shape[0]-1
    msize=matrix.shape[1]-1
    for n in range(-NN, NN+1,1) :
        for m in range(-NN, NN+1,1):
            if m==0 and n==0:
                continue

            if  y_axis+n>nsize and   x_axis+m>msize:  #Cycles around far corner back to 0,0 for arbitrary size
                k = (y_axis+n)
                k = k % nsize-1
                l = (x_axis+m)
                l = l % msize-1
                local_data.append(matrix[k, l])
                numb.append([k, l])
                continue

            if y_axis+n>nsize:
                k = y_axis+n
                k = k% nsize-1
                local_data.append(matrix[k, x_axis + m])
                numb.append([k,x_axis + m])
                continue
            if   x_axis+m>msize:
                l = (x_axis+m)
                l = l % msize-1
                local_data.append(matrix[y_axis +n, l])
                numb.append([y_axis, l])
                continue

            local_data.append(matrix[y_axis+n,x_axis+m])
            numb.append([y_axis+n,x_axis+m])

    loc_mean=np.mean(local_data)

    return loc_mean

def diff_outliers(image,quart=5,num_stdev=2,dist='Normal',axis='both',index=False):
    from warnings import warn
    ''' Returns an array of shape image with the outliers (based on X and Y derivatives aka slopes)
    of the image of type True. Outliers are designated by outliers function.
    axis determines the direction of derivate and selecting both combines both derivatives
    If index=True, the function returns the index of where the outliers are in the image.
    '''
    if image.ndim !=2:
        raise ValueError("input not 2 dim (n x m)")
    if image.size<25:
        warn('Outliers may not be found since Matrix dimensions are small')
    mat=image
    if axis=='both':
        A0=np.expand_dims(mat[0,:],axis=0)
        A1=np.expand_dims(mat[-1,:],axis=0)
        maty = np.concatenate((mat,A0),axis=0)
        maty1=np.flip(np.concatenate((A1,mat),axis=0),axis=0)
        if dist=='Normal':
            diffy1 = outliers(np.diff(maty,axis=0),num_stdev=num_stdev)
            diffy1a = outliers(np.flip(np.diff(maty1,axis=0),axis=0),num_stdev=num_stdev)
        else:
            diffy1 = outliers(np.diff(maty,axis=0),quart=quart, dist='a')
            diffy1a = outliers(np.flip(np.diff(maty1,axis=0),axis=0),quart=quart,dist='a')

        diffy= np.logical_and(diffy1,diffy1a)

        B0=np.expand_dims(mat[:,0],axis=1)
        B1=np.expand_dims(mat[:,-1],axis=1)
        matx = np.concatenate((mat,B0),axis=1)
        matx1=np.flip(np.concatenate((B1,mat),axis=1),axis=1)
        if dist=='Normal':
            diffx1 = outliers(np.diff(matx,axis=1),num_stdev=num_stdev)
            diffx1a = outliers(np.flip(np.diff(matx1,axis=1),axis=1),num_stdev=num_stdev)
        else:
            diffx1 = outliers(np.diff(matx,axis=1),quart=quart,dist='a')
            diffx1a = outliers(np.flip(np.diff(matx1,axis=1),axis=1),quart=quart,dist='a')
        diffx= np.logical_and(diffx1,diffx1a)


        diff=np.logical_or(diffx,diffy)
        if index==True:
            return np.argwhere(diff)
        return diff

    if axis==0:

        A0=np.expand_dims(mat[0,:],axis=0)
        A1=np.expand_dims(mat[-1,:],axis=0)
        maty = np.concatenate((mat,A0),axis=0)
        maty1=np.flip(np.concatenate((A1,mat),axis=0),axis=0)

        if dist=='Normal':
            diffy1 = outliers(np.diff(maty,axis=0),num_stdev=num_stdev)
            diffy1a = outliers(np.flip(np.diff(maty1,axis=0),axis=0),num_stdev=num_stdev)
        else:
            diffy1 = outliers(np.diff(maty,axis=0),quart=quart, dist='a')
            diffy1a = outliers(np.flip(np.diff(maty1,axis=0),axis=0),quart=quart,dist='a')

        diffy= np.logical_and(diffy1,diffy1a)
        if index==True:
            return np.argwhere(diffy)
        return diffy
    if axis==1:

        B0=np.expand_dims(mat[:,0],axis=1)
        B1=np.expand_dims(mat[:,-1],axis=1)
        matx = np.concatenate((mat,B0),axis=1)
        matx1=np.flip(np.concatenate((B1,mat),axis=1),axis=1)

        if dist=='Normal':
            diffx1 = outliers(np.diff(matx,axis=1),num_stdev=num_stdev)
            diffx1a = outliers(np.flip(np.diff(matx1,axis=1),axis=1),num_stdev=num_stdev)
        else:
            diffx1 = outliers(np.diff(matx,axis=1),quart=quart,dist='a')
            diffx1a = outliers(np.flip(np.diff(matx1,axis=1),axis=1),quart=quart,dist='a')

        diffx= np.logical_and(diffx1,diffx1a)
        if index==True:
            return np.argwhere(diffx)
        return diffx

def remove_outliers(image,NN=1,quart=5,num_stdev=2, get_out=False, dist='Normal',axis='both', num_out=False, diff=True):
    '''Takes an image (numpy array) and replaces outliers (determined by outliers function)
    with the local means determined by the local mean function
    Returns the image with the outliers removed.
    If get_out=True it returns the values of the outliers from the image
    '''
    if diff==True:
        '''Removes outliers based on the slopes using the diff_outlier function
        Returns the original image with the difference outliers replaced by the local mean
        if num_out =True it returns number of outliers
        If get_out=True it returns the values of the outliers from the image

        '''
        outlier_index=diff_outliers(image, quart=quart,num_stdev=num_stdev, dist=dist,axis=axis, index=True)
        outliers_removed=image.copy()
        for [i,j] in outlier_index: # looks at positions where true only - faster than looking at all values
            loc_mean=local_mean(image, i, j, NN)
            outliers_removed[i,j] = loc_mean # replaces all values where true with the local mean
        if num_out==True:
            return outlier_index.shape[0]
        if get_out ==True:
            return outliers_removed, outliers_removed[outlier_index]

        return outliers_removed

    if dist=='Normal':
        if get_out==False:
            outlier_bool= outliers(image, num_stdev=num_stdev,get_cutoff=get_out, dist=dist)
        if get_out==True:
            outlier_bool, cutoffs= outliers(image, num_stdev=num_stdev,get_cutoff=get_out, dist=dist)
        outlier_index=np.argwhere(outlier_bool==True)
        outliers_removed=image.copy()
        for [i,j] in outlier_index: # looks at positions where true only - faster than looking at all values
            loc_mean=local_mean(image, i, j, NN)
            outliers_removed[i,j] = loc_mean # replaces all values where true with the local mean

        if get_out ==True:
            return image[outlier_index]

        return outliers_removed
    else:
        if get_out==False:
            outlier_bool= outliers(image, num_stdev=num_stdev, get_cutoff=get_out, dist=dist)
        if get_out==True:
            outlier_bool, cutoffs= outliers(image, num_stdev=num_stdev,get_cutoff=get_out, dist=dist)
        outlier_index=np.argwhere(outlier_bool==True)
        outliers_removed=image.copy()
        for [i,j] in outlier_index: # looks at positions where true only - faster than looking at all values
            loc_mean=local_mean(image, i, j, NN)
            outliers_removed[i,j] = loc_mean # replaces all values where true with the local mean

        if get_out ==True:
            return image[outlier_index]

        return outliers_removed

def thresh_norm(matrix,thresh=0,norm=1,copy=True):
    '''Throws away values less than 0 and normalizes to 0,1'''
    if copy==True:
        m=matrix.copy()
    else:
        m=matrix
    m[m<thresh]=thresh
    m-=np.min(m)
    m=m/np.max(m)
    m=m*norm
    return m
def thresh_norm_filt(image,thresh=0,norm=1):
    from skimage.restoration import denoise_bilateral
    '''Thresholds image to >0 to get rid of dark current,
    applies a simple denoise filter, and threshes and renormalizes again
    Returns the normalized image
    '''
    normed=thresh_norm(image,thresh=thresh,norm=norm, copy=True)
    filt=denoise_bilateral(normed,multichannel=False)
    norm=thresh_norm(filt,norm=norm,copy=False)
    return norm
def cleaner(image,thresh=0,num_stdev=1,NN=3,norm=1,copy=True):
    '''Combines outlier removal with thresholding and normalizing
    Returns a cleaned image
    '''
    if copy==True:
        m=image.copy()
    else:
        m=image
    out_m=remove_outliers(m, num_stdev=num_stdev,NN=NN, diff=True)
    cleaned_im=thresh_norm_filt(out_m,thresh=thresh,norm=norm)
    return cleaned_im
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
    m=slope_from_points(pts)
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
    theta=np.arctan(m)
    if m<0:
        if x1>x2:
            botx_start=np.rint(x2-np.cos(theta)*width/2)
            topx_start=np.rint(x2+np.cos(theta)*width/2)
            botx_end=np.rint(x1-np.cos(theta)*width/2)
            topx_end=np.rint(x1+np.cos(theta)*width/2)
            
        if x1<x2:
            botx_start=np.rint(x1-np.cos(theta)*width/2)
            topx_start=np.rint(x2+np.cos(theta)*width/2)
            botx_end=np.rint(x1-np.cos(theta)*width/2)
            topx_end=np.rint(x2-np.cos(theta)*width/2)
    
        
        boty_start=np.rint(y1-np.sin(theta)*width/2)
        topy_start=np.rint(y1+np.sin(theta)*width/2)
        boty_end=np.rint(y2-np.cos(theta)*width/2)
        topy_end=np.rint(y2+np.cos(theta)*width/2)
        
        
    if m>0:
        if x1>x2:
            botx_start=np.rint(x1+np.cos(theta)*width/2)
            topx_start=np.rint(x1-np.cos(theta)*width/2)
            botx_end=np.rint(x2+np.cos(theta)*width/2)
            topx_end=np.rint(x2-np.cos(theta)*width/2)
            
        if x1<x2:
            botx_start=np.rint(x1+np.cos(theta)*width/2)
            topx_start=np.rint(x1-np.cos(theta)*width/2)
            botx_end=np.rint(x2+np.cos(theta)*width/2)
            topx_end=np.rint(x2-np.cos(theta)*width/2)
            
        
        boty_start=np.rint(y1-np.sin(theta)*width/2)
        topy_start=np.rint(y1+np.sin(theta)*width/2)
        boty_end=np.rint(y2-np.cos(theta)*width/2)
        topy_end=np.rint(y2+np.cos(theta)*width/2)
        
    if m==0:
        if x1>x2:
            botx_start=np.rint(x2)
            topx_start=np.rint(x2)
            botx_end=np.rint(x1)
            topx_end=np.rint(x1)
        if x1<x2:
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
    return outlist[0],outlist[1]
def linescan(image,pts=None,filt_stdevs=3, mask_stdevs=1, nearest_nb=3,mode='Linear', pad=True,show_im=False, plotline=False,overlay=False, width=100):
    '''
    Fit line to data and project data onto that line
    '''
    masked_im=masker(cleaner(image,num_stdev=filt_stdevs, NN=nearest_nb),stds=mask_stdevs)
    x=positions_nonzero(masked_im,z_values=True)
    x_orig=x[:,0].copy()
    if pts!=None:
        mode='Points'
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
            x_lnrange=np.arange(x_orig.min(),x_orig.max())
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
            x_lnrange=np.arange(x_orig.min(),x_orig.max())
            pad_size=x_lnrange.size*.1
            xleft=np.min(x_lnrange)-pad_size
            xright=np.max(x_lnrange)+pad_size
            if xleft<0:
                xleft=0
            if xright>image.shape[1]:
                xright=image.shape[1]
            x_lnrange=np.arange(xleft,xright)
            xrange=x_lnrange
            if pad==False:
                x_lnrange=np.arange(x_orig.min(),x_orig.max())
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
    