# -*- coding:utf-8 -*-
# @Time   : 2021/6/4 10:43 
# @Author : Leo Li
# @Email  : 846016122@qq.com

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import math
import cv2
from PIL import Image as IMG
from tqdm import tqdm
np.set_printoptions(threshold=1000000)

class NLM:
    def __init__(self):
        #Load image
        self.img_path = "noisy_image2.jpg"
        image = IMG.open(self.img_path)

        image = image.convert('L')#转灰度图
        self.image = np.array(image) 

        self.results = []
        self.method_noise = []
        self.method_names = []

    def GaussianTemplate(self, kernel_size, sigma):
        """
        @description  : Given a gaussian distribution, this function can generate a kernel_size * kernel_size matrix. 
        @param  :  kernel_size = size of matrix
        @param  :  sigma = sigma of the gaussian distribution
        @Returns  : the matrix
        """
        '''
        Formula H_{i,j}=\frac{1}{2\pi \sigma^2}e^{-\frac{(i-k-1)^2+(j-k-1)^2}{2\sigma^2}}
        H_ij = 1/(2*pi*sigma^2)*exp(-((i-k-1)^2+(j-k-1)^2)/(2*sigma^2))
                k1                                            k2
        kernelsize = 2*k+1
        '''

        template = np.ones(shape = (kernel_size, kernel_size), dtype="float64", )

        k = int(kernel_size / 2)
        k1 = 1/(2*np.pi*sigma*sigma)
        k2 = 2*sigma*sigma

        SUM = 0.0
        for i in range(kernel_size):
            for j in range(kernel_size):
                template[i,j] = k1*np.exp(-((i-k)*(i-k)+(j-k)*(j-k)) / k2)
                SUM += template[i,j]

        for i in range(kernel_size):
            for j in range(kernel_size):
                template[i,j] /= SUM
        #print("Gaussian Template = \n", template)
        return template

    def Gaussian_Filtering(self, src, dst = [], kernel_size = 3, sigma=0.8):
        """
        @description  : Given a image, return the gaussian filter result.
        @param  : src : source image
        @param  : dst: destination/ result image
        @param  : kernel_size: the conv kernel size.
        @param  : sigma: sigma of gaussian distribution
        @Returns  : Image that has been gaussian filtered
        """
        print("Gaussian Filtering start. Kernel size = {}, sigma = {}".format(kernel_size,sigma))
        start_time = time.time()
        if kernel_size == 1:
            dst = src
            return dst
        elif kernel_size%2 == 0 or kernel_size <= 0:
            print("卷积核大小必须是大于1的奇数")
            return -1

        padding_size = int((kernel_size - 1) / 2)
        img_width = src.shape[0] + padding_size*2
        img_height = src.shape[1] + padding_size*2

        tmp = np.zeros((img_width,img_height))
        dst = np.zeros((src.shape[0],src.shape[1]))
        #padding
        for i in range(padding_size,img_width-padding_size):
            for j in range(padding_size,img_height-padding_size):
                tmp[i,j] = src[i-padding_size,j-padding_size]
        kernel = self.GaussianTemplate(kernel_size, sigma)
        #Gaussian Filtering
        for row in range(padding_size, img_width-padding_size):
            for col in range(padding_size, img_height-padding_size):
                #sum = [0.0,0.0,0.0] #3Channel
                sum = 0
                for i in range(-padding_size,padding_size+1):
                    for j in range(-padding_size,padding_size+1):
                        #for channel in range(0,3):
                            #sum[channel] += tmp[(row+i),(col+j),channel] * kernel[i+padding_size,j+padding_size]
                        sum += tmp[(row+i),(col+j)] * kernel[i+padding_size,j+padding_size]
                # for channel in range(3):
                #     if sum[channel] > 255:
                #         sum[channel] = 255
                #     if sum[channel] < 0:
                #         sum[channel] = 0
                #     dst[row - padding_size, col - padding_size,channel] = sum[channel]
                if sum > 255:
                    sum = 255
                if sum < 0:
                    sum = 0
                dst[row - padding_size, col - padding_size] = sum
        dst = dst.astype('int32')
        end_time = time.time()

        print('Gaussian Filtering Complete. Time:{}'.format(end_time-start_time))
        method_noise = src - dst
        #Visualization
        # plt.subplot(1,3,1)
        # plt.imshow(src,cmap="gray")
        # plt.title("Source Image")
        # plt.subplot(1,3,2)
        # plt.imshow(dst,cmap = "gray")
        # plt.title("Gaussian Filtering")
        # plt.subplot(1, 3, 3)
        # plt.imshow(method_noise, cmap="gray")
        # plt.title("Gaussian Filtering")
        # plt.show()
        self.results.append(dst)
        self.method_noise.append(method_noise)
        self.method_names.append("Gaussian Filtering")

        return dst, method_noise

    def Anisotropic_Filtering(self, src, dst=[], iterations = 10, k = 15, _lambda = 0.25):
        """
        @description  : Anisotropic filtering
        @param  : src : source image
        @param  : dst: destination/ result image
        @Returns  : Anisotropic filtering result
        """
        print("Anisotropic Filtering start. iterations = {}, k = {}, lambda = {}".format(iterations,k,_lambda))
        start_time = time.time()
        image_width = src.shape[0]
        image_height = src.shape[1]
        k2 = 1.0*k*k                                  # Since we only need k^2
        old_dst = src.copy().astype("float64")
        new_dst = src.copy().astype("float64")


        for i in range(iterations):
            for row in range(1,image_width-1):
                for col in range(1,image_height-1):
                    # for channel in range(0,3):
                    #     N_grad = int(old_dst[row-1,col,channel]) - int(old_dst[row,col,channel])
                    #     S_grad = int(old_dst[row+1,col,channel]) - int(old_dst[row,col,channel])
                    #     E_grad = int(old_dst[row,col-1,channel]) - int(old_dst[row,col,channel])
                    #     W_grad = int(old_dst[row,col+1,channel]) - int(old_dst[row,col,channel])
                    #     N_c = np.exp(-N_grad*N_grad/k2)
                    #     S_c = np.exp(-S_grad*S_grad/k2)
                    #     E_c = np.exp(-E_grad*E_grad/k2)
                    #     W_c = np.exp(-W_grad*W_grad/k2)
                    #     new_dst[row,col,channel] = old_dst[row,col,channel] + int(_lambda *(N_grad*N_c + S_grad*S_c + E_grad*E_c + W_grad*W_c))
                    N_grad = old_dst[row-1,col] - old_dst[row,col]
                    S_grad = old_dst[row+1,col] - old_dst[row,col]
                    E_grad = old_dst[row,col-1] - old_dst[row,col]
                    W_grad = old_dst[row,col+1] - old_dst[row,col]
                    N_c = np.exp(-N_grad*N_grad/k2)
                    S_c = np.exp(-S_grad*S_grad/k2)
                    E_c = np.exp(-E_grad*E_grad/k2)
                    W_c = np.exp(-W_grad*W_grad/k2)
                    new_dst[row,col] = old_dst[row,col] + _lambda *(N_grad*N_c + S_grad*S_c + E_grad*E_c + W_grad*W_c)
            old_dst = new_dst

        dst = new_dst#.astype("uint8")
        end_time = time.time()
        print("Anisotropic filtering complete. Time:{}".format(end_time-start_time))
        method_noise = src - dst
        #Visualization
        # plt.subplot(1, 3, 1)
        # plt.imshow(src,cmap='gray')
        # plt.title("Source Image")
        # plt.subplot(1, 3, 2)
        # plt.imshow(dst,cmap='gray')
        # plt.title("Anisotropic Filtering")
        # plt.subplot(1, 3, 3)
        # plt.imshow(method_noise,cmap='gray')
        # plt.title("Method Noise")
        # plt.show()

        self.results.append(dst)
        self.method_noise.append(method_noise)
        self.method_names.append("Anisotropic Filtering")

        return dst,method_noise
    def Total_Variation_Minimization(self, src, dst=[], iterations = 100, _lambda = 0.03):
        """
        @description  : Total variation minimization
        @param  : src: source image
        @param  : dst: destination/ result image
        @Returns  : Total variation minimization result
        """
        print("Total Variation Minimization start. iterations = {}, lambda = {}".format(iterations,_lambda))
        start_time = time.time()
        image_width = src.shape[0]
        image_height = src.shape[1]
        dst = src.copy()
        u0 = src.copy()
        h = 1
        Energy = []
        cnt= 0
        for i in range(0,iterations):
            for row in range(1,image_width-1):
                for col in range(1,image_height-1):
                    # for channel in range(3):
                    #     ux = (float(dst[row+1,col,channel]) - float(dst[row,col,channel]))/h
                    #     uy = (float(dst[row,col+1,channel]) - float(dst[row,col-1,channel]))/(2*h)
                    #     grad_u = math.sqrt(ux*ux+uy*uy)
                    #     c1 = 0
                    #     if grad_u == 0:
                    #         cnt += 1
                    #     else:
                    #         c1 = 1/grad_u
                    #
                    #     ux = (float(dst[row, col, channel]) - float(dst[row-1, col, channel])) / h
                    #     uy = (float(dst[row-1, col + 1, channel]) - float(dst[row-1, col - 1, channel])) / (2 * h)
                    #     grad_u = math.sqrt(ux * ux + uy * uy)
                    #     c2 = 0
                    #     if grad_u == 0:
                    #         cnt += 1
                    #     else:
                    #         c2 = 1 / grad_u
                    #
                    #     ux = (float(dst[row + 1, col, channel]) - float(dst[row-1, col, channel])) / (2 * h)
                    #     uy = (float(dst[row, col + 1, channel]) - float(dst[row, col, channel])) / h
                    #     grad_u = math.sqrt(ux * ux + uy * uy)
                    #     c3 = 0
                    #     if grad_u == 0:
                    #         cnt += 1
                    #     else:
                    #         c3 = 1 / grad_u
                    #
                    #     ux = (float(dst[row + 1, col-1, channel]) - float(dst[row-1, col-1, channel])) / (2 * h)
                    #     uy = (float(dst[row, col, channel]) - float(dst[row, col - 1, channel])) / h
                    #     grad_u = math.sqrt(ux * ux + uy * uy)
                    #     c4 = 0
                    #     if grad_u == 0:
                    #         cnt += 1
                    #     else:
                    #         c4 = 1 / grad_u
                    #
                    #     dst[row,col,channel] = (u0[row,col,channel] + (1/(_lambda*h*h)) * (c1*dst[row+1,col,channel] + c2*dst[row-1,col,channel] + c3*dst[row,col+1,channel] + c4*dst[row,col-1,channel]) ) * (1/(1+(1/(_lambda*h*h)*(c1+c2+c3+c4))))
                    ux = (float(dst[row + 1, col]) - float(dst[row, col])) / h
                    uy = (float(dst[row, col + 1]) - float(dst[row, col - 1])) / (2 * h)
                    grad_u = math.sqrt(ux * ux + uy * uy)
                    c1 = 0
                    if grad_u == 0:
                        cnt += 1
                    else:
                        c1 = 1 / grad_u

                    ux = (float(dst[row, col]) - float(dst[row - 1, col])) / h
                    uy = (float(dst[row - 1, col + 1]) - float(dst[row - 1, col - 1])) / (2 * h)
                    grad_u = math.sqrt(ux * ux + uy * uy)
                    c2 = 0
                    if grad_u == 0:
                        cnt += 1
                    else:
                        c2 = 1 / grad_u

                    ux = (float(dst[row + 1, col]) - float(dst[row - 1, col])) / (2 * h)
                    uy = (float(dst[row, col + 1]) - float(dst[row, col])) / h
                    grad_u = math.sqrt(ux * ux + uy * uy)
                    c3 = 0
                    if grad_u == 0:
                        cnt += 1
                    else:
                        c3 = 1 / grad_u

                    ux = (float(dst[row + 1, col - 1]) - float(dst[row - 1, col - 1])) / (2 * h)
                    uy = (float(dst[row, col]) - float(dst[row, col - 1])) / h
                    grad_u = math.sqrt(ux * ux + uy * uy)
                    c4 = 0
                    if grad_u == 0:
                        cnt += 1
                    else:
                        c4 = 1 / grad_u

                    dst[row, col] = (u0[row, col] + (1 / (_lambda * h * h)) * (
                                c1 * dst[row + 1, col] + c2 * dst[row - 1, col] + c3 * dst[
                            row, col + 1] + c4 * dst[row, col - 1])) * (
                                                         1 / (1 + (1 / (_lambda * h * h) * (c1 + c2 + c3 + c4))))
            # 处理边缘
            for row in range(1,image_width-1):
                dst[row,0] = dst[row,1]
                dst[row,image_height-1] = dst[row,image_height-1-1]
            for col in range(1,image_height-1):
                dst[0,col] = dst[1,col]
                dst[image_width-1,col] = dst[image_width-1-1,col]

            dst[0,0] = dst[1,1]
            dst[0,image_height-1] = dst[1,image_height-1-1]
            dst[image_width-1,0] = dst[image_width-1-1,1]
            dst[image_width-1,image_height-1] = dst[image_width-1-1,image_height-1-1]

            energy = 0
            for row in range(1, image_width - 1):
                for col in range(1, image_height - 1):
                    # for channel in range(3):
                    #     ux = (float(dst[row+1,col,channel]) - float(dst[row,col,channel]))/h
                    #     uy = (float(dst[row,col+1,channel]) - float(dst[row,col,channel]))/h
                    #     tmp = (float(u0[row,col,channel]) - float(dst[row,col,channel]))
                    #     fid = tmp*tmp
                    #     energy += math.sqrt(ux*ux + uy*uy) + _lambda*fid
                    ux = (float(dst[row+1,col]) - float(dst[row,col]))/h
                    uy = (float(dst[row,col+1]) - float(dst[row,col]))/h
                    tmp = (float(u0[row,col]) - float(dst[row,col]))
                    fid = tmp*tmp
                    energy += math.sqrt(ux*ux + uy*uy) + _lambda*fid
            Energy.append(energy)
        end_time = time.time()
        print('Total Variation Minimization Complete. Time:{}'.format((end_time - start_time)))
        method_noise = src - dst
        
        self.results.append(dst)
        self.method_noise.append(method_noise)
        self.method_names.append("Total Variation Minimization")

        return dst,method_noise

    def Yaroslavsky_Filtering(self,src,dst=[],kernel_size=3,h=1):
        """
        @description  : Yaroslavsky filtering
        @param  : src : source image
        @param  : dst : destination/ result image
        @Returns  : Yaroslavsky filter result
        """
        print("Yaroslavsky Filtering start. Kernel size = {}, h = {}".format(kernel_size, h))
        start_time = time.time()
        image_width = src.shape[0]
        image_height = src.shape[1]
        weight = np.zeros(src.shape).astype('float64')
        dst = np.zeros(src.shape).astype('float64')
        padding_size = int((kernel_size - 1) / 2)
        padded_img = np.pad(src, padding_size, 'symmetric').astype('float64')
        for row in range(0, image_width):
            for col in range(0, image_height):
                sum = 0
                for i in range(-padding_size, padding_size + 1):
                    for j in range(-padding_size, padding_size + 1):
                        if i == 0 and j == 0:
                            continue
                        sum += np.exp(-(padded_img[(row + i), (col + j)] - padded_img[row,col])**2/(h*h))
                weight[row,col] = sum

        for row in range(padding_size, image_width - padding_size):
            for col in range(padding_size, image_height - padding_size):
                sum = 0
                sum_weight = 0
                for i in range(-padding_size, padding_size + 1):
                    for j in range(-padding_size, padding_size + 1):
                        sum += weight[(row+i),(col+j)]*int(src[(row+i),(col+j)])
                        sum_weight += weight[(row+i),(col+j)]
                dst[row,col] = sum/sum_weight
        end_time = time.time()
        print('Yaroslavsky Filtering Complete. Time:{}'.format(end_time - start_time))
        method_noise = src - dst

        # Visualization
        # plt.subplot(1, 3, 1)
        # plt.imshow(src, cmap='gray')
        # plt.title("Source Image")
        # plt.subplot(1, 3, 2)
        # plt.imshow(dst, cmap='gray')
        # plt.title("Yaroslavsky Filtering")
        # plt.subplot(1, 3, 3)
        # plt.imshow(method_noise, cmap='gray')
        # plt.title("Method Noise")
        # plt.show()

        self.results.append(dst)
        self.method_noise.append(method_noise)
        self.method_names.append("Yaroslavsky Filtering")
        return dst, method_noise

    def NLMeans(self, src, dst=[], t=10, f=3, h=1):
        """
        @description  : Non local means algorithm
        @param  : t : radius of search window
        @param  : f : radius of similarity window 
        @Returns  :
        """
        # t: radius of search window
        # f:radius of similarity window
        # H:degree of filtering
        print("Non-Local Means start. Radius of search window = {}, Radius of similarity window = {}, h = {}".format(t,f, h))
        start_time = time.time()
        width, height = np.shape(src)[0], np.shape(src)[1]
        dst = np.zeros((width,height),dtype='float64')
        padded_img = np.pad(src,((f,f),(f,f)) , 'edge').astype('float64')

        kernel = self.GaussianTemplate(2*f+1,1)
        pbar = tqdm(total=width*height)
        for x in range(0, width):
            for y in range(0, height):
                pbar.update(1)
                x1 = x + f
                y1 = y + f
                W1 = padded_img[x1 - f : x1 + f + 1, y1 - f : y1 + f + 1]
                # print(x1-f,x1+f)
                wmax = 0
                average = 0
                sweight = 0
                rmin = max(x1 - t, f + 1)
                rmax = min(x1 + t, width + f)
                smin = max(y1 - t, f + 1)
                smax = min(y1+ t, height + f)

                for r in range(rmin-1, rmax):
                    for s in range(smin-1, smax):
                        if (r == x1 and s == y1):
                            continue
                        W2 = padded_img[r - f: r+ f + 1, s-f : s + f + 1]
                        dis = np.sum(np.square(kernel*(W1-W2)))
                        w = np.exp(-dis / (h*h))
                        if w > wmax:
                            wmax = w
                        sweight = sweight + w
                        average = average + w * padded_img[r][s]
                average = average + wmax * padded_img[x1][y1]
                sweight = sweight + wmax
                if sweight > 0:
                    dst[x][y] = average / sweight
                else:
                    dst[x][y] = src[x][y]
        end_time = time.time()
        pbar.close()
        print('Non Local Means Complete. Time:{}'.format(end_time - start_time))
        # Visualization
        # plt.subplot(1, 5, 1)
        # plt.imshow(src, cmap='gray')
        # plt.subplot(1, 5, 2)
        # plt.imshow(dst, cmap='gray')
        # plt.subplot(1, 5, 3)
        # plt.imshow(src - dst, cmap='gray')

        # plt.subplot(1, 5, 4)
        # cv2_nlmeans_dst = cv2.fastNlMeansDenoising(src, dst, h, 2*t+1, 2*f+1)
        # plt.imshow(cv2_nlmeans_dst, cmap='gray')
        # plt.subplot(1, 5, 5)
        # plt.imshow(src - cv2_nlmeans_dst, cmap="gray")
        # plt.show()

        method_noise  =  src - dst

        self.results.append(dst)
        self.method_noise.append(method_noise)
        self.method_names.append("Non-Local Means")
        return dst, method_noise
if __name__ == "__main__":
    NLM = NLM()
    dst = []
    
    dst1,method_noise1 = NLM.Gaussian_Filtering(NLM.image, dst, 5, 2)
    dst2,method_noise2 = NLM.Anisotropic_Filtering(NLM.image, dst, 30, 15, 0.1)
    dst3,method_noise3 = NLM.Total_Variation_Minimization(NLM.image,dst,100,0.03)
    dst4,method_noise4 = NLM.Yaroslavsky_Filtering(NLM.image,dst,3,0.8)
    dst5,method_noise5 = NLM.NLMeans(NLM.image,dst,10,3,1.5)

    results = NLM.results
    method_noise = NLM.method_noise
    names = NLM.method_names

    # Visualization
    save_path = "./Result.png"
    plt.figure(figsize=(16,9))
    for i in range(0,len(results)):
        plt.subplot(2,len(results),i+1)
        plt.imshow(results[i],cmap = "gray")
        plt.title(names[i])
    for i in range(0,len(results)):
        plt.subplot(2, len(results), len(results) + i + 1)
        plt.imshow(method_noise[i], cmap="gray")
        plt.title(names[i])
    plt.savefig(save_path)
    plt.show()

