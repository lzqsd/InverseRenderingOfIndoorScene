import glob
import numpy as np
import os.path as osp
from PIL import Image
import random
import struct
from torch.utils.data import Dataset
import scipy.ndimage as ndimage
import cv2
from skimage.measure import block_reduce 
import h5py
import scipy.ndimage as ndimage


class BatchLoader(Dataset):
    def __init__(self, dataRoot, dirs = ['main_xml', 'main_xml1',
        'mainDiffLight_xml', 'mainDiffLight_xml1', 
        'mainDiffMat_xml', 'mainDiffMat_xml1'], 
            imHeight = 240, imWidth = 320, 
            phase='TRAIN', rseed = None, cascadeLevel = 0,
            isLight = False, isAllLight = False,
            envHeight = 8, envWidth = 16, envRow = 120, envCol = 160, 
            SGNum = 12 ):
        
        if phase.upper() == 'TRAIN':
            self.sceneFile = osp.join(dataRoot, 'train.txt')
        elif phase.upper() == 'TEST':
            self.sceneFile = osp.join(dataRoot, 'test.txt') 
        else:
            print('Unrecognized phase for data loader')
            assert(False ) 
        
        with open(self.sceneFile, 'r') as fIn:
            sceneList = fIn.readlines() 
        sceneList = [x.strip() for x in sceneList]

        self.imHeight = imHeight
        self.imWidth = imWidth
        self.phase = phase.upper()
        self.cascadeLevel = cascadeLevel
        self.isLight = isLight
        self.isAllLight = isAllLight
        self.envWidth = envWidth
        self.envHeight = envHeight
        self.envRow = envRow
        self.envCol = envCol
        self.envWidth = envWidth 
        self.envHeight = envHeight
        self.SGNum = SGNum
        
        shapeList = []
        for d in dirs:
            shapeList = shapeList + [osp.join(dataRoot, d, x) for x in sceneList ]
        shapeList = sorted(shapeList)
        print('Shape Num: %d' % len(shapeList ) )

        self.imList = []
        for shape in shapeList:
            imNames = sorted(glob.glob(osp.join(shape, 'im_*.hdr') ) )
            self.imList = self.imList + imNames

        if isAllLight:
            self.imList = [x for x in self.imList if
                    osp.isfile(x.replace('im_', 'imenv_') ) ]
            if cascadeLevel > 0:
                self.imList = [x for x in self.imList if
                        osp.isfile(x.replace('im_',
                            'imenv_').replace('.hdr', '_%d.h5' %
                                (self.cascadeLevel - 1 )  ) ) ]


        print('Image Num: %d' % len(self.imList ) )

        # BRDF parameter
        self.albedoList = [x.replace('im_', 'imbaseColor_').replace('hdr', 'png') for x in self.imList ] 

        self.normalList = [x.replace('im_', 'imnormal_').replace('hdr', 'png') for x in self.imList ]
        self.normalList = [x.replace('DiffLight', '') for x in self.normalList ]

        self.roughList = [x.replace('im_', 'imroughness_').replace('hdr', 'png') for x in self.imList ]

        self.depthList = [x.replace('im_', 'imdepth_').replace('hdr', 'dat') for x in self.imList ]
        self.depthList = [x.replace('DiffLight', '') for x in self.depthList ]
        self.depthList = [x.replace('DiffMat', '') for x in self.depthList ]
        
        self.segList = [x.replace('im_', 'immask_').replace('hdr', 'png') for x in self.imList ]
        self.segList = [x.replace('DiffMat', '') for x in self.segList ]

        if self.cascadeLevel == 0:
            if self.isLight:
                self.envList = [x.replace('im_', 'imenv_') for x in self.imList ]
        else:
            if self.isLight:
                self.envList = [x.replace('im_', 'imenv_') for x in self.imList ]
                self.envPreList = [x.replace('im_', 'imenv_').replace('.hdr', '_%d.h5'  % (self.cascadeLevel -1) ) for x in self.imList ]
            
            self.albedoPreList = [x.replace('im_', 'imbaseColor_').replace('.hdr', '_%d.h5' % (self.cascadeLevel - 1) ) for x in self.imList ]
            self.normalPreList = [x.replace('im_', 'imnormal_').replace('.hdr', '_%d.h5' % (self.cascadeLevel-1) ) for x in self.imList ]
            self.roughPreList = [x.replace('im_', 'imroughness_').replace('.hdr', '_%d.h5' % (self.cascadeLevel-1) ) for x in self.imList ]
            self.depthPreList = [x.replace('im_', 'imdepth_').replace('.hdr', '_%d.h5' % (self.cascadeLevel-1) ) for x in self.imList ]

            self.diffusePreList = [x.replace('im_', 'imdiffuse_').replace('.hdr', '_%d.h5' % (self.cascadeLevel - 1) ) for x in self.imList ]
            self.specularPreList = [x.replace('im_', 'imspecular_').replace('.hdr', '_%d.h5' % (self.cascadeLevel - 1) ) for x in self.imList ]

        # Permute the image list
        self.count = len(self.albedoList )
        self.perm = list(range(self.count ) )

        if rseed is not None:
            random.seed(0)
        random.shuffle(self.perm )

    def __len__(self):
        return len(self.perm )

    def __getitem__(self, ind):
        # Read segmentation
        seg = 0.5 * (self.loadImage(self.segList[self.perm[ind] ] ) + 1)[0:1, :, :]
        segArea = np.logical_and(seg > 0.49, seg < 0.51 ).astype(np.float32 )
        segEnv = (seg < 0.1).astype(np.float32 )
        segObj = (seg > 0.9) 
        
        if self.isLight:
            segObj = segObj.squeeze()
            segObj = ndimage.binary_erosion(segObj, structure=np.ones((7, 7) ),
                    border_value=1)
            segObj = segObj[np.newaxis, :, :]

        segObj = segObj.astype(np.float32 )

        # Read Image
        im = self.loadHdr(self.imList[self.perm[ind] ] )
        # Random scale the image
        im, scale = self.scaleHdr(im, seg)

        # Read albedo
        albedo = self.loadImage(self.albedoList[self.perm[ind] ], isGama = False)
        albedo = (0.5 * (albedo + 1) ) ** 2.2

        # normalize the normal vector so that it will be unit length
        normal = self.loadImage(self.normalList[self.perm[ind] ] )
        normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5) )[np.newaxis, :]

        # Read roughness
        rough = self.loadImage(self.roughList[self.perm[ind] ] )[0:1, :, :]

        # Read depth
        depth = self.loadBinary(self.depthList[self.perm[ind] ])

        if self.isLight == True:
            envmaps, envmapsInd = self.loadEnvmap(self.envList[self.perm[ind] ] )
            envmaps = envmaps * scale 
            if self.cascadeLevel > 0: 
                envmapsPre = self.loadH5(self.envPreList[self.perm[ind] ] ) 
                if envmapsPre is None:
                    print("Wrong envmap pred")
                    envmapsInd = envmapsInd * 0 
                    envmapsPre = np.zeros((84, 120, 160), dtype=np.float32 ) 

        if self.cascadeLevel > 0:
            # Read albedo
            albedoPre = self.loadH5(self.albedoPreList[self.perm[ind] ] )
            albedoPre = albedoPre / np.maximum(np.mean(albedoPre ), 1e-10) / 3

            # normalize the normal vector so that it will be unit length
            normalPre = self.loadH5(self.normalPreList[self.perm[ind] ] )
            normalPre = normalPre / np.sqrt(np.maximum(np.sum(normalPre * normalPre, axis=0), 1e-5) )[np.newaxis, :]
            normalPre = 0.5 * (normalPre + 1)

            # Read roughness
            roughPre = self.loadH5(self.roughPreList[self.perm[ind] ] )[0:1, :, :]
            roughPre = 0.5 * (roughPre + 1)

            # Read depth
            depthPre = self.loadH5(self.depthPreList[self.perm[ind] ] )
            depthPre = depthPre / np.maximum(np.mean(depthPre), 1e-10) / 3

            diffusePre = self.loadH5(self.diffusePreList[self.perm[ind] ] )
            diffusePre = diffusePre / max(diffusePre.max(), 1e-10)

            specularPre = self.loadH5(self.specularPreList[self.perm[ind] ] )
            specularPre = specularPre / max(specularPre.max(), 1e-10)



        batchDict = {'albedo': albedo,
                'normal': normal,
                'rough': rough,
                'depth': depth,
                'segArea': segArea,
                'segEnv': segEnv,
                'segObj': segObj,
                'im': im,
                'name': self.imList[self.perm[ind] ]
                }

        if self.isLight:
            batchDict['envmaps'] = envmaps
            batchDict['envmapsInd'] = envmapsInd

            if self.cascadeLevel > 0:
                batchDict['envmapsPre'] = envmapsPre

        if self.cascadeLevel > 0:
            batchDict['albedoPre'] = albedoPre
            batchDict['normalPre'] = normalPre
            batchDict['roughPre'] = roughPre
            batchDict['depthPre'] = depthPre

            batchDict['diffusePre'] = diffusePre
            batchDict['specularPre'] = specularPre

        return batchDict


    def loadImage(self, imName, isGama = False):
        if not(osp.isfile(imName ) ):
            print(imName )
            assert(False )

        im = Image.open(imName)
        im = im.resize([self.imWidth, self.imHeight], Image.ANTIALIAS )

        im = np.asarray(im, dtype=np.float32)
        if isGama:
            im = (im / 255.0) ** 2.2
            im = 2 * im - 1
        else:
            im = (im - 127.5) / 127.5
        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1] )

        return im

    def loadHdr(self, imName):
        if not(osp.isfile(imName ) ):
            print(imName )
            assert(False )
        im = cv2.imread(imName, -1)
        if im is None:
            print(imName )
            assert(False )
        im = cv2.resize(im, (self.imWidth, self.imHeight), interpolation = cv2.INTER_AREA )
        im = np.transpose(im, [2, 0, 1])
        im = im[::-1, :, :]
        return im

    def scaleHdr(self, hdr, seg):
        intensityArr = (hdr * seg).flatten()
        intensityArr.sort()
        if self.phase.upper() == 'TRAIN':
            scale = (0.95 - 0.1 * np.random.random() )  / np.clip(intensityArr[int(0.95 * self.imWidth * self.imHeight * 3) ], 0.1, None)
        elif self.phase.upper() == 'TEST':
            scale = (0.95 - 0.05)  / np.clip(intensityArr[int(0.95 * self.imWidth * self.imHeight * 3) ], 0.1, None)
        hdr = scale * hdr
        return np.clip(hdr, 0, 1), scale 

    def loadBinary(self, imName ):
        if not(osp.isfile(imName ) ):
            print(imName )
            assert(False )
        with open(imName, 'rb') as fIn:
            hBuffer = fIn.read(4)
            height = struct.unpack('i', hBuffer)[0]
            wBuffer = fIn.read(4)
            width = struct.unpack('i', wBuffer)[0]
            dBuffer = fIn.read(4 * width * height )
            depth = np.asarray(struct.unpack('f' * height * width, dBuffer), dtype=np.float32 )
            depth = depth.reshape([height, width] )
            depth = cv2.resize(depth, (self.imWidth, self.imHeight), interpolation=cv2.INTER_AREA )

        return depth[np.newaxis, :, :]

    def loadH5(self, imName ): 
        try:
            hf = h5py.File(imName, 'r')
            im = np.array(hf.get('data' ) )
            return im 
        except:
            return None


    def loadEnvmap(self, envName ):
        if not osp.isfile(envName ):
            env = np.zeros( [3, self.envRow, self.envCol,
                self.envHeight, self.envWidth], dtype = np.float32 )
            envInd = np.zeros([1, 1, 1], dtype=np.float32 )
            print('Warning: the envmap %s does not exist.' % envName )
            return env, envInd
        else:
            envHeightOrig, envWidthOrig = 16, 32
            assert( (envHeightOrig / self.envHeight) == (envWidthOrig / self.envWidth) )
            assert( envHeightOrig % self.envHeight == 0)
            
            env = cv2.imread(envName, -1 ) 

            if not env is None:
                env = env.reshape(self.envRow, envHeightOrig, self.envCol,
                    envWidthOrig, 3)
                env = np.ascontiguousarray(env.transpose([4, 0, 2, 1, 3] ) )

                scale = envHeightOrig / self.envHeight
                if scale > 1:
                    env = block_reduce(env, block_size = (1, 1, 1, 2, 2), func = np.mean )

                envInd = np.ones([1, 1, 1], dtype=np.float32 )
                return env, envInd
            else:
                env = np.zeros( [3, self.envRow, self.envCol,
                    self.envHeight, self.envWidth], dtype = np.float32 )
                envInd = np.zeros([1, 1, 1], dtype=np.float32 )
                print('Warning: the envmap %s does not exist.' % envName )
                return env, envInd
                

            return env, envInd
