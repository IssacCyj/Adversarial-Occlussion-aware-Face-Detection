#only modify siftfacelayer
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""
import pickle
import caffe
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv


class RoIDataLayer(caffe.Layer):
    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            return get_minibatch(minibatch_db, self._num_classes)

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(10)
            self._prefetch_process = BlobFetcher(self._blob_queue,
                                                 self._roidb,
                                                 self._num_classes)
            self._prefetch_process.start()
            # Terminate the child process when the parent exists
            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()
            import atexit
            atexit.register(cleanup)

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
            max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
        self._name_to_top_map['data'] = idx
        idx += 1

        if cfg.TRAIN.HAS_RPN:
            top[idx].reshape(1, 3)
            self._name_to_top_map['im_info'] = idx
            idx += 1

            top[idx].reshape(1, 4)
            self._name_to_top_map['gt_boxes'] = idx
            idx += 1

            top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 1,
            max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
            self._name_to_top_map['gt_mask'] = idx
            idx += 1
        else: # not using RPN
            # rois blob: holds R regions of interest, each is a 5-tuple
            # (n, x1, y1, x2, y2) specifying an image batch index n and a
            # rectangle (x1, y1, x2, y2)
            top[idx].reshape(1, 5)
            self._name_to_top_map['rois'] = idx
            idx += 1

            # labels blob: R categorical labels in [0, ..., K] for K foreground
            # classes plus background
            top[idx].reshape(1)
            self._name_to_top_map['labels'] = idx
            idx += 1

            if cfg.TRAIN.BBOX_REG:
                # bbox_targets blob: R bounding-box regression targets with 4
                # targets per class
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_targets'] = idx
                idx += 1

                # bbox_inside_weights blob: At most 4 targets per roi are active;
                # thisbinary vector sepcifies the subset of active targets
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_inside_weights'] = idx
                idx += 1

                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_outside_weights'] = idx
                idx += 1

        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

        #print('data blob shape: ', top[0].data.shape)
        #print('mask blob shape: ', top[3].data.shape)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, queue, roidb, num_classes):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._roidb = roidb
        self._num_classes = num_classes
        self._perm = None
        self._cur = 0
        self._shuffle_roidb_inds()
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        # TODO(rbg): remove duplicated code
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        # TODO(rbg): remove duplicated code
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def run(self):
        print 'BlobFetcher started'
        while True:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            blobs = get_minibatch(minibatch_db, self._num_classes)
            self._queue.put(blobs)






class TileLayer(caffe.Layer):
    def setup(self, bottom, top):
        """Setup the TileLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._channels = layer_params['channels']
        self._count_drop  = layer_params['count_drop']
        self._permute_count  = layer_params['permute_count']

        self._iter_size = layer_params['iter_size']
        self._maintain_before = layer_params['maintain_before'] # maintain the first image unchanged 

        self._count_iter = 0
        self.cnt = 0
        self._name_to_bottom_map = {
            'mask_pred': 0 }

        # 0 means block, 1 means maintain 

        self._name_to_top_map = {
            'mask_pred_tile': 0 ,
            'mask_pred_thres':1,
            'mask_inv':2}


        # top[0].reshape(*(bottom[0].data.shape))
        top[0].reshape(bottom[0].data.shape[0], self._channels, bottom[0].data.shape[2], bottom[0].data.shape[3])
        top[1].reshape(bottom[0].data.shape[0], 1, bottom[0].data.shape[2], bottom[0].data.shape[3])
        top[2].reshape(bottom[0].data.shape[0], 1, bottom[0].data.shape[2], bottom[0].data.shape[3])

        print 'TileLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def select_mask(self, mask_pred):
#1 means block in the input
        self.cnt = 0
        pool_len = mask_pred.shape[2]
        sample_num = mask_pred.shape[0]

        mask_pixels = pool_len * pool_len

        count_drop = self._count_drop #15
        permute_count = self._permute_count #20

        mask_sel = np.ones((sample_num, 1, pool_len, pool_len))
        mask_for_loss = np.ones((sample_num, 1, pool_len, pool_len))
        for i in range(sample_num):

            #not exactly as it mentioned in the paper
            #15/49 = 1/3 are selected as 0
            #first choose top 20 lowest predicted pixels in mask (trained in stage 2)
            #randomly choose 15 of 20 pixels to set zero
            rp = np.random.permutation(np.arange(permute_count))
            rp = rp[0: count_drop]

            final_mask = np.ones(mask_pixels)

            now_mask_pred = mask_pred[i]
            now_mask_pred_array = np.reshape(now_mask_pred, mask_pixels)
            #convert the mask to an array and sort it ascendingly with the pixel value
            sorted_ids = np.argsort(now_mask_pred_array) 
            now_ids = sorted_ids[rp]

            sel = np.zeros(mask_pixels)
            sel[now_ids] = 1
            _final_mask = sel * now_mask_pred_array
 	    if i==10000:
            #use this method later
            final_mask[np.where(_final_mask==0)] = 1
	        final_mask[np.where(_final_mask!=0)] = 0
	    #have to try this first, finding thr most important part to mask.
            #final_mask[np.where(final_mask!=0)] = 1

            now_mask = np.reshape(final_mask, (pool_len, pool_len))
	        _final_mask[np.where(_final_mask==0)] = 1
            _now_mask = np.reshape(_final_mask, (pool_len, pool_len))
            if self.cnt==0:
	        #print(sel)
		self.cnt += 1
	        print(now_mask)

            mask_sel[i,0,:,:] = np.copy(now_mask)
            mask_for_loss[i,0,:,:] = np.copy(_now_mask)
        return mask_sel,mask_for_loss

    def forward(self, bottom, top):

#1 means block!!
        mask_pred = np.copy(bottom[0].data)
        sample_num = mask_pred.shape[0]
        pool_len = mask_pred.shape[2]


        self._count_iter = (self._count_iter + 1) % self._iter_size
        if self._count_iter >= self._maintain_before:        
            mask_sel,mask_for_loss = self.select_mask(mask_pred)
        else:
            mask_sel = np.ones((sample_num, 1, pool_len, pool_len))
            mask_for_loss = np.ones((sample_num, 1, pool_len, pool_len))


	#print(mask_sel[0,0,:,:])
        mask_pred_tile = np.tile(mask_sel, [1, self._channels, 1, 1])

        mask_inv = np.abs(1-mask_sel)
        #print(mask_sel[0,0,:,:])                                 
        #print(mask_inv[0,0,:,:])                                                                                              
        top_ind = self._name_to_top_map['mask_pred_tile']
        top[top_ind].reshape(*(mask_pred_tile.shape))
        top[top_ind].data[...] = mask_pred_tile.astype(np.float32, copy=False)

        top_ind = self._name_to_top_map['mask_pred_thres']
        top[top_ind].reshape(*(mask_sel.shape))
        top[top_ind].data[...] = mask_sel.astype(np.float32, copy=False)

        top_ind = self._name_to_top_map['mask_inv']
        top[top_ind].reshape(*(mask_inv.shape))
        top[top_ind].data[...] = mask_inv.astype(np.float32, copy=False)
        

    def backward(self, top, propagate_down, bottom):
        top_0_diff = np.zeros(np.shape(top[2].diff))
        top_0_diff[:,0,:,:] = np.mean(top[0].diff, axis=1)
        #bottom[0].diff[...] *= top[1].diff
        bottom[0].diff[...] = (top[2].diff + top_0_diff)
        #print("\n\n\n\nbottom[0].diff:")
      #  print('inv_diff: \n',top[2].diff[0,0,:,:])
      #  print('cls_diff: \n',top_0_diff[0,0,:,:])
        #print(top_0_diff.shape) 
        #tile_diff = np.tile(top[0].diff, [1, 1, 1, 1])
        #bottom[0].diff[...] *= (tile_diff + top[1].diff)

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass





class TileLayer2(caffe.Layer):
    def setup(self, bottom, top):
        """Setup the TileLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._channels = layer_params['channels']
        self._count_drop  = layer_params['count_drop']
        self._permute_count  = layer_params['permute_count']

        self._iter_size = layer_params['iter_size']
        self._maintain_before = layer_params['maintain_before'] # maintain the first image unchanged 

        self._count_iter = 0

        self._name_to_bottom_map = {
            'mask_pred': 0,
            'gt_mask_fg': 1}
        # 0 means block, 1 means maintain 

        self._name_to_top_map = {
            'mask_pred_tile': 0 ,
            'mask_pred_thres':1,
            'mask_inv':2}


        # top[0].reshape(*(bottom[0].data.shape))
        top[0].reshape(bottom[0].data.shape[0], self._channels, bottom[0].data.shape[2], bottom[0].data.shape[3])
        top[1].reshape(bottom[0].data.shape[0], 1, bottom[0].data.shape[2], bottom[0].data.shape[3])
        top[2].reshape(bottom[0].data.shape[0], 1, bottom[0].data.shape[2], bottom[0].data.shape[3])

        assert len(top) == len(self._name_to_top_map)

    def select_mask(self, mask_pred):
#1 means block in the input
        cnt = 0
        pool_len = mask_pred.shape[2]
        sample_num = mask_pred.shape[0]

        mask_pixels = pool_len * pool_len

        count_drop = self._count_drop #15
        permute_count = self._permute_count #20

        mask_sel = np.ones((sample_num, 1, pool_len, pool_len))
        mask_for_loss = np.ones((sample_num, 1, pool_len, pool_len))
        for i in range(sample_num):

            #not exactly as it mentioned in the paper
            #15/49 = 1/3 are selected as 0
            #first choose top 20 lowest predicted pixels in mask (trained in stage 2)
            #randomly choose 15 of 20 pixels to set zero
            rp = np.random.permutation(np.arange(permute_count))
            rp = rp[0: count_drop]

            final_mask = np.ones(mask_pixels)

            now_mask_pred = mask_pred[i]
            now_mask_pred_array = np.reshape(now_mask_pred, mask_pixels)
            #convert the mask to an array and sort it ascendingly with the pixel value
            sorted_ids = np.argsort(now_mask_pred_array) 
            now_ids = sorted_ids[rp]

            sel = np.zeros(mask_pixels)
            sel[now_ids] = 1
            _final_mask = sel * now_mask_pred_array
            if i==100000:
                print(mask_pred[i,0,:,:])
            #use this method later
            final_mask[np.where(_final_mask==0)] = 1
            final_mask[np.where(_final_mask!=0)] = 0
            #have to try this first, finding thr most important part to mask.
            #final_mask[np.where(final_mask!=0)] = 1

            now_mask = np.reshape(final_mask, (pool_len, pool_len))
            _final_mask[np.where(_final_mask==0)] = 1
            _now_mask = np.reshape(_final_mask, (pool_len, pool_len))
            if cnt==10000000:
                print('GT: ')
                print(now_mask)
            cnt += 1
            mask_sel[i,0,:,:] = np.copy(now_mask)
            mask_for_loss[i,0,:,:] = np.copy(_now_mask)
        return mask_sel,mask_for_loss

    def forward(self, bottom, top):
        gt_mask_fg = np.copy(bottom[1].data)
        mask_pred = np.copy(bottom[0].data)
        sample_num = mask_pred.shape[0]
        pool_len = mask_pred.shape[2]
        #print("\n\nN:",sample_num)
        self._count_iter = (self._count_iter + 1) % self._iter_size#itersize = 5
        if self._count_iter == 0:        
            mask_sel,mask_for_loss = self.select_mask(mask_pred)
        elif self._count_iter == 1:
            mask_sel = np.ones((sample_num, 1, pool_len, pool_len))
            for i in range(sample_num):
                mask_sel_pre = np.ones(pool_len*pool_len)
                arg_array = np.random.permutation(range(pool_len*pool_len))[:20]
                mask_sel_pre[arg_array] = 0
                mask_rand = np.reshape(mask_sel_pre,(pool_len,pool_len))
                mask_sel[i,0,:,:] = np.copy(mask_rand)
            mask_for_loss = np.ones((sample_num, 1, pool_len, pool_len))
        elif self._count_iter == 2:
            mask_sel = np.ones((sample_num, 1, pool_len, pool_len))
            for i in range(sample_num):
                mask_sel_pre = np.ones((pool_len,pool_len))
                rnd = np.random.randint(0,4)
                drop = pool_len/2 + 1
                if rnd == 0 :
                    mask_sel_pre[:,:drop] = 0
                elif rnd == 1 :
                    mask_sel_pre[:,drop-1:] = 0
                elif rnd == 2 :
                    mask_sel_pre[:drop,:] = 0
                else:
                    mask_sel_pre[drop-1:,:] = 0
                mask_sel[i,0,:,:] = np.copy(mask_sel_pre)
            mask_for_loss = np.ones((sample_num, 1, pool_len, pool_len))
        else:
            mask_sel = np.ones((sample_num, 1, pool_len, pool_len))
            mask_for_loss = np.ones((sample_num, 1, pool_len, pool_len))

        if not np.all(np.unique(gt_mask_fg) == 1 ):
#            print(np.unique(gt_mask_fg))
            mask_sel = np.ones((sample_num, 1, pool_len, pool_len))
            mask_for_loss = np.ones((sample_num, 1, pool_len, pool_len))
#        else:
#            print(np.unique(gt_mask_fg))

        mask_inv = np.abs(1-mask_sel)

        #print(mask_sel[0,0,:,:])
        mask_pred_tile = np.tile(mask_sel, [1, self._channels, 1, 1])
        #print(mask_pred_tile[0,0,:,:])
        top_ind = self._name_to_top_map['mask_pred_tile']
        top[top_ind].reshape(*(mask_pred_tile.shape))
        top[top_ind].data[...] = mask_pred_tile.astype(np.float32, copy=False)

        top_ind = self._name_to_top_map['mask_pred_thres']
        top[top_ind].reshape(*(mask_sel.shape))
        top[top_ind].data[...] = mask_sel.astype(np.float32, copy=False)

        top_ind = self._name_to_top_map['mask_inv']
        top[top_ind].reshape(*(mask_inv.shape))
        top[top_ind].data[...] = mask_inv.astype(np.float32, copy=False)
        

    def backward(self, top, propagate_down, bottom):
        top_0_diff = np.zeros(np.shape(top[2].diff))
        top_0_diff[:,0,:,:] = np.mean(top[0].diff, axis=1)
        #bottom[0].diff[...] *= top[1].diff
        bottom[0].diff[...] = (top[2].diff + top_0_diff)
        #print("\n\n\n\nbottom[0].diff:")
        #print(top_0_diff[:,0,:,:])
        #print(top_0_diff.shape) 
        #tile_diff = np.tile(top[0].diff, [1, 1, 1, 1])
        #bottom[0].diff[...] *= (tile_diff + top[1].diff)

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass



class SiftFaceLayer(caffe.Layer):
    def setup(self, bottom, top):

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
        self.onlyface_mask = np.ones(bottom[0].data.shape)

        self._name_to_bottom_map = {
            'conv5_3': 0,
            'bbox_pred': 1,
            'cls_score': 2,
            'gt_mask': 3,
            'rois': 4,
            'im_info': 5}

        self._name_to_top_map = {
            'onlyface': 0,
            'gt_mask_fg': 1}

        top[0].reshape(*(bottom[0].data.shape))
        top[1].reshape(*(bottom[3].data.shape))

        print 'SiftFaceLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):

        #conv5_3 = np.copy(bottom[0].data)
        assert(bottom[0].data.shape[0] == 1)
        box_deltas = np.copy(bottom[1].data)
        scores =  np.copy(bottom[2].data)
        gt_mask_fg =  np.copy(bottom[3].data)
        #print(np.mean(gt_mask_fg))
        onlyface = np.copy(bottom[0].data)
        rois = np.copy(bottom[4].data)
        im_info = np.copy(bottom[5].data)

        boxes = rois[:, 1:5] 
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        # boxes = clip_boxes(pred_boxes, gt_mask_fg[0,0,:,:].shape[::-1])
        boxes = clip_boxes(pred_boxes, (int(im_info[0][0]),int(im_info[0][1])))


        if np.all(np.unique(gt_mask_fg) == 1):
            ## masks for imges other than occlude are set ones
            onlyface = np.zeros(onlyface.shape)
            gt_mask_fg = np.zeros(gt_mask_fg.shape)
            #print(np.sum(gt_mask_fg))
        else:
            #print('nonzero input !!!')
            CONF_THRESH = 0.6
            NMS_THRESH = 0.25
            zoom = 16

            #find face areas
            cls_ind = 1
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                    cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]

            keep = np.where(dets[:, 4] > CONF_THRESH)
            dets = dets[keep] #shape(n,5) n means n predictes boxes, 5 includes top left and bottom right coords and a score 
            #enlarge boxes 
            # dets[:,:4] *= 1.1 
        #    print(dets)
        #    print(dets.shape)
        #    print(bottom[3].data.shape)
        #    print(bottom[0].data.shape)
            #generate a mask for gt mask
            mask4gt = np.zeros(bottom[3].data.shape)
            for each in dets:
                mask4gt[:,:,each[1]:each[3]+1,each[0]:each[2]+1] = 1

      #      gt_mask_fg *= mask4gt

            # map to conv5_3
            dets[:,:4] //= zoom 

            #generate a mask for conv5_3
            mask4conv = np.zeros(bottom[0].data.shape)
            for each in dets:
                mask4conv[:,:,each[1]:each[3]+1,each[0]:each[2]+1] = 1

     #       onlyface *= mask4conv
            self.onlyface_mask = mask4conv

#        print(np.sum(onlyface))

        top_ind = self._name_to_top_map['onlyface']
        top[top_ind].reshape(*(onlyface.shape))
        top[top_ind].data[...] = onlyface.astype(np.float32, copy=False)

        top_ind = self._name_to_top_map['gt_mask_fg']
        top[top_ind].reshape(*(gt_mask_fg.shape))
        top[top_ind].data[...] = gt_mask_fg.astype(np.float32, copy=False)


    def backward(self, top, propagate_down, bottom):
        gt_mask_fg =  np.copy(bottom[3].data)
        # mask for imges other than occlude are set ones
        if np.all(np.unique(gt_mask_fg) == 1):
            #print('back 0')
            bottom[0].diff[...] = 0
        else:
            #print("back")
            for i in range(4):
                if not propagate_down[i]:
                    continue
                else:
      #              bottom[0].diff[...] = top[0].diff * self.onlyface_mask
                    bottom[0].diff[...] = top[0].diff

    def reshape(self, bottom, top):

        """Reshaping happens during the call to forward."""
        pass



class SiftFace4TestLayer(caffe.Layer):
    def setup(self, bottom, top):

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
        self.onlyface_mask = np.ones(bottom[0].data.shape)

        self._name_to_bottom_map = {
            'conv5_3': 0,
            'bbox_pred': 1,
            'cls_score': 2,
            'rois': 3,
            'im_info': 4}

        self._name_to_top_map = {
            'onlyface': 0}

        top[0].reshape(*(bottom[0].data.shape))
        

        print 'SiftFaceLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):

        #conv5_3 = np.copy(bottom[0].data)
        assert(bottom[0].data.shape[0] == 1)
        box_deltas = np.copy(bottom[1].data)
        scores =  np.copy(bottom[2].data)
        onlyface = np.copy(bottom[0].data)
        rois = np.copy(bottom[3].data)
        im_info = np.copy(bottom[4].data)
        #print('layer rois: ',rois)
        boxes = rois[:, 1:5] 
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        # boxes = clip_boxes(pred_boxes, gt_mask_fg[0,0,:,:].shape[::-1])
        boxes = clip_boxes(pred_boxes, (int(im_info[0][0]),int(im_info[0][1])))
        #print('im_info',(int(im_info[0][0]),int(im_info[0][1]),int(im_info[0][2])))


        CONF_THRESH = 0.65
        NMS_THRESH = 0.15
        zoom = 16


       # print('layerbox:', boxes)
        #find face areas
        cls_ind = 1
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        keep = np.where(dets[:, 4] > CONF_THRESH)
        dets = dets[keep] #shape(n,5) n means n predictes boxes, 5 includes top left and bottom right coords and a score 
        #enlarge boxes 
        #print('dddets: ',dets)
    #    dets[:,:4] *= 1 
    #    print(dets)
    #    print(dets.shape)
    #    print(bottom[3].data.shape)
    #    print(bottom[0].data.shape)
        #generate a mask for gt mask
        # mask4gt = np.zeros(bottom[3].data.shape)
        # for each in dets:
        #     mask4gt[:,:,each[0]:each[2]+1,each[1]:each[3]+1] = 1

        # gt_mask_fg *= mask4gt
        
     
        # map to conv5_3
        dets[:,:4] //= zoom 
        #print('conv53:', bottom[0].data.shape)
        #print('premask: ',dets.shape)
        #generate a mask for conv5_3
        mask4conv = np.zeros(bottom[0].data.shape)
        for each in dets:
            mask4conv[:,:,each[1]:each[3]+1,each[0]:each[2]+1] = 1

#        pickle.dump(mask4conv, open("vis.txt", "w"))
        onlyface *= mask4conv
        self.onlyface_mask = mask4conv

#        print(np.sum(onlyface))

        top_ind = self._name_to_top_map['onlyface']
        top[top_ind].reshape(*(onlyface.shape))
        top[top_ind].data[...] = onlyface.astype(np.float32, copy=False)



    def backward(self, top, propagate_down, bottom):
         pass
    def reshape(self, bottom, top):

        """Reshaping happens during the call to forward."""
        pass


# class SiftFace4TestLayer(caffe.Layer):
#     def setup(self, bottom, top):

#         # parse the layer parameter string, which must be valid YAML
#         layer_params = yaml.load(self.param_str_)
#         self.onlyface_mask = np.ones(bottom[0].data.shape)

#         self._name_to_bottom_map = {
#             'conv5_3': 0,
#             'bbox_pred': 1,
#             'cls_score':2 }

#         self._name_to_top_map = {
#             'onlyface': 0}

#         top[0].reshape(*(bottom[0].data.shape))

#         print 'SiftFaceLayer: name_to_top:', self._name_to_top_map
#         assert len(top) == len(self._name_to_top_map)

#     def forward(self, bottom, top):

#         #conv5_3 = np.copy(bottom[0].data)
#         assert(bottom[0].data.shape[0] == 1)
#         boxes = np.copy(bottom[1].data)
#         scores =  np.copy(bottom[2].data)


#         onlyface = np.copy(bottom[0].data)

#         CONF_THRESH = 0.6
#         NMS_THRESH = 0.3
#         zoom = 16

#         #find face areas
#         cls_ind = 1
#         cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
#         cls_scores = scores[:, cls_ind]
#         dets = np.hstack((cls_boxes,
#                 cls_scores[:, np.newaxis])).astype(np.float32)
#         keep = nms(dets, NMS_THRESH)
#         dets = dets[keep, :]

#         keep = np.where(dets[:, 4] > CONF_THRESH)
#         dets = dets[keep] #shape(n,5) n means n predictes boxes, 5 includes top left and bottom right coords and a score 
#         #enlarge boxes 
#         dets[:,:4] *= 1.3 

#         # map to conv5_3
#         dets[:,:4] //= zoom 

#         #generate a mask for conv5_3
#         mask4conv = np.zeros(bottom[0].data.shape)
#         for each in dets:
#             mask4conv[:,:,each[0]:each[2]+1,each[1]:each[3]+1] = 1

# #            onlyface *= mask4conv
#         self.onlyface_mask = mask4conv

# #        print(np.sum(onlyface))

#         top_ind = self._name_to_top_map['onlyface']
#         top[top_ind].reshape(*(onlyface.shape))
#         top[top_ind].data[...] = onlyface.astype(np.float32, copy=False)



#     def backward(self, top, propagate_down, bottom):
#         pass

#     def reshape(self, bottom, top):

#         """Reshaping happens during the call to forward."""
#         pass










class ShuffleMaskLayer(caffe.Layer):
    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._channels = layer_params['channels']
        self._name_to_bottom_map = {
            'mask_pred_thres': 0 }
        self._name_to_top_map = {
            'mask_pred_tile_shuffle': 0}

        top[0].reshape(bottom[0].data.shape[0], self._channels, bottom[0].data.shape[2], bottom[0].data.shape[3])
        print 'ShuffleMaskLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        mask_pred_thres = np.copy(bottom[0].data)
        sample_num = mask_pred_thres.shape[0]
        pool_len = mask_pred_thres.shape[2]
        mask_pixels = pool_len * pool_len

        mask_pred_tile_shuffle = np.ones((sample_num, self._channels, pool_len, pool_len))

        for i in range(sample_num):
            drop_cnt = len(np.where(mask_pred_thres[i,0,:,:]==0)[0])
            mask_thres_array = np.reshape(mask_pred_thres[i,0,:,:], mask_pixels)
            drop_ind = np.where(mask_thres_array==0)[0]
            for j in range(self._channels):
                rnd = np.random.rand(drop_cnt)
                shuffle_mask = np.ones(mask_pixels)
                shuffle_mask[drop_ind] = rnd
                _shuffle_mask = np.reshape(shuffle_mask, (pool_len, pool_len))
                mask_pred_tile_shuffle[i,j,:,:] = np.copy(_shuffle_mask)

        top_ind = self._name_to_top_map['mask_pred_tile_shuffle']
        top[top_ind].reshape(*(mask_pred_tile_shuffle.shape))
        top[top_ind].data[...] = mask_pred_tile_shuffle.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        pass


    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


         
            
            
            
class MaskPredLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._name_to_bottom_map = {
            'mask_pred': 0,
            'mask_gt': 1}
        self._name_to_top_map = {
            'loss': 0}
        self.ignore_label = None
        top[0].reshape(1)
        print 'MaskPredLossLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        N = bottom[0].shape[0]
        mask_pred  = bottom[0].data
        mask_label = bottom[1].data
       
        ary = np.reshape(mask_pred[0,0,:,:],49)
        ids = np.argsort(ary)
        ary[ids[:15]]=0
        ary[np.where(ary!=0)]=1
        msk = np.reshape(ary,(7,7))
        #print("mask_pred: ")
        #print(msk)        

        count_bit = 1 
        for i in range(len(bottom[0].shape)):
            count_bit = count_bit * bottom[0].shape[i]

        # copy from: https://github.com/philkr/voc-classification/blob/master/src/python_layers.py#L52
        f, df, t = bottom[0].data, bottom[0].diff, bottom[1].data
        mask = (self.ignore_label is None or t != self.ignore_label)
        lZ  = np.log(1+np.exp(-np.abs(f))) * mask
        dlZ = np.exp(np.minimum(f,0))/(np.exp(np.minimum(f,0))+np.exp(-np.maximum(f,0))) * mask


        # top[0].data[...] = np.sum(lZ + ((f>0)-t)*f * mask) / N
        # df[...] = (dlZ - t*mask) / N

        lZ = lZ + ((f>0)-t)*f * mask
        df[...] = (dlZ - t*mask) / count_bit

        for i in range(N):
            if (np.sum(mask_label[i,0,:,:])==49 or np.sum(mask_label[i,0,:,:])==20):
                lZ[i] = lZ[i] * 0.0
                df[i] = lZ[i] * 0.0

        # for i in range(N):
        #     lbl = labels[i]
        #     prop_before_select = prop_before[i][lbl]
        #     prop_after_select = prop_after[i][lbl]

        #     if (lbl > 0 and prop_after_select + self._score_thres < prop_before_select) == False :
        #         lZ[i] = lZ[i] * 0.0
        #         df[i] = lZ[i] * 0.0

        top[0].data[...] = np.sum(lZ) / count_bit

    def backward(self, top, prop, bottom):
        bottom[0].diff[...] *= top[0].diff

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass




class MaskGenLayer(caffe.Layer):
    def setup(self, bottom, top):

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._channels = layer_params['channels']
        self._means = layer_params['means']
        #self._count_drop  = layer_params['count_drop']
        #self._permute_count  = layer_params['permute_count']

#        self._iter_size = layer_params['iter_size']
#        self._maintain_before = layer_params['maintain_before'] # maintain the first image unchanged 
#
#        self._count_iter = 0

        self._name_to_bottom_map = {
            'mask_pred': 0 }

        # 0 means block, 1 means maintain 

        self._name_to_top_map = {
            'mask_pred_tile': 0,
            'mask_pred_thres': 1 }


        # top[0].reshape(*(bottom[0].data.shape))
        top[0].reshape(bottom[0].data.shape[0], self._channels, 7, 7)
        top[1].reshape(bottom[0].data.shape[0], self._channels, 7, 7)


        assert len(top) == len(self._name_to_top_map)

    def generate_mask(self, mask_pred):
#0 means block in the input
        pool_len = 7
        k = mask_pred.shape[2]
        stride = pool_len/k
        stride_up = int(pool_len/k)+int(pool_len%k > 0)
        sample_num = mask_pred.shape[0]

        mask_pixels = k * k

        # count_drop = self._count_drop #15
        # permute_count = self._permute_count #20

        mask_gen = np.ones((sample_num, 1, pool_len, pool_len))
        mask_2_2 = np.ones((sample_num, 1, stride, stride))

        for i in range(sample_num):

            now_mask_pred = mask_pred[i]
            now_mask_pred_array = np.reshape(now_mask_pred, mask_pixels)
            #convert the mask to an array and sort it ascendingly with the pixel value
            sorted_ids = np.argsort(now_mask_pred_array) 
            now_ids = sorted_ids[:2]
            for ii in now_ids:
                if ii/2 ==0:
                    mask_gen[i,0,:stride_up,ii*stride:ii*stride+stride_up] = 0
                    mask_2_2[i,0,0,ii%2] = 0
                else:
                    j = ii%2
                    mask_gen[i,0,stride:stride+stride,j*stride:j*stride+stride] = 0
                    mask_2_2[i,0,1,ii%2] = 0
        
	    #if ii == 0:
	       #print(now_ids)
	       #print(mask_pred[0])
	       # print("mask:")
	       #print(mask_gen[0])
	    
        return mask_gen,mask_2_2

    def forward(self, bottom, top):

#0 means block!!
        mask_pred = np.copy(bottom[0].data)

#       self._count_iter = (self._count_iter + 1) % self._iter_size
#       if self._count_iter >= self._maintain_before:
        mask_gen,mask_2_2 = self.generate_mask(mask_pred)
#	else:
#	    mask_gen = np.ones((sample_num, 1, pool_len, pool_len))

        mask_pred_tile = np.tile(mask_gen, [1, self._channels, 1, 1])

        top_ind = self._name_to_top_map['mask_pred_tile']
        top[top_ind].reshape(*(mask_pred_tile.shape))
        top[top_ind].data[...] = mask_pred_tile.astype(np.float32, copy=False)

        top_ind = self._name_to_top_map['mask_pred_thres']
        top[top_ind].reshape(*(mask_gen.shape))
        top[top_ind].data[...] = mask_gen.astype(np.float32, copy=False)

	#print("\n\nind_shape: ",mask_gen.shape)        

    def backward(self, top, propagate_down, bottom):
        top_diff = np.zeros((top[1].diff.shape[0],top[1].diff.shape[1],top[1].diff.shape[2]/3,top[1].diff.shape[3]/3))
        for i in range(top[1].diff.shape[0]):
            top_diff[i,0,0,0] = np.mean(top[1].diff[i,0,:4,:4])
            top_diff[i,0,0,1] = np.mean(top[1].diff[i,0,:4,3:])
            top_diff[i,0,1,0] = np.mean(top[1].diff[i,0,3:,:4])
            top_diff[i,0,1,1] = np.mean(top[1].diff[i,0,3:,3:])        
        bottom[0].diff[...] = top_diff

        #bottom[0].diff[...] = top[1].diff
        #print("ind_loss:")
        #print(bottom[0].diff[0,:,:,:])

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass




class SumLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._name_to_bottom_map = {
            'mask_pred': 0}
        self._name_to_top_map = {
            'loss': 0}
        top[0].reshape(1)
        print 'SumforLossLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):   
        mask_pred = np.copy(bottom[0].data)
        #print('\nmask_pred: ', mask_pred[0,0,:,:])
        batchSz = bottom[0].data.shape[0]
        mask_pred[np.where(mask_pred<0)] = 0
        #print('\n\nmask_pred:')
        #print(mask_pred[0,0,:,:])
        top[0].data[...] =  np.sum(mask_pred)/batchSz
        
    def backward(self, top, propagate_down, bottom):

        mask_pred = np.copy(bottom[0].data) 
        mask_pred[np.where(mask_pred<0)] = 0 
        back = 1e-5 * np.ones(bottom[0].data.shape)
        back *= mask_pred 
        batchSz = bottom[0].data.shape[0]
        #print('\ndiff: ',back[0,0,:,:])
        bottom[0].diff[...] = back
 
    def reshape(self, bottom, top):
        pass    
        
        

#Simple L1 loss layer
class L1LossLayer(caffe.Layer):
    def setup(self, bottom, top):

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self.loss_weight = layer_params['loss_weight']

        self._name_to_bottom_map = {
            'mask_gen_thres': 0, 
            'mask_ind_thres': 1 }

        # 0 means block, 1 means maintain 

        self._name_to_top_map = {
            'loss': 0 }

        assert len(bottom) == 2, 'There should be two bottom blobs'
        predShape = bottom[0].data.shape
        gtShape   = bottom[1].data.shape
        for i in range(len(predShape)):
            assert predShape[i] == gtShape[i], 'Mismatch: %d, %d' % (predShape[i], gtShape[i])
        assert bottom[0].data.squeeze().ndim == bottom[1].data.squeeze().ndim, 'Shape Mismatch'

        print("bottom[0].shape",bottom[0].shape)
        print("bottom[0].data.shape",bottom[0].data.shape)

        #Get the batchSz
        self.batchSz_ = gtShape[0]
        #Form the top
        assert len(top)==1, 'There should be only one output blob'
        top[0].reshape(1,1,1,1)


        
    def forward(self, bottom, top):
        #print("lossbottomshape:",bottom[0].data.shape,bottom[1].data.shape)
        batchSz = bottom[0].data.shape[0]
        top[0].data[...] = np.sum(np.abs(bottom[0].data[...].squeeze()\
                                                     - bottom[1].data[...].squeeze()))/float(batchSz*24) 
        #print("loss weight: ",self.loss_weight)
        #print('Loss is %f' % top[0].data[0])
        #print(bottom[0].data[...].squeeze()[0])
        #print(bottom[1].data[...].squeeze()[0])
        #print(np.sum(np.abs(bottom[0].data[...].squeeze() - bottom[1].data[...].squeeze()))/float(batchSz))
        #print("bath_Sz:")
        #print(float(self.batchSz_))

    def backward(self, top, propagate_down, bottom):
        batchSz = bottom[0].data.shape[0]
        bottom[0].diff[...] = np.sign(bottom[0].data[...].squeeze()\
                                                         - bottom[1].data[...].squeeze())/float(batchSz*24)
        bottom[1].diff[...] = np.sign(bottom[0].data[...].squeeze()\
                                                         - bottom[1].data[...].squeeze())/float(batchSz*24)
        #print("\n\n\n\nloss.diff:")
        #print(bottom[0].diff)
    def reshape(self, bottom, top):
        top[0].reshape(1,1,1,1)
        pass









