import cv2
import numpy as np
import math
import time
from scipy.ndimage.filters import gaussian_filter
import torch
from torchvision import transforms
import torch.nn.functional as F

from src import util_fast
from src.model import bodypose_model

class Body(object):
    def __init__(self, model_path):
        self.model = bodypose_model()
        
        # Determine device and optimize accordingly
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        
        # Load model weights with appropriate device mapping
        model_weights = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(model_weights, strict=False)
        self.model.eval()
        
        # Use TorchScript to optimize model if possible
        try:
            self.model = torch.jit.script(self.model)
            print("Model optimized with TorchScript")
        except Exception as e:
            print(f"Could not optimize with TorchScript: {e}")
        
        # Enable CUDA optimizations if available
        if torch.cuda.is_available():
            # Enable cudnn benchmark mode
            torch.backends.cudnn.benchmark = True
            
            # Try to use mixed precision
            try:
                if hasattr(torch.cuda, 'amp'):
                    self.amp_enabled = True
                    print("Mixed precision enabled")
                else:
                    self.amp_enabled = False
            except:
                self.amp_enabled = False
        else:
            self.amp_enabled = False
        
        # Pre-define transform for faster image processing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])
        ])
        
        # Cache for commonly used values
        self.boxsize = 368
        self.stride = 8
        self.padValue = 128
        self.thre1 = 0.1
        self.thre2 = 0.05
        
        # Simplified scale search - just use one scale factor for speed
        self.scale_search = [0.5]
        
        # Pre-compute limbSeq and mapIdx which don't change
        self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                   [1, 16], [16, 18], [3, 17], [6, 18]]
        self.mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22],
                  [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52],
                  [55, 56], [37, 38], [45, 46]]
        
        # Cache for peak detection arrays
        self._create_peak_detection_cache()
    
    def _create_peak_detection_cache(self):
        """Create cached arrays for faster peak detection"""
        self.peak_offset_x = np.array([1, -1, 0, 0])  # right, left, same, same
        self.peak_offset_y = np.array([0, 0, 1, -1])  # same, same, down, up
    
    def __call__(self, oriImg):
        """
        Original detection method, maintained for compatibility
        """
        # Use a faster approach for single-scale detection
        height, width = oriImg.shape[0], oriImg.shape[1]
        scale = self.scale_search[0] * self.boxsize / height
        
        # Resize image more efficiently
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        # Pad image
        imageToTest_padded, pad = util_fast.padRightDownCorner(imageToTest, self.stride, self.padValue)
        
        # More efficient tensor conversion
        im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
        im = np.ascontiguousarray(im)
        data = torch.from_numpy(im).float().to(self.device)
        
        # Run inference with optimizations
        if self.amp_enabled and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(data)
        else:
            with torch.no_grad():
                Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(data)
        
        # Move tensors to CPU more efficiently
        Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().numpy()
        Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().numpy()
        
        # Process heatmap
        heatmap = np.transpose(np.squeeze(Mconv7_stage6_L2), (1, 2, 0))
        heatmap = cv2.resize(heatmap, (0, 0), fx=self.stride, fy=self.stride, interpolation=cv2.INTER_LINEAR)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Process PAF
        paf = np.transpose(np.squeeze(Mconv7_stage6_L1), (1, 2, 0))
        paf = cv2.resize(paf, (0, 0), fx=self.stride, fy=self.stride, interpolation=cv2.INTER_LINEAR)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Skip the averaging since we're only using one scale
        heatmap_avg = heatmap
        paf_avg = paf
        
        all_peaks = []
        peak_counter = 0
        
        # Vectorize operations where possible
        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            # Use a smaller sigma for faster Gaussian filtering
            one_heatmap = gaussian_filter(map_ori, sigma=2)
            
            # Create shifted maps for peak detection
            map_left = np.zeros(one_heatmap.shape)
            map_left[1:, :] = one_heatmap[:-1, :]
            map_right = np.zeros(one_heatmap.shape)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = np.zeros(one_heatmap.shape)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = np.zeros(one_heatmap.shape)
            map_down[:, :-1] = one_heatmap[:, 1:]
            
            # Detect peaks with vectorized operations
            peaks_binary = np.logical_and.reduce(
                (one_heatmap >= map_left, one_heatmap >= map_right, 
                 one_heatmap >= map_up, one_heatmap >= map_down, 
                 one_heatmap > self.thre1))
            
            # Further optimize by reducing the number of peaks if there are too many
            if np.sum(peaks_binary) > 20:  # If too many peaks
                # Keep only the strongest peaks
                flat_binary = peaks_binary.flatten()
                flat_heatmap = one_heatmap.flatten()
                indices = np.argsort(-flat_heatmap[flat_binary])[:20]  # Get indices of top 20 peaks
                new_binary = np.zeros_like(flat_binary)
                new_binary[np.nonzero(flat_binary)[0][indices]] = True
                peaks_binary = new_binary.reshape(peaks_binary.shape)
            
            # Get peak coordinates
            y, x = np.nonzero(peaks_binary)
            
            # Get all peaks info in one go
            if len(x) > 0:
                scores = map_ori[y, x]
                peaks_with_score_and_id = [(x[i], y[i], scores[i], peak_counter + i) for i in range(len(x))]
                all_peaks.append(peaks_with_score_and_id)
                peak_counter += len(x)
            else:
                all_peaks.append([])
        
        # Connection all for people assembly
        connection_all = []
        special_k = []
        mid_num = 10  # Reduced from original for better performance
        
        for k in range(len(self.mapIdx)):
            score_mid = paf_avg[:, :, [x - 19 for x in self.mapIdx[k]]]
            candA = all_peaks[self.limbSeq[k][0] - 1]
            candB = all_peaks[self.limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = self.limbSeq[k]
            
            if nA == 0 or nB == 0:
                special_k.append(k)
                connection_all.append([])
                continue
                
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = max(0.001, np.linalg.norm(vec))
                    vec = vec / norm
                    
                    startend = list(zip(
                        np.linspace(candA[i][0], candB[j][0], num=mid_num).astype(int),
                        np.linspace(candA[i][1], candB[j][1], num=mid_num).astype(int)
                    ))
                    
                    vec_x = np.array([score_mid[startend[I][1], startend[I][0], 0] for I in range(len(startend))])
                    vec_y = np.array([score_mid[startend[I][1], startend[I][0], 1] for I in range(len(startend))])
                    
                    score_midpts = vec_x * vec[0] + vec_y * vec[1]
                    score_with_dist_prior = np.sum(score_midpts) / len(score_midpts) + min(0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = np.sum(score_midpts > self.thre2) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])
            
            # No connection candidates
            if len(connection_candidate) == 0:
                special_k.append(k)
                connection_all.append([])
                continue
                
            # Sort candidates by score
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            
            # Create connections
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if i not in connection[:, 3] and j not in connection[:, 4]:
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if len(connection) >= min(nA, nB):
                        break
                        
            connection_all.append(connection)
        
        # Initialize subset array
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])
        
        # Build people from connections
        for k in range(len(self.mapIdx)):
            if k in special_k:
                continue
                
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(self.limbSeq[k])