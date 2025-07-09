import cv2
import numpy as np
import math
import time
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
import torch
from torchvision import transforms

from src import util_fast
from src.model import bodypose_model

class Body(object):
    def __init__(self, model_path):
        self.model = bodypose_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        model_weights = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.load_state_dict(model_weights, strict=False)
        self.model.eval()
        
        # Add these optimizations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Use torch.jit to optimize model if possible
        try:
            self.model = torch.jit.script(self.model)
            print("Model optimized with TorchScript")
        except Exception as e:
            print(f"Could not optimize with TorchScript: {e}")
        
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
        
        # Simplified scale search - just use one scale factor
        self.scale_search = [0.5]  # Reduced from multiple scales
        
        # Pre-compute limbSeq and mapIdx which don't change
        self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                   [1, 16], [16, 18], [3, 17], [6, 18]]
        self.mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22],
                  [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52],
                  [55, 56], [37, 38], [45, 46]]

    def __call__(self, oriImg):
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
            
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]
            
            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)
        
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
            if (nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        norm = max(0.001, norm)
                        vec = np.divide(vec, norm)

                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                            np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                          for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * oriImg.shape[0] / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_midpts > self.thre2)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if (len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(self.mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(self.limbSeq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
        
        # delete some rows of subset which has few parts occur
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        return candidate, subset
        
    def fast_inference(self, oriImg):
        """Faster inference with more aggressive optimizations"""
        height, width = oriImg.shape[0], oriImg.shape[1]
        
        # Use an even smaller scale for faster processing
        scale = 0.25 * self.boxsize / height
        
        # Resize to smaller size for faster processing
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        # Pad image
        imageToTest_padded, pad = util_fast.padRightDownCorner(imageToTest, self.stride, self.padValue)
        
        # More efficient tensor conversion
        im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
        im = np.ascontiguousarray(im)
        data = torch.from_numpy(im).float().to(self.device)
        
        # Run inference
        with torch.no_grad():
            Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(data)
            
        # Process outputs
        Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().numpy()
        Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().numpy()
        
        # Process heatmap more efficiently
        heatmap = np.transpose(np.squeeze(Mconv7_stage6_L2), (1, 2, 0))
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Process PAF more efficiently
        paf = np.transpose(np.squeeze(Mconv7_stage6_L1), (1, 2, 0))
        paf = cv2.resize(paf, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Simplified peak detection with higher threshold
        all_peaks = []
        peak_counter = 0
        
        for part in range(18):
            map_ori = heatmap[:, :, part]
            
            # Skip gaussian filter for speed
            one_heatmap = map_ori
            
            # Only keep very confident peaks
            peaks_binary = one_heatmap > self.thre1 * 1.5
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
            
            # Limit number of peaks
            if len(peaks) > 10:
                peak_values = [one_heatmap[p[1], p[0]] for p in peaks]
                sorted_indices = np.argsort(peak_values)
                peaks = [peaks[i] for i in sorted_indices[-10:]]
            
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]
            
            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)
        
        # Simplified connection logic - focus only on major connections
        connection_all = []
        special_k = []
        
        # Create a simplified subset for speed
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])
        
        # If we have candidates, create at least one person
        if len(candidate) > 0:
            # Create a simple subset focusing on important parts (like arms)
            row = -1 * np.ones(20)
            
            # Try to find elbow and wrist keypoints (important for the ball game)
            arm_parts = [3, 4]  # elbow and wrist indices
            for part_idx in arm_parts:
                if len(all_peaks[part_idx - 1]) > 0:
                    # Use the highest scoring detection for this part
                    best_detection = max(all_peaks[part_idx - 1], key=lambda x: x[2])
                    row[part_idx - 1] = best_detection[3]  # Store the peak ID
                    
            # Count how many parts we found
            parts_found = np.sum(row[:-2] >= 0)
            if parts_found > 0:
                row[-1] = parts_found
                row[-2] = sum([candidate[int(idx), 2] for idx in row[:-2] if idx >= 0])
                subset = np.vstack([subset, row])
        
        return candidate, subset

if __name__ == "__main__":
    body_estimation = Body('../model/body_pose_model.pth')

    test_image = '../images/ski.jpg'
    oriImg = cv2.imread(test_image)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    canvas = util_fast.draw_bodypose(oriImg, candidate, subset)
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.show()
