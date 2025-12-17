from event_slicer import EventSlicer
import h5py
import hdf5plugin
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import imageio

def rectify_and_write(rect_map, events_curr, events_prev, output_dir, idx):
    #rectify events
    p = events_curr['p']
    t = events_curr['t']
    x = events_curr['x']
    y = events_curr['y']
    xy_rect = rect_map[y, x]
    x_rect = xy_rect[:, 0]
    y_rect = xy_rect[:, 1]
    events1 = np.stack([x_rect, y_rect, t, p], axis=1)

    p = events_prev['p']
    t = events_prev['t']
    x = events_prev['x']
    y = events_prev['y']
    xy_rect = rect_map[y, x]
    x_rect = xy_rect[:, 0]
    y_rect = xy_rect[:, 1]
    events0 = np.stack([x_rect, y_rect, t, p], axis=1)
    #write events
    output_name = os.path.join(output_dir, '{:06d}'.format(idx))
    if os.path.exists(output_name):
        return
    np.savez(output_name, events_prev=events0, events_curr=events1)

if __name__ == '__main__':
    event_path = 'DSEC/test_events'
    ts_path = 'DSEC/test_forward_optical_flow_timestamps'

    csvfiles = os.listdir(ts_path)
    sequences = [csv.split('.')[0] for csv in csvfiles]
    for seq in sequences:
        event_h5 = os.path.join(event_path, seq, 'events/left', 'events.h5')
        rectify_h5 = os.path.join(event_path, seq, 'events/left', 'rectify_map.h5')
        
        ts_file = os.path.join(ts_path, seq+'.csv')
        ts = np.genfromtxt(ts_file, delimiter=',')
        timestamps, indexs = ts[:,:2], ts[:,2]
        assert len(timestamps) == len(indexs)
        h5f = h5py.File(event_h5, 'r')
        slicer = EventSlicer(h5f)
        with h5py.File(rectify_h5, 'r') as h5_rect:
            rectify_map = h5_rect['rectify_map'][()]


        output_dir = os.path.join('datasets/dsec_full/test', seq)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in tqdm(range(len(timestamps)), ncols=60):
            #current
            t_curr = timestamps[i,0]
            t_next = timestamps[i,1]
            events_curr = slicer.get_events(t_curr, t_next)
            if events_curr == None:
                print(f'None data can be converted to voxel in {seq} at {i}th timestamps for current condition!')
                continue

            # previous 
            dt = 100 * 1000#us
            t_prev = t_curr - dt
            events_prev = slicer.get_events(t_prev, t_curr)
            if events_prev == None:
                print(f'None data can be converted to voxel in {seq} at {i}th timestamps for previous condition!')
                continue
            rectify_and_write(rectify_map, events_curr, events_prev, output_dir, int(indexs[i]))

        h5f.close()